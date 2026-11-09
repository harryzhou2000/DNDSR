#include "CFV/Detail/ReconstructionData.hpp"
#include "Geom/Quadratures/Quadrature.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"

namespace DNDS::CFV
{
    namespace R = Runtime;
    using namespace Geom;
    using namespace Geom::Elem;
    using Math::SmallMatrix;
    using Matrix = Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>;

    namespace
    {
        /// Only requested tables acquire writers. An omitted table has no Array
        /// allocation or lease, including on ranks with zero owned cells.
        struct OptionalTableWriter
        {
            std::optional<R::Array<real>::WriteLease> lease;
            R::ArrayView<real, false, R::TableRows> values;
            explicit OptionalTableWriter(std::optional<R::Array<real>> &array)
            {
                if (array)
                {
                    lease.emplace(array->Write());
                    values = lease->View<R::TableRows>();
                }
            }
        };
        /// Face scratch dependencies. Full support retains the original slot
        /// order; sweep-only setup omits the two secondary change-of-base maps.
        struct FaceSystemLayout
        {
            int a = -1, b = -1, mean = -1, secondary = -1, boundaryWeight = -1, width = 0;
            FaceSystemLayout(int n, const ReconstructionCapabilities &capabilities)
            {
                if (capabilities.sweeps || capabilities.linearSystem || capabilities.localPreconditioner || capabilities.inverseDiagnostics)
                {
                    a = width;
                    width += 2 * n * n;
                    // A's self-periodic diagonal subtracts B even for an
                    // inverse-only profile; it cannot be formed from A alone.
                    b = width;
                    width += n * n;
                }
                if (capabilities.sweeps || capabilities.linearSystem)
                {
                    mean = width;
                    width += 2 * n;
                }
                if (capabilities.wbap)
                {
                    secondary = width;
                    width += 2 * n * n;
                }
                if (capabilities.sweeps || capabilities.linearSystem)
                    boundaryWeight = width++;
            }
        };
    }

    VariationalReconstructionPlan VariationalReconstructionPlan::ConstructRecCoeff(FiniteVolumePlan metrics, ReconstructionSettings settings,
                                                                                   ReconstructionCapabilities capabilities)
    {
        auto geometry = metrics._data;
        // Capability differences change payload construction and diagnostics
        // collectives. This is required protocol agreement, not optional safety.
        int modes[]{capabilities.Mask(), -capabilities.Mask()};
        MPI_Allreduce(MPI_IN_PLACE, modes, 2, MPI_INT, MPI_MIN, geometry->communication->Communicator());
        DNDS_check_throw_info(modes[0] == -modes[1], "VR preparation capabilities differ between ranks");
        std::shared_ptr<Detail::ReconstructionData> data;
        geometry->communication->LocalStage([&]
                                            {
            metrics.Validate();
            settings.Validate();
            DNDS_check_throw_info(settings.intOrder == geometry->intOrder, "VR quadrature differs from prepared metrics");
            auto meshRead = geometry->mesh->AcquireRead(MeshReadRequirements::CellFlux());
            const auto mesh = meshRead.HostView();
            data = std::make_shared<Detail::ReconstructionData>(metrics, settings, capabilities);
            const bool needInverse = capabilities.sweeps || capabilities.localPreconditioner || capabilities.inverseDiagnostics;
            data->workPartition.emplace(metrics.CellWorkPartition());
            const int n = data->n, dim = geometry->dim, degree = settings.maxOrder;
            const Math::TaylorBasis polynomial(dim, degree);
            const FaceSystemLayout faceSystem(n, capabilities);
            const index nCell = mesh.NumCell(), nFace = mesh.NumFaceProc();
            data->coreFirstPosition.resize(nCell);
            index position = 0;
            for (auto iCell : data->workPartition->Core())
                data->coreFirstPosition[iCell] = position++;
            for (auto iCell : data->workPartition->HaloDependent())
                data->coreFirstPosition[iCell] = position++;
            DNDS_check_throw_info(position == nCell, "reconstruction work partition does not cover owned cells");
            auto cellRead = geometry->cell.Read();
            const auto cell = cellRead.View<R::FixedRows<8>>();
            auto faceOffsetRead = geometry->faceOffsets.Read();
            const auto faceOffset = faceOffsetRead.View<R::FixedRows<1>>();
            auto pointRead = geometry->faceQuadrature.Read();
            const auto point = pointRead.View<R::FixedRows<7>>();
            auto originalRead = geometry->mesh->CurrentNativeState()->ReadOriginalEntityIds(EntityKind::Cell);
            const auto original = originalRead->View<R::FixedRows<1>>();

            // Count all incidence and boundary rows before allocating payloads.
            index incidences = 0, boundaryCount = 0;
            std::vector<uint8_t> active(size_t(nFace), 0); // local faces only
            for (index iCell = 0; iCell < nCell; ++iCell)
            {
                const auto faces = mesh.CellFaces(iCell);
                incidences += faces.size();
                for (index slot = 0; slot < faces.size(); ++slot)
                {
                    const index iFace = faces[slot];
                    active[size_t(iFace)] = 1;
                    if (mesh.CellFaceOther(iCell, iFace, rowsize(slot)) == UnInitIndex)
                        boundaryCount += faceOffset(iFace + 1, 0) - faceOffset(iFace, 0);
                }
            }
            data->moments = R::Array<real>(R::RowLayout::Dense(mesh.NumCellProc(), n + 1));
            data->offsets = R::Array<index>(R::RowLayout::Dense(nCell + 1, 1));
            data->faces = R::Array<index>(R::RowLayout::Dense(incidences, 1));
            data->neighbors = R::Array<index>(R::RowLayout::Dense(incidences, 1));
            data->boundaryOffsets = R::Array<index>(R::RowLayout::Dense(incidences + 1, 1));
            // Central operation-to-storage dependency table. Do not construct
            // full tables and discard them after setup: that retains peak cost.
            if (capabilities.sweeps)
            {
                data->matrixAInvB.emplace(R::RowLayout::Dense(incidences, n * n));
                data->vectorAInvB.emplace(R::RowLayout::Dense(incidences, n));
                data->boundaryInjection.emplace(R::RowLayout::Dense(boundaryCount, n));
            }
            if (capabilities.wbap)
                data->secondary.emplace(R::RowLayout::Dense(incidences, n * n));
            if (capabilities.linearSystem)
            {
                data->localMatrix.emplace(R::RowLayout::Dense(nCell, n * n));
                data->rawMatrixB.emplace(R::RowLayout::Dense(incidences, n * n));
                data->rawVectorB.emplace(R::RowLayout::Dense(incidences, n));
                data->rawBoundaryInjection.emplace(R::RowLayout::Dense(boundaryCount, n));
            }
            if (capabilities.localPreconditioner)
                data->localInverse.emplace(R::RowLayout::Dense(nCell, n * n));
            if (capabilities.inverseDiagnostics)
                data->inverseConditions.emplace(R::RowLayout::Dense(nCell, 4));
            data->neighborRotation = R::Array<real>(R::RowLayout::Dense(incidences, 9));
            auto rotationWrite = data->neighborRotation.Write();
            const auto rotations = rotationWrite.View<R::FixedRows<9>>();
            OptionalTableWriter rawMatrixWrite(data->rawMatrixB), rawVectorWrite(data->rawVectorB), rawInjectionWrite(data->rawBoundaryInjection);
            const auto rawMatrices = rawMatrixWrite.values, rawVectors = rawVectorWrite.values, rawInjections = rawInjectionWrite.values;
            OptionalTableWriter localMatrixWrite(data->localMatrix), localInverseWrite(data->localInverse);
            const auto localMatrices = localMatrixWrite.values, localInverses = localInverseWrite.values;
            data->boundaryPoints = R::Array<index>(R::RowLayout::Dense(boundaryCount, 4));
            if (settings.cacheDiffBase)
                data->faceBasis = R::Array<real>(R::RowLayout::Dense(2 * point.RowCount(), data->cachedDerivatives * n));
            std::optional<R::Array<real>::WriteLease> cacheWrite;
            R::ArrayView<real, false, R::TableRows> cache;
            if (settings.cacheDiffBase)
            {
                cacheWrite.emplace(data->faceBasis.Write());
                cache = cacheWrite->View<R::TableRows>();
                for (index iRow = 0; iRow < cache.RowCount(); ++iRow)
                    std::fill(cache[iRow].begin(), cache[iRow].end(), 0);
            }
            // Only requested face-system blocks exist. This is preparation
            // scratch, released before returning the immutable plan.
            std::optional<R::Array<real>> systems;
            if (faceSystem.width)
            {
                systems.emplace(R::RowLayout::Dense(nFace, faceSystem.width));
                data->preparationRuntimeScratchBytes = systems->Storage().Bytes();
            }
            auto momentWrite = data->moments.Write();
            const auto moment = momentWrite.View<R::TableRows>();
            SmallMatrix full, left, right, weighted, aL, aR, bLR, secondaryL, secondaryR;
            tSmallCoords coords;
            std::vector<tPoint> halfBoxes(size_t(mesh.NumCellProc()));
            for (index iCell = 0; iCell < mesh.NumCellProc(); ++iCell)
            {
                std::fill(moment[iCell].begin(), moment[iCell].end(), 0.);
                mesh.GetCoordsOnCell(iCell, coords);
                const tPoint bary{cell(iCell, 4), cell(iCell, 5), cell(iCell, 6)};
                halfBoxes[size_t(iCell)] = (coords.colwise() - bary).array().abs().rowwise().maxCoeff();
                const auto element = mesh.GetCellElement(iCell);
                const Quadrature q(element, INT_ORDER_MAX);
                const tPoint center{cell(iCell, 1), cell(iCell, 2), cell(iCell, 3)};
                for (int iG = 0; iG < q.GetNumPoints(); ++iG)
                {
                    const auto &shape = Geom::Elem::GetCachedShapeAtQuadrature(element.type, INT_ORDER_MAX, iG);
                    const tPoint p = PPhysicsCoordD01Nj(coords, shape);
                    polynomial.Derivatives(full, p, center, cell(iCell, 7));
                    const real weight = CellJacobianDet(dim, coords, shape) * q.GetWeight(iG) * (geometry->axisymmetric ? p[1] : 1.);
                    for (int j = 0; j <= n; ++j) moment(iCell, j) += full(0, j) * weight;
                }
                const real volume = moment(iCell, 0);
                for (int j = 0; j <= n; ++j) moment(iCell, j) /= volume;
            }
            OptionalTableWriter systemWrite(systems);
            const auto system = systemWrite.values;
            auto basis = [&](index iCell, const tPoint &p, SmallMatrix &out)
            {
                polynomial.Reconstruction(out, p, {cell(iCell, 1), cell(iCell, 2), cell(iCell, 3)}, cell(iCell, 7), moment[iCell], degree);
            };
            for (index iFace = 0; iFace < nFace; ++iFace)
            {
                if (!active[size_t(iFace)]) continue;
                if (!faceSystem.width && !settings.cacheDiffBase) continue;
                if (systems) std::fill(system[iFace].begin(), system[iFace].end(), 0.);
                const auto parents = mesh.FaceCells(iFace);
                const index iL = parents[0], iR = parents[1];
                DNDS_check_throw_info(iL >= 0 && iL < mesh.NumCellProc() &&
                                          (iR == UnInitIndex || (iR >= 0 && iR < mesh.NumCellProc())), "VR stencil lacks face parent closure");
                mesh.GetCoordsOnFace(iFace, coords);
                const tPoint midpoint = PPhysicsCoordD01Nj(coords, Geom::Elem::GetCachedShapeAtQuadrature(mesh.GetFaceElement(iFace).type, 1, 0));
                const auto jac = ShapeJacobianCoordD01Nj(coords, Geom::Elem::GetCachedShapeAtQuadrature(mesh.GetFaceElement(iFace).type, 1, 0));
                tPoint normal = dim == 2 ? FacialJacobianToNormVec<2>(jac) : FacialJacobianToNormVec<3>(jac);
                normal.normalize();
                const tPoint baryL{cell(iL, 4), cell(iL, 5), cell(iL, 6)};
                const real distance = iR == UnInitIndex ? std::abs(2 * (midpoint - baryL).dot(normal))
                    : (Detail::OtherPointInCell(mesh, iFace, 0, {cell(iR, 4), cell(iR, 5), cell(iR, 6)}) - baryL).norm();
                DNDS_check_throw_info(distance > 0 && std::isfinite(distance), "VR face has zero barycenter distance");
                real area = 0;
                for (index iQ = faceOffset(iFace, 0); iQ < faceOffset(iFace + 1, 0); ++iQ) area += point(iQ, 6);
                if (geometry->axisymmetric)
                {
                    // Weight policies use physical face length, not r*length.
                    // In particular an axis face has zero radial measure but
                    // nonzero geometric length; no division by radius occurs.
                    area = 0;
                    const auto element = mesh.GetFaceElement(iFace);
                    const Quadrature quadrature(element, geometry->intOrder);
                    for (int iG = 0; iG < quadrature.GetNumPoints(); ++iG)
                    {
                        const auto &shape = Geom::Elem::GetCachedShapeAtQuadrature(element.type, geometry->intOrder, iG);
                        area += FacialJacobianToNormVec<2>(ShapeJacobianCoordD01Nj(coords, shape)).norm() * quadrature.GetWeight(iG);
                    }
                }
                auto functional = settings.functional;
                if (settings.HQMWeights)
                {
                    functional.dirWeightScheme = ReconstructionFunctionalSettings::DirectionalWeight::HQM_OPT;
                    functional.geomWeightScheme = ReconstructionFunctionalSettings::GeometricWeight::HQM_SD;
                    functional.geomWeightBias = 1;
                }
                // Tangential face spans for TEST_OPT; no cell-anisotropy policy
                // is implied by these surface-only principal directions.
                tPoint faceSpan = tPoint::Zero();
                if (dim == 3 && degree == 3 && functional.dirWeightScheme == ReconstructionFunctionalSettings::DirectionalWeight::TEST_OPT)
                {
                    faceSpan = Geom::Elem::GetElemNodeMajorSpan(coords);
                }
                auto weights = Math::ReconstructionOrderWeights(functional, dim, degree, area, distance, faceSpan);
                if (iR == UnInitIndex)
                {
                    const auto override = settings.boundaryZoneWeights.find(mesh.GetFaceZone(iFace));
                    const auto &boundaryWeights = override == settings.boundaryZoneWeights.end() ? settings.boundaryDerivativeWeights : override->second;
                    for (int order = 0; order <= degree; ++order)
                        weights[order] *= settings.bcWeight * boundaryWeights[order];
                }
                real diagonal[20];
                real length = distance;
                if (functional.scaleType == ReconstructionFunctionalSettings::LengthScale::MeanAACBB)
                {
                    const tPoint box = iR == UnInitIndex ? halfBoxes[size_t(iL)] : (.5 * (halfBoxes[size_t(iL)] + halfBoxes[size_t(iR)])).eval();
                    length = std::sqrt(box.head(dim).squaredNorm() / dim);
                }
                if (functional.scaleType == ReconstructionFunctionalSettings::LengthScale::CellMax)
                    length = iR == UnInitIndex ? cell(iL, 7) : std::max(cell(iL, 7), cell(iR, 7));
                Math::FunctionalWeights(diagonal, dim, degree, length * functional.scaleMultiplier, weights.data(), functional.dirWeightCombPowV);
                const tGPoint rightToLeft = Detail::OtherFrameInCell(mesh, iFace, 0);
                const bool rotated = !rightToLeft.isApprox(tGPoint::Identity(), 1e-14);
                DNDS_check_throw_info(faceSystem.a < 0 || !rotated || functional.dirWeightCombPowV == 1,
                                      "rotated periodic VR requires the Frobenius derivative metric");
                if (faceSystem.boundaryWeight >= 0) system(iFace, faceSystem.boundaryWeight) = diagonal[0];
                if (faceSystem.a >= 0) aL.setZero(n, n), aR.setZero(n, n);
                if (faceSystem.b >= 0) bLR.setZero(n, n);
                Eigen::Map<Eigen::VectorXd> bL(faceSystem.mean >= 0 ? system[iFace].data() + faceSystem.mean : nullptr, faceSystem.mean >= 0 ? n : 0);
                Eigen::Map<Eigen::VectorXd> bR(faceSystem.mean >= 0 ? system[iFace].data() + faceSystem.mean + n : nullptr, faceSystem.mean >= 0 ? n : 0);
                for (index iQ = faceOffset(iFace, 0); iQ < faceOffset(iFace + 1, 0); ++iQ)
                {
                    const tPoint p{point(iQ, 0), point(iQ, 1), point(iQ, 2)};
                    basis(iL, p, left);
                    if (settings.cacheDiffBase)
                        Eigen::Map<Matrix>(cache[2 * iQ].data(), data->cachedDerivatives, n) = left.topRows(data->cachedDerivatives);
                    if (faceSystem.a >= 0)
                    {
                        weighted = left;
                        for (int j = 0; j <= n; ++j) weighted.row(j) *= diagonal[j] * point(iQ, 6);
                        aL.noalias() += left.transpose() * weighted;
                    }
                    if (faceSystem.mean >= 0)
                        bL.noalias() += left.row(0).transpose() * (diagonal[0] * point(iQ, 6));
                    if (iR != UnInitIndex)
                    {
                        basis(iR, Detail::FacePointInCell(mesh, iFace, 1, p), right);
                        if (settings.cacheDiffBase)
                            Eigen::Map<Matrix>(cache[2 * iQ + 1].data(), data->cachedDerivatives, n) = right.topRows(data->cachedDerivatives);
                        if (rotated) polynomial.Pullback(right, rightToLeft.transpose());
                        if (faceSystem.b >= 0) bLR.noalias() += weighted.transpose() * right;
                        if (faceSystem.a >= 0)
                        {
                            weighted = right;
                            for (int j = 0; j <= n; ++j) weighted.row(j) *= diagonal[j] * point(iQ, 6);
                            aR.noalias() += right.transpose() * weighted;
                        }
                        if (faceSystem.mean >= 0)
                            bR.noalias() += right.row(0).transpose() * (diagonal[0] * point(iQ, 6));
                    }
                }
                if (faceSystem.a >= 0)
                {
                    Eigen::Map<Matrix>(system[iFace].data() + faceSystem.a, n, n) = aL;
                    Eigen::Map<Matrix>(system[iFace].data() + faceSystem.a + n * n, n, n) = aR;
                }
                if (faceSystem.b >= 0)
                    Eigen::Map<Matrix>(system[iFace].data() + faceSystem.b, n, n) = bLR;
                if (capabilities.wbap && iR != UnInitIndex)
                {
                    const tPoint centerL{cell(iL, 1), cell(iL, 2), cell(iL, 3)}, centerR{cell(iR, 1), cell(iR, 2), cell(iR, 3)};
                    basis(iR, Detail::OtherPointInCell(mesh, iFace, 1, centerL), right);
                    if (rotated) polynomial.Pullback(right, rightToLeft.transpose());
                    basis(iL, Detail::OtherPointInCell(mesh, iFace, 0, centerR), left);
                    if (rotated) polynomial.Pullback(left, rightToLeft);
                    secondaryL = right.bottomRows(n), secondaryR = left.bottomRows(n);
                    // Exact Taylor change of center/scale/frame, not a local
                    // numerical inversion: c_alpha=s^|alpha|*D^alpha/alpha!.
                    // No spectral cutoff belongs to this coordinate map.
                    for (int mode = 1; mode <= n; ++mode)
                    {
                        const auto &alpha = polynomial.MultiIndex(mode);
                        const real factorial = Geom::Base::factorials[alpha[0]] * Geom::Base::factorials[alpha[1]] * Geom::Base::factorials[alpha[2]];
                        secondaryL.row(mode - 1) *= std::pow(cell(iL, 7), polynomial.Degree(mode)) / factorial;
                        secondaryR.row(mode - 1) *= std::pow(cell(iR, 7), polynomial.Degree(mode)) / factorial;
                    }
                    Eigen::Map<Matrix>(system[iFace].data() + faceSystem.secondary, n, n) = secondaryL;
                    Eigen::Map<Matrix>(system[iFace].data() + faceSystem.secondary + n * n, n, n) = secondaryR;
                }
            }
            auto offsetsWrite = data->offsets.Write(), neighborsWrite = data->neighbors.Write(), facesWrite = data->faces.Write();
            auto bOffsetsWrite = data->boundaryOffsets.Write(), bPointsWrite = data->boundaryPoints.Write();
            OptionalTableWriter matrixWrite(data->matrixAInvB), vectorWrite(data->vectorAInvB), secondaryWrite(data->secondary), injectionWrite(data->boundaryInjection);
            const auto offsets = offsetsWrite.View<R::FixedRows<1>>(), neighbors = neighborsWrite.View<R::FixedRows<1>>(), faces = facesWrite.View<R::FixedRows<1>>();
            const auto bOffsets = bOffsetsWrite.View<R::FixedRows<1>>();
            const auto bPoints = bPointsWrite.View<R::FixedRows<4>>();
            const auto matrices = matrixWrite.values, vectors = vectorWrite.values, secondaries = secondaryWrite.values, injections = injectionWrite.values;
            index iEntry = 0, iBoundary = 0;
            OptionalTableWriter conditionsWrite(data->inverseConditions);
            const auto conditions = conditionsWrite.values;
            data->inverseDiagnostics.cells = nCell;
            SmallMatrix A, inverse, product, rawMatrixScratch(n, n);
            Eigen::VectorXd rawVectorScratch(n);
            for (index iCell = 0; iCell < nCell; ++iCell)
            {
                offsets(iCell, 0) = iEntry;
                A.setZero(n, n);
                const auto c2f = mesh.CellFaces(iCell);
                for (index slot = 0; faceSystem.a >= 0 && slot < c2f.size(); ++slot)
                {
                    const index iFace = c2f[slot];
                    const int side = mesh.CellIsFaceBack(iCell, iFace, rowsize(slot)) ? 0 : 1;
                    A += Eigen::Map<const Matrix>(system[iFace].data() + faceSystem.a + side * n * n, n, n);
                    if (mesh.CellFaceOther(iCell, iFace, rowsize(slot)) == iCell &&
                        Detail::OtherFrameInCell(mesh, iFace, side).isApprox(tGPoint::Identity(), 1e-14))
                    {
                        const Eigen::Map<const Matrix> B(system[iFace].data() + faceSystem.b, n, n);
                        if (side == 0) A -= B; else A -= B.transpose();
                    }
                }
                HardEigen::SVDInverseDiagnostics spectrum{};
                if (needInverse)
                    spectrum = HardEigen::EigenSVDInverse(A, inverse, settings.svdTolerance, settings.svdAbsoluteTolerance);
                if (capabilities.linearSystem)
                    Eigen::Map<Matrix>(localMatrices[iCell].data(), n, n) = A;
                if (capabilities.localPreconditioner)
                    Eigen::Map<Matrix>(localInverses[iCell].data(), n, n) = inverse;
                if (capabilities.inverseDiagnostics)
                {
                    auto &summary = data->inverseDiagnostics;
                    summary.filteredCells += spectrum.rank < n;
                    summary.discardedModes += n - spectrum.rank;
                    summary.rankZeroCells += spectrum.rank == 0;
                    ++summary.rankCounts[size_t(spectrum.rank)];
                    if (spectrum.rawCondition > summary.maxCondition ||
                        (spectrum.rawCondition == summary.maxCondition && original(iCell, 0) < summary.maxConditionCell))
                    {
                        summary.maxCondition = spectrum.rawCondition;
                        summary.maxConditionCell = original(iCell, 0);
                    }
                    conditions(iCell, 0) = spectrum.rank;
                    conditions(iCell, 1) = spectrum.rawCondition;
                    conditions(iCell, 2) = spectrum.retainedCondition;
                    conditions(iCell, 3) = spectrum.relativeCutoff;
                }
                for (index slot = 0; slot < c2f.size(); ++slot, ++iEntry)
                {
                    const index iFace = c2f[slot], other = mesh.CellFaceOther(iCell, iFace, rowsize(slot));
                    const int side = mesh.CellIsFaceBack(iCell, iFace, rowsize(slot)) ? 0 : 1;
                    faces(iEntry, 0) = iFace, neighbors(iEntry, 0) = other;
                    const tGPoint rotation = Detail::OtherFrameInCell(mesh, iFace, side);
                    Eigen::Map<tGPoint>(rotations[iEntry].data()) = rotation;
                    data->hasRotatedPeriodicity |= !rotation.isApprox(tGPoint::Identity(), 1e-14);
                    bOffsets(iEntry, 0) = iBoundary;
                    if (capabilities.wbap)
                    {
                        auto secondary = Eigen::Map<Matrix>(secondaries[iEntry].data(), n, n);
                        secondary.setZero();
                        if (other != UnInitIndex)
                            secondary = Eigen::Map<const Matrix>(system[iFace].data() + faceSystem.secondary + side * n * n, n, n);
                    }
                    if (capabilities.sweeps || capabilities.linearSystem)
                    {
                        // Retain originals only when requested. A bounded local
                        // scratch block gives sweeps the same Map/product shape;
                        // it is never reconstructed from a truncated inverse.
                        auto rawM = Eigen::Map<Matrix>(capabilities.linearSystem ? rawMatrices[iEntry].data() : rawMatrixScratch.data(), n, n);
                        auto rawB = Eigen::Map<Eigen::VectorXd>(capabilities.linearSystem ? rawVectors[iEntry].data() : rawVectorScratch.data(), n);
                        rawM.setZero(), rawB.setZero();
                        if (other != UnInitIndex)
                        {
                            const Eigen::Map<const Matrix> B(system[iFace].data() + faceSystem.b, n, n);
                            if (other != iCell || !rotation.isApprox(tGPoint::Identity(), 1e-14))
                            {
                                if (side == 0) rawM = B; else rawM = B.transpose();
                            }
                            rawB = Eigen::Map<const Eigen::VectorXd>(system[iFace].data() + faceSystem.mean + side * n, n);
                        }
                        if (capabilities.sweeps)
                        {
                            auto M = Eigen::Map<Matrix>(matrices[iEntry].data(), n, n);
                            auto b = Eigen::Map<Eigen::VectorXd>(vectors[iEntry].data(), n);
                            M.setZero(), b.setZero();
                            if (other != UnInitIndex)
                            {
                                if (other != iCell || !rotation.isApprox(tGPoint::Identity(), 1e-14))
                                    M.noalias() = inverse * rawM;
                                b.noalias() = inverse * rawB;
                            }
                        }
                    }
                    if (other == UnInitIndex)
                        for (index iQ = faceOffset(iFace, 0); iQ < faceOffset(iFace + 1, 0); ++iQ, ++iBoundary)
                        {
                            if (capabilities.sweeps || capabilities.linearSystem)
                            {
                                basis(iCell, {point(iQ, 0), point(iQ, 1), point(iQ, 2)}, left);
                                auto rawInjection = Eigen::Map<Eigen::VectorXd>(capabilities.linearSystem ? rawInjections[iBoundary].data() : rawVectorScratch.data(), n);
                                rawInjection = left.row(0).transpose() * (system(iFace, faceSystem.boundaryWeight) * point(iQ, 6));
                                if (capabilities.sweeps)
                                    Eigen::Map<Eigen::VectorXd>(injections[iBoundary].data(), n).noalias() = inverse * rawInjection;
                            }
                            bPoints(iBoundary, 0) = iCell, bPoints(iBoundary, 1) = iFace;
                            bPoints(iBoundary, 2) = iQ - faceOffset(iFace, 0), bPoints(iBoundary, 3) = iQ;
                        }
                }
            }
            offsets(nCell, 0) = iEntry, bOffsets(iEntry, 0) = iBoundary; });
        if (capabilities.inverseDiagnostics)
        {
            auto &summary = data->inverseDiagnostics;
            index local[]{summary.cells, summary.filteredCells, summary.discardedModes, summary.rankZeroCells}, global[4]{};
            MPI::Allreduce(local, global, 4, DNDS_MPI_INDEX, MPI_SUM, geometry->mesh->getMPI().comm);
            real condition = 0;
            MPI::Allreduce(&summary.maxCondition, &condition, 1, DNDS_MPI_REAL, MPI_MAX, geometry->mesh->getMPI().comm);
            const index localCell = summary.maxCondition == condition && summary.maxConditionCell != UnInitIndex ? summary.maxConditionCell : INT64_MAX;
            index globalCell = INT64_MAX;
            MPI::Allreduce(&localCell, &globalCell, 1, DNDS_MPI_INDEX, MPI_MIN, geometry->mesh->getMPI().comm);
            if (globalCell == INT64_MAX)
                globalCell = UnInitIndex;
            std::array<index, 20> ranks{};
            MPI::Allreduce(summary.rankCounts.data(), ranks.data(), int(ranks.size()), DNDS_MPI_INDEX, MPI_SUM, geometry->mesh->getMPI().comm);
            summary = {global[0], global[1], global[2], global[3], condition, globalCell, ranks};
        }
        return VariationalReconstructionPlan(std::move(data));
    }
}
