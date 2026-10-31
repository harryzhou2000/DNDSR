#include "CFV/Plans/ReconstructionData.hpp"
#include "Geom/Quadrature.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"

namespace DNDS::CFV
{
    namespace R = Runtime;
    using namespace Geom;
    using namespace Geom::Elem;
    using Math::SmallMatrix;
    using Matrix = Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>;

    VariationalReconstructionPlan VariationalReconstructionPlan::ConstructRecCoeff(FiniteVolumePlan metrics, ReconstructionSettings settings)
    {
        auto geometry = metrics._data;
        std::shared_ptr<Detail::ReconstructionData> data;
        MPI::LocalStage(geometry->mesh->getMPI(), [&]
                        {
            metrics.Validate();
            settings.Validate();
            DNDS_check_throw_info(settings.intOrder == geometry->intOrder, "VR quadrature differs from prepared metrics");
            auto meshRead = geometry->mesh->AcquireRead(MeshReadRequirements::CellFlux());
            const auto mesh = meshRead.HostView();
            data = std::make_shared<Detail::ReconstructionData>(metrics, settings);
            data->workPartition.emplace(Geom::EntityWorkPartition::CellNodeCell(*geometry->mesh));
            const int n = data->n, dim = geometry->dim, degree = settings.maxOrder;
            const Math::TaylorBasis polynomial(dim, degree);
            const index nCell = mesh.NumCell(), nFace = mesh.NumFaceProc();
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
            data->matrixAInvB = R::Array<real>(R::RowLayout::Dense(incidences, n * n));
            data->vectorAInvB = R::Array<real>(R::RowLayout::Dense(incidences, n));
            data->secondary = R::Array<real>(R::RowLayout::Dense(incidences, n * n));
            data->localMatrix = R::Array<real>(R::RowLayout::Dense(nCell, n * n));
            data->localInverse = R::Array<real>(R::RowLayout::Dense(nCell, n * n));
            data->rawMatrixB = R::Array<real>(R::RowLayout::Dense(incidences, n * n));
            data->rawVectorB = R::Array<real>(R::RowLayout::Dense(incidences, n));
            data->rawBoundaryInjection = R::Array<real>(R::RowLayout::Dense(boundaryCount, n));
            data->neighborRotation = R::Array<real>(R::RowLayout::Dense(incidences, 9));
            auto rotationWrite = data->neighborRotation.Write();
            const auto rotations = rotationWrite.View<R::FixedRows<9>>();
            auto rawMatrixWrite = data->rawMatrixB.Write(), rawVectorWrite = data->rawVectorB.Write(), rawInjectionWrite = data->rawBoundaryInjection.Write();
            const auto rawMatrices = rawMatrixWrite.View<R::TableRows>(), rawVectors = rawVectorWrite.View<R::TableRows>(), rawInjections = rawInjectionWrite.View<R::TableRows>();
            auto localMatrixWrite = data->localMatrix.Write(), localInverseWrite = data->localInverse.Write();
            const auto localMatrices = localMatrixWrite.View<R::TableRows>(), localInverses = localInverseWrite.View<R::TableRows>();
            data->boundaryInjection = R::Array<real>(R::RowLayout::Dense(boundaryCount, n));
            data->boundaryPoints = R::Array<index>(R::RowLayout::Dense(boundaryCount, 4));
            if (settings.cacheDiffBase)
                data->faceBasis = R::Array<real>(R::RowLayout::Dense(2 * point.RowCount(), (n + 1) * n));
            std::optional<R::Array<real>::WriteLease> cacheWrite;
            R::ArrayView<real, false, R::TableRows> cache;
            if (settings.cacheDiffBase)
            {
                cacheWrite.emplace(data->faceBasis.Write());
                cache = cacheWrite->View<R::TableRows>();
                for (index iRow = 0; iRow < cache.RowCount(); ++iRow)
                    std::fill(cache[iRow].begin(), cache[iRow].end(), 0);
            }
            // Face-local setup scratch: A_left, A_right, B_left-right, b_left/right
            // and two secondary maps. No dynamic per-face/per-cell row growth.
            R::Array<real> systems(R::RowLayout::Dense(nFace, 5 * n * n + 2 * n + 1));
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
                    const auto &shape = Geom::Elem::detail::NBufferAtQuadrature.buf.at(element.type).at(INT_ORDER_MAX).at(iG);
                    const tPoint p = PPhysicsCoordD01Nj(coords, shape);
                    polynomial.Derivatives(full, p, center, cell(iCell, 7));
                    const real weight = CellJacobianDet(dim, coords, shape) * q.GetWeight(iG) * (geometry->axisymmetric ? p[1] : 1.);
                    for (int j = 0; j <= n; ++j) moment(iCell, j) += full(0, j) * weight;
                }
                const real volume = moment(iCell, 0);
                for (int j = 0; j <= n; ++j) moment(iCell, j) /= volume;
            }
            auto systemWrite = systems.Write();
            const auto system = systemWrite.View<R::TableRows>();
            auto basis = [&](index iCell, const tPoint &p, SmallMatrix &out)
            {
                polynomial.Derivatives(full, p, {cell(iCell, 1), cell(iCell, 2), cell(iCell, 3)}, cell(iCell, 7));
                out = full.rightCols(n);
                for (int j = 0; j < n; ++j) out(0, j) -= moment(iCell, j + 1);
            };
            for (index iFace = 0; iFace < nFace; ++iFace)
            {
                if (!active[size_t(iFace)]) continue;
                std::fill(system[iFace].begin(), system[iFace].end(), 0.);
                const auto parents = mesh.FaceCells(iFace);
                const index iL = parents[0], iR = parents[1];
                DNDS_check_throw_info(iL >= 0 && iL < mesh.NumCellProc() &&
                                          (iR == UnInitIndex || (iR >= 0 && iR < mesh.NumCellProc())), "VR stencil lacks face parent closure");
                mesh.GetCoordsOnFace(iFace, coords);
                const tPoint midpoint = PPhysicsCoordD01Nj(coords, Geom::Elem::detail::NBufferAtQuadrature.buf.at(mesh.GetFaceElement(iFace).type).at(1).at(0));
                const auto jac = ShapeJacobianCoordD01Nj(coords, Geom::Elem::detail::NBufferAtQuadrature.buf.at(mesh.GetFaceElement(iFace).type).at(1).at(0));
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
                        const auto &shape = Geom::Elem::detail::NBufferAtQuadrature.buf.at(element.type).at(geometry->intOrder).at(iG);
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
                DNDS_check_throw_info(!rotated || functional.dirWeightCombPowV == 1,
                                      "rotated periodic VR requires the Frobenius derivative metric");
                system(iFace, 5 * n * n + 2 * n) = diagonal[0];
                aL.setZero(n, n), aR.setZero(n, n), bLR.setZero(n, n);
                Eigen::Map<Eigen::VectorXd> bL(system[iFace].data() + 3 * n * n, n), bR(system[iFace].data() + 3 * n * n + n, n);
                for (index iQ = faceOffset(iFace, 0); iQ < faceOffset(iFace + 1, 0); ++iQ)
                {
                    const tPoint p{point(iQ, 0), point(iQ, 1), point(iQ, 2)};
                    basis(iL, p, left);
                    if (settings.cacheDiffBase)
                        Eigen::Map<Matrix>(cache[2 * iQ].data(), n + 1, n) = left;
                    weighted = left;
                    for (int j = 0; j <= n; ++j) weighted.row(j) *= diagonal[j] * point(iQ, 6);
                    aL.noalias() += left.transpose() * weighted;
                    bL.noalias() += left.row(0).transpose() * (diagonal[0] * point(iQ, 6));
                    if (iR != UnInitIndex)
                    {
                        basis(iR, Detail::FacePointInCell(mesh, iFace, 1, p), right);
                        if (settings.cacheDiffBase)
                            Eigen::Map<Matrix>(cache[2 * iQ + 1].data(), n + 1, n) = right;
                        if (rotated) polynomial.Pullback(right, rightToLeft.transpose());
                        bLR.noalias() += weighted.transpose() * right;
                        weighted = right;
                        for (int j = 0; j <= n; ++j) weighted.row(j) *= diagonal[j] * point(iQ, 6);
                        aR.noalias() += right.transpose() * weighted;
                        bR.noalias() += right.row(0).transpose() * (diagonal[0] * point(iQ, 6));
                    }
                }
                Eigen::Map<Matrix>(system[iFace].data(), n, n) = aL;
                Eigen::Map<Matrix>(system[iFace].data() + n * n, n, n) = aR;
                Eigen::Map<Matrix>(system[iFace].data() + 2 * n * n, n, n) = bLR;
                if (iR != UnInitIndex)
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
                    Eigen::Map<Matrix>(system[iFace].data() + 3 * n * n + 2 * n, n, n) = secondaryL;
                    Eigen::Map<Matrix>(system[iFace].data() + 4 * n * n + 2 * n, n, n) = secondaryR;
                }
            }
            auto offsetsWrite = data->offsets.Write(), neighborsWrite = data->neighbors.Write(), facesWrite = data->faces.Write();
            auto bOffsetsWrite = data->boundaryOffsets.Write(), bPointsWrite = data->boundaryPoints.Write();
            auto matrixWrite = data->matrixAInvB.Write(), vectorWrite = data->vectorAInvB.Write(), secondaryWrite = data->secondary.Write();
            auto injectionWrite = data->boundaryInjection.Write();
            const auto offsets = offsetsWrite.View<R::FixedRows<1>>(), neighbors = neighborsWrite.View<R::FixedRows<1>>(), faces = facesWrite.View<R::FixedRows<1>>();
            const auto bOffsets = bOffsetsWrite.View<R::FixedRows<1>>();
            const auto bPoints = bPointsWrite.View<R::FixedRows<4>>();
            const auto matrices = matrixWrite.View<R::TableRows>(), vectors = vectorWrite.View<R::TableRows>(), secondaries = secondaryWrite.View<R::TableRows>(), injections = injectionWrite.View<R::TableRows>();
            index iEntry = 0, iBoundary = 0;
            data->inverseConditions = R::Array<real>(R::RowLayout::Dense(nCell, 4));
            auto conditionsWrite = data->inverseConditions.Write();
            const auto conditions = conditionsWrite.View<R::FixedRows<4>>();
            data->inverseDiagnostics.cells = nCell;
            SmallMatrix A, inverse, product;
            for (index iCell = 0; iCell < nCell; ++iCell)
            {
                offsets(iCell, 0) = iEntry;
                A.setZero(n, n);
                const auto c2f = mesh.CellFaces(iCell);
                for (index slot = 0; slot < c2f.size(); ++slot)
                {
                    const index iFace = c2f[slot];
                    const int side = mesh.CellIsFaceBack(iCell, iFace, rowsize(slot)) ? 0 : 1;
                    A += Eigen::Map<const Matrix>(system[iFace].data() + side * n * n, n, n);
                    if (mesh.CellFaceOther(iCell, iFace, rowsize(slot)) == iCell &&
                        Detail::OtherFrameInCell(mesh, iFace, side).isApprox(tGPoint::Identity(), 1e-14))
                    {
                        const Eigen::Map<const Matrix> B(system[iFace].data() + 2 * n * n, n, n);
                        if (side == 0) A -= B; else A -= B.transpose();
                    }
                }
                const auto spectrum = HardEigen::EigenSVDInverse(A, inverse, settings.svdTolerance, settings.svdAbsoluteTolerance);
                Eigen::Map<Matrix>(localMatrices[iCell].data(), n, n) = A;
                Eigen::Map<Matrix>(localInverses[iCell].data(), n, n) = inverse;
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
                for (index slot = 0; slot < c2f.size(); ++slot, ++iEntry)
                {
                    const index iFace = c2f[slot], other = mesh.CellFaceOther(iCell, iFace, rowsize(slot));
                    const int side = mesh.CellIsFaceBack(iCell, iFace, rowsize(slot)) ? 0 : 1;
                    faces(iEntry, 0) = iFace, neighbors(iEntry, 0) = other;
                    const tGPoint rotation = Detail::OtherFrameInCell(mesh, iFace, side);
                    Eigen::Map<tGPoint>(rotations[iEntry].data()) = rotation;
                    data->hasRotatedPeriodicity |= !rotation.isApprox(tGPoint::Identity(), 1e-14);
                    bOffsets(iEntry, 0) = iBoundary;
                    auto M = Eigen::Map<Matrix>(matrices[iEntry].data(), n, n);
                    auto b = Eigen::Map<Eigen::VectorXd>(vectors[iEntry].data(), n);
                    auto secondary = Eigen::Map<Matrix>(secondaries[iEntry].data(), n, n);
                    auto rawM = Eigen::Map<Matrix>(rawMatrices[iEntry].data(), n, n);
                    auto rawB = Eigen::Map<Eigen::VectorXd>(rawVectors[iEntry].data(), n);
                    rawM.setZero(), rawB.setZero();
                    M.setZero(), b.setZero(), secondary.setZero();
                    if (other != UnInitIndex)
                    {
                        const Eigen::Map<const Matrix> B(system[iFace].data() + 2 * n * n, n, n);
                        if (other != iCell || !Detail::OtherFrameInCell(mesh, iFace, side).isApprox(tGPoint::Identity(), 1e-14))
                        {
                            if (side == 0) rawM = B; else rawM = B.transpose();
                            M.noalias() = inverse * rawM;
                        }
                        rawB = Eigen::Map<const Eigen::VectorXd>(system[iFace].data() + 3 * n * n + side * n, n);
                        b.noalias() = inverse * rawB;
                        secondary = Eigen::Map<const Matrix>(system[iFace].data() + (3 + side) * n * n + 2 * n, n, n);
                    }
                    else
                        for (index iQ = faceOffset(iFace, 0); iQ < faceOffset(iFace + 1, 0); ++iQ, ++iBoundary)
                        {
                            basis(iCell, {point(iQ, 0), point(iQ, 1), point(iQ, 2)}, left);
                            Eigen::Map<Eigen::VectorXd>(rawInjections[iBoundary].data(), n) = left.row(0).transpose() * (system(iFace, 5 * n * n + 2 * n) * point(iQ, 6));
                            Eigen::Map<Eigen::VectorXd>(injections[iBoundary].data(), n).noalias() = inverse * Eigen::Map<const Eigen::VectorXd>(rawInjections[iBoundary].data(), n);
                            bPoints(iBoundary, 0) = iCell, bPoints(iBoundary, 1) = iFace;
                            bPoints(iBoundary, 2) = iQ - faceOffset(iFace, 0), bPoints(iBoundary, 3) = iQ;
                        }
                }
            }
            offsets(nCell, 0) = iEntry, bOffsets(iEntry, 0) = iBoundary; });
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
        return VariationalReconstructionPlan(std::move(data));
    }
}
