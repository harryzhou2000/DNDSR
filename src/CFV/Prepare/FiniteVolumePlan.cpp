#include "DNDS/Algebra/DofHalo.hpp"
#include "CFV/Detail/ReconstructionData.hpp"
#include "Geom/Quadratures/Quadrature.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include "DNDS/Math/HardEigen.hpp"
#include "DNDS/Core/OMP.hpp"

namespace DNDS::CFV
{
    namespace R = Runtime;
    using namespace Geom;
    using namespace Geom::Elem;
    const Algebra::DofSpace &FiniteVolumePlan::Space() const { return _data->dofs; }
    std::shared_ptr<R::CommunicationContext> FiniteVolumePlan::Communication() const { return _data->communication; }

    namespace
    {
        void ConstructCellSmoothScale(const FiniteVolumePlan &plan, R::Array<real> &factors, int iterations)
        {
            const auto communication = plan.Communication();
            const index ownedCount = plan.NumCell(), localCount = plan.NumCellProc();
            std::optional<FiniteVolumeTopologyRead> topology;
            std::optional<FVField> scale;
            std::vector<real> next;
            communication->LocalStage([&]
                                      {
                scale.emplace(plan.Space(), factors.Layout(), R::Device::Host());
                next.resize(ownedCount);
                topology.emplace(plan.AcquireTopologyRead());
                const auto mesh = topology->View();
                const auto geometry = plan.AcquireRead();
                const auto metric = geometry.HostView();
                auto scaleWrite = scale->Owned().Write();
                auto factorWrite = factors.Write();
                const auto current = scaleWrite.View<R::FixedRows<1>>();
                const auto initial = factorWrite.View<R::FixedRows<1>>();
                OMP::ForSetupRows(0, ownedCount, [&](index iCell)
                                  {
                    real area = 0;
                    for (const auto iFace : mesh.CellFaces(iCell))
                        area += metric.GetFaceArea(iFace);
                    const real h = metric.GetCellVol(iCell) / (area + verySmallReal);
                    initial(iCell, 0) = current(iCell, 0) = h; }); });

            const auto halo = plan.BindHalo(*scale);
            const auto mesh = topology->View();
            for (int pass = 0; pass < iterations; ++pass)
            {
                halo.Begin().Wait();
                communication->LocalStage([&]
                                          {
                    {
                        const auto ownedRead = scale->Owned().Read();
                        const auto ghostRead = scale->Ghost().Read();
                        const auto owned = ownedRead.View<R::FixedRows<1>>();
                        const auto ghost = ghostRead.View<R::FixedRows<1>>();
                        OMP::ForSetupRows(0, ownedCount, [&](index iCell)
                                          {
                            const auto faces = mesh.CellFaces(iCell);
                            real sum = 0, count = 0;
                            for (rowsize slot = 0; slot < faces.size(); ++slot)
                            {
                                const index iOther = mesh.CellFaceOther(iCell, faces[slot], slot);
                                if (iOther == UnInitIndex)
                                    continue;
                                DNDS_check_throw_info(iOther >= 0 && iOther < localCount,
                                                      "FV smoothing needs complete face-neighbor closure");
                                sum += iOther < ownedCount ? owned(iOther, 0) : ghost(iOther - ownedCount, 0);
                                ++count;
                            }
                            const real h = owned(iCell, 0);
                            // Euler's min-filter with weight 1 for the cell and all its neighbors combined.
                            next[iCell] = count ? std::min(h, (sum + count * h) / (2 * count)) : h; });
                    }
                    auto scaleWrite = scale->Owned().Write();
                    const auto current = scaleWrite.View<R::FixedRows<1>>();
                    OMP::ForSetupRows(0, ownedCount, [&](index iCell)
                                      { current(iCell, 0) = next[iCell]; }); });
            }
            communication->LocalStage([&]
                                      {
                const auto scaleRead = scale->Owned().Read();
                auto factorWrite = factors.Write();
                const auto smoothed = scaleRead.View<R::FixedRows<1>>();
                const auto ratio = factorWrite.View<R::FixedRows<1>>();
                OMP::ForSetupRows(0, ownedCount, [&](index iCell)
                                  { ratio(iCell, 0) = smoothed(iCell, 0) / ratio(iCell, 0); }); });
        }
    }

    void ReconstructionSettings::Validate() const
    {
        functional.Validate();
        storage.Validate();
        cudaSweep.Validate();
        // Quadrature and polynomial degrees are independent numerical choices.
        // No degree+1 rule guarantees exact integration of the full functional.
        // Keep the supported-rule range; SVD handles the chosen discrete system.
        DNDS_check_throw_info(maxOrder >= 1 && maxOrder <= 3 && intOrder >= 1 && intOrder <= INT_ORDER_MAX &&
                                  cacheDiffBaseOrder >= 0 && cacheDiffBaseOrder <= 3 &&
                                  std::isfinite(jacobiRelax) && jacobiRelax > 0 && jacobiRelax < 2 &&
                                  std::isfinite(bcWeight) && bcWeight >= 0 &&
                                  std::isfinite(svdTolerance) && svdTolerance >= 0 && svdTolerance <= 1 &&
                                  std::isfinite(svdAbsoluteTolerance) && svdAbsoluteTolerance >= 0 &&
                                  std::isfinite(WBAP_nStd) && WBAP_nStd > 0,
                              "invalid finite-volume reconstruction settings");
        for (const auto weight : boundaryDerivativeWeights)
            DNDS_check_throw_info(std::isfinite(weight) && weight >= 0, "invalid boundary derivative weight");
        for (const auto &[zone, weights] : boundaryZoneWeights)
            for (const auto weight : weights)
                DNDS_check_throw_info(std::isfinite(weight) && weight >= 0, "invalid zone boundary derivative weight");
    }
    FiniteVolumePlan FiniteVolumePlan::ConstructMetrics(ssp<UnstructuredMesh> mesh, FiniteVolumeMetricOptions options)
    {
        auto setupStage = options.progress.Stage("FV/metrics");
        DNDS_check_throw_info(mesh, "finite-volume metrics require a mesh");
        const auto mpi = mesh->getMPI();
        const int intOrder = options.integrationOrder;
        const int faceIntOrder = options.faceIntegrationOrder ? options.faceIntegrationOrder : intOrder;
        const int geomOrder = options.geometryOrder ? options.geometryOrder : INT_ORDER_MAX;
        const bool axisymmetric = options.axisymmetric;
        const auto meshCommunication = mesh->PreparationCommunication();
        auto communication = meshCommunication
                                 ? meshCommunication->WithPreflight(options.collectivePreflight)
                                 : R::CommunicationContext::Duplicate(mpi.comm, options.collectivePreflight);
        std::shared_ptr<Detail::FiniteVolumeData> data;
        // All local allocation/geometry failures agree before the halo collective.
        communication->LocalStage([&]
                                  {
            DNDS_check_throw_info(intOrder >= 1 && intOrder <= INT_ORDER_MAX, "invalid FV integration order");
            DNDS_check_throw_info(geomOrder >= 1 && geomOrder <= INT_ORDER_MAX, "invalid FV geometry integration order");
            DNDS_check_throw_info(faceIntOrder == 1 || faceIntOrder == intOrder, "FV face integration must be midpoint or the metric order");
            DNDS_check_throw_info(options.nIterCellSmoothScale >= 0, "negative FV cell smooth scale iteration count");
            auto read = mesh->AcquireRead(MeshReadRequirements::CellFlux());
            const auto view = read.HostView();
            DNDS_check_throw_info(view.Dimension() == 2 || view.Dimension() == 3, "FV requires 2D/3D geometry");
            DNDS_check_throw_info(!axisymmetric || view.Dimension() == 2, "axisymmetric metrics require planar 2D geometry, radius=y");
            if (axisymmetric && view.IsPeriodic())
                for (index iFace = 0; iFace < view.NumFaceProc(); ++iFace)
                {
                    const int zone = view.GetFaceZone(iFace);
                    if (!FaceIDIsPeriodic(zone)) continue;
                    const int frame = FaceIDIsPeriodicMain(zone) ? -zone : -zone - 3;
                    DNDS_check_throw_info(view.Periodic().rotation[frame].map().row(1).isApprox(tPoint::UnitY().transpose(), 1e-12) &&
                                              std::abs(view.Periodic().translation[frame].map()[1]) < 1e-12,
                                          "axisymmetric periodic transforms must preserve radius=y");
                }
            data = std::make_shared<Detail::FiniteVolumeData>(mesh, intOrder, geomOrder, axisymmetric, communication);
            const index nCellProc = view.NumCellProc();
            data->cellVolume = R::Array<real>(R::RowLayout::Dense(nCellProc, 1));
            data->cellCent = R::Array<real>(R::RowLayout::Dense(nCellProc, 3));
            data->cellBary = R::Array<real>(R::RowLayout::Dense(nCellProc, 3));
            data->cellScale = R::Array<real>(R::RowLayout::Dense(nCellProc, 1));
            data->cellMaxLenScale = R::Array<real>(R::RowLayout::Dense(nCellProc, 1));
            data->cellVolume.SetName("CFV/metrics/cellVolume", DNDS_ALLOCATION_SITE());
            data->cellCent.SetName("CFV/metrics/cellCent", DNDS_ALLOCATION_SITE());
            data->cellBary.SetName("CFV/metrics/cellBary", DNDS_ALLOCATION_SITE());
            data->cellScale.SetName("CFV/metrics/cellScale", DNDS_ALLOCATION_SITE());
            data->cellMaxLenScale.SetName("CFV/metrics/cellMaxLenScale", DNDS_ALLOCATION_SITE());
            const index scaleRows = options.nIterCellSmoothScale ? view.NumCell() : 0;
            data->cellSmoothScaleFactors = R::Array<real>(R::RowLayout::Dense(scaleRows, 1));
            data->cellSmoothScaleFactors.SetName("CFV/metrics/cellSmoothScaleFactors", DNDS_ALLOCATION_SITE());
            data->radialMoment = R::Array<real>(R::RowLayout::Dense(axisymmetric ? nCellProc : 0, 3));
            data->radialMoment.SetName("CFV/metrics/radialMoment", DNDS_ALLOCATION_SITE());
            auto radialWrite = data->radialMoment.Write();
            const auto radial = radialWrite.View<R::FixedRows<3>>();
            data->faceOffsets = R::Array<index>(R::RowLayout::Dense(view.NumFaceProc() + 1, 1));
            data->faceOffsets.SetName("CFV/metrics/faceOffsets", DNDS_ALLOCATION_SITE());
            auto offsets = data->faceOffsets.Write();
            auto offset = offsets.View<R::FixedRows<1>>();
            std::vector<index> facePointCounts(static_cast<size_t>(view.NumFaceProc()));
            OMP::ForSetupRows(0, view.NumFaceProc(), [&](index iFace)
                              { facePointCounts[size_t(iFace)] = Quadrature(view.GetFaceElement(iFace), faceIntOrder).GetNumPoints(); });
            const index count = OMP::ExclusivePrefixRows(
                view.NumFaceProc(), [&](index iFace) noexcept { return facePointCounts[size_t(iFace)]; },
                [&](index iFace, index value) noexcept { offset(iFace, 0) = value; });
            data->faceQuadrature = R::Array<real>(R::RowLayout::Dense(count, 7));
            data->faceQuadrature.SetName("CFV/metrics/faceQuadrature", DNDS_ALLOCATION_SITE());
            auto volWrite = data->cellVolume.Write();
            auto centWrite = data->cellCent.Write();
            auto baryWrite = data->cellBary.Write();
            auto scaleWrite = data->cellScale.Write();
            auto maxLenWrite = data->cellMaxLenScale.Write();
            const auto cellVol = volWrite.View<R::FixedRows<1>>();
            const auto cellCent = centWrite.View<R::FixedRows<3>>();
            const auto cellBary = baryWrite.View<R::FixedRows<3>>();
            const auto cellScale = scaleWrite.View<R::FixedRows<1>>();
            const auto cellMaxLen = maxLenWrite.View<R::FixedRows<1>>();
            // Immutable process-wide shape tables are fully constructed before
            // this setup. Each worker owns coordinates and integration scratch.
            OMP::ForSetupRows(0, nCellProc, [&](index iCell)
            {
                tSmallCoords coords;
                view.GetCoordsOnCell(iCell, coords);
                const auto element = view.GetCellElement(iCell);
                const tPoint cent = PPhysicsCoordD01Nj(coords, Geom::Elem::GetCachedShapeAtQuadrature(element.type, 1, 0));
                // Single quadrature loop at geomOrder (default INT_ORDER_MAX) computes
                // volume, barycenter, axisymmetric radial moment, and principal-axis inertia
                // in cell-center-shifted coordinates (avoiding origin-offset cancellation).
                const Quadrature geomQuad(element, geomOrder);
                real volume = 0.;
                real plainVolume = 0.;
                tPoint baryRel = tPoint::Zero();
                tPoint plainMomentRel = tPoint::Zero();
                tJacobi rawInertiaRel = tJacobi::Zero();
                for (int iG = 0; iG < geomQuad.GetNumPoints(); ++iG)
                {
                    const auto &shape = Geom::Elem::GetCachedShapeAtQuadrature(element.type, geomOrder, iG);
                    const real det = CellJacobianDet(view.Dimension(), coords, shape);
                    DNDS_check_throw_info(std::isfinite(det) && det > 0, "FV cell has nonpositive Jacobian");
                    const tPoint p = PPhysicsCoordD01Nj(coords, shape);
                    DNDS_check_throw_info(!axisymmetric || p[1] >= 0, "negative axisymmetric radius");
                    const real plainWeight = det * geomQuad.GetWeight(iG);
                    const real weighted = plainWeight * (axisymmetric ? p[1] : 1.);
                    const tPoint q = p - cent;
                    plainVolume += plainWeight;
                    plainMomentRel += q * plainWeight;
                    // Legacy inertia uses geometric Jacobian even in radial integration.
                    rawInertiaRel += (q * q.transpose()) * plainWeight;
                    volume += weighted;
                    baryRel += q * weighted;
                }
                DNDS_check_throw_info(std::isfinite(volume) && volume > 0, "nonpositive FV cell volume");
                const tPoint baryOffset = baryRel / volume;
                const tPoint bary = cent + baryOffset;
                if (axisymmetric)
                    for (int d = 0; d < 3; ++d)
                        radial(iCell, d) = (plainMomentRel[d] - plainVolume * baryOffset[d]) / volume;
                const tSmallCoords coordsFromBary = coords.colwise() - bary;
                const tPoint box = coordsFromBary.array().abs().rowwise().maxCoeff();
                const real scale = box.head(view.Dimension()).maxCoeff();
                // Parallel-axis shift from cent to bary (all terms O(h^2)).
                tJacobi inertia = (rawInertiaRel - plainMomentRel * baryOffset.transpose() -
                                   baryOffset * plainMomentRel.transpose() +
                                   plainVolume * (baryOffset * baryOffset.transpose())) /
                                  volume;
                inertia = 0.5 * (inertia + inertia.transpose());
                tJacobi frame = tJacobi::Identity();
                if (view.Dimension() == 3)
                    frame = HardEigen::Eigen3x3RealSymPrincipalFrame(inertia);
                else
                    frame({0, 1}, {0, 1}) = HardEigen::Eigen2x2RealSymPrincipalFrame(inertia({0, 1}, {0, 1}));
                const tSmallCoords centered = frame.transpose() * coordsFromBary;
                const tPoint halfBox = centered.array().abs().rowwise().maxCoeff();
                const real maxLenScale = 2.0 * halfBox.head(view.Dimension()).maxCoeff();
                DNDS_check_throw_info(std::isfinite(volume) && scale > 0 &&
                                          std::isfinite(maxLenScale) && maxLenScale > 0,
                                      "invalid FV cell geometry");
                cellVol(iCell, 0) = volume;
                for (int d = 0; d < 3; ++d)
                    cellCent(iCell, d) = cent[d], cellBary(iCell, d) = bary[d];
                cellScale(iCell, 0) = scale;
                cellMaxLen(iCell, 0) = maxLenScale;
            });
            auto points = data->faceQuadrature.Write();
            const auto point = points.View<R::FixedRows<7>>();
            data->faceCent = R::Array<real>(R::RowLayout::Dense(view.NumFaceProc(), 3));
            data->faceMeanNorm = R::Array<real>(R::RowLayout::Dense(view.NumFaceProc(), 3));
            data->faceArea = R::Array<real>(R::RowLayout::Dense(view.NumFaceProc(), 1));
            data->faceCent.SetName("CFV/metrics/faceCent", DNDS_ALLOCATION_SITE());
            data->faceMeanNorm.SetName("CFV/metrics/faceMeanNorm", DNDS_ALLOCATION_SITE());
            data->faceArea.SetName("CFV/metrics/faceArea", DNDS_ALLOCATION_SITE());
            data->facePeriodicCodes = R::Array<uint8_t>(R::RowLayout::Dense(view.IsPeriodic() ? view.NumFaceProc() : 0, 1));
            data->periodicParameters = R::Array<FacePeriodicParameters>(R::RowLayout::Dense(1, 1));
            data->facePeriodicCodes.SetName("CFV/metrics/facePeriodicCodes", DNDS_ALLOCATION_SITE());
            data->periodicParameters.SetName("CFV/metrics/periodicParameters", DNDS_ALLOCATION_SITE());
            auto periodicWrite = data->periodicParameters.Write();
            auto codesWrite = data->facePeriodicCodes.Write();
            const auto parameters = periodicWrite.View<R::FixedRows<1>, false>();
            const auto codes = codesWrite.View<R::FixedRows<1>, false>();
            parameters(0, 0).generators = view.Periodic();
            parameters(0, 0).rotatedGenerators = 0;
            for (int generator = 1; generator <= 3; ++generator)
                if (!(view.Periodic().rotation[generator].map().array() == tGPoint::Identity().array()).all())
                    parameters(0, 0).rotatedGenerators |= uint8_t(1) << (generator - 1);
            auto faceCentWrite = data->faceCent.Write();
            auto faceNormWrite = data->faceMeanNorm.Write();
            auto faceAreaWrite = data->faceArea.Write();
            const auto fCent = faceCentWrite.View<R::FixedRows<3>>();
            const auto fNorm = faceNormWrite.View<R::FixedRows<3>>();
            const auto fArea = faceAreaWrite.View<R::FixedRows<1>>();
            OMP::ForSetupRows(0, view.NumFaceProc(), [&](index iFace)
            {
                if (view.IsPeriodic())
                {
                    const int zone = view.GetFaceZone(iFace);
                    // The zone encodes the canonical parent-zero -> parent-one
                    // map even if the local own-side Pbi is the base image.
                    codes(iFace, 0) = FaceIDIsPeriodicMain(zone) ? uint8_t(-zone)
                                        : FaceIDIsPeriodicDonor(zone) ? uint8_t(-zone) : 0;
                    DNDS_check_throw_info(codes(iFace, 0) < uint8_t(FacePeriodicCode::Composed),
                                          "a composed face transform requires an explicit shared registry");
                }
                tSmallCoords coords;
                view.GetCoordsOnFace(iFace, coords);
                const auto element = view.GetFaceElement(iFace);
                real integratedArea = 0;
                tPoint integratedNormal = tPoint::Zero(), geometricNormal = tPoint::Zero();
                {
                    // With midpoint RHS, put the integrated area and mean
                    // normal into its one sample, including curved faces.
                    const int faceMeasureOrder = intOrder == 1 ? geomOrder : intOrder;
                    const Quadrature measure(element, faceMeasureOrder);
                    for (int iG = 0; iG < measure.GetNumPoints(); ++iG)
                    {
                        const auto &shape = Geom::Elem::GetCachedShapeAtQuadrature(element.type, faceMeasureOrder, iG);
                        const auto jacobian = ShapeJacobianCoordD01Nj(coords, shape);
                        const tPoint normal = view.Dimension() == 2 ? FacialJacobianToNormVec<2>(jacobian) : FacialJacobianToNormVec<3>(jacobian);
                        const auto position = PPhysicsCoordD01Nj(coords, shape);
                        const real det = normal.norm();
                        DNDS_check_throw_info(std::isfinite(det) && det > 0, "FV face has zero measure");
                        DNDS_check_throw_info(!axisymmetric || position[1] >= 0, "negative axisymmetric face radius");
                        const real weight = measure.GetWeight(iG) * (axisymmetric ? position[1] : 1.);
                        integratedArea += det * weight;
                        integratedNormal += normal * weight;
                        geometricNormal += normal * measure.GetWeight(iG);
                        // Reuse this geometry pass for high-order RHS records.
                        // The midpoint record is filled after integrating the
                        // scalar area and mean normal; it needs no second pass.
                        if (faceIntOrder != 1)
                        {
                            const index iPoint = offset(iFace, 0) + iG;
                            for (int d = 0; d < 3; ++d)
                                point(iPoint, d) = position[d], point(iPoint, d + 3) = normal[d] / det;
                            point(iPoint, 6) = det * weight;
                        }
                    }
                    DNDS_check_throw_info(std::isfinite(integratedArea) && integratedArea >= 0, "invalid integrated face measure");
                }
                const tPoint center = PPhysicsCoordD01Nj(coords, Geom::Elem::GetCachedShapeAtQuadrature(element.type, 1, 0));
                // A face on the symmetry axis has zero weighted area but still
                // needs a valid geometric direction for boundary evaluation.
                if (axisymmetric && integratedArea == 0)
                    integratedNormal = geometricNormal;
                const real normalLength = integratedNormal.norm();
                DNDS_check_throw_info(std::isfinite(normalLength) && normalLength > 0, "degenerate FV mean face normal");
                integratedNormal /= normalLength;
                for (int d = 0; d < 3; ++d)
                    fCent(iFace, d) = center[d], fNorm(iFace, d) = integratedNormal[d];
                fArea(iFace, 0) = integratedArea;
                if (faceIntOrder == 1)
                {
                    const index iPoint = offset(iFace, 0);
                    for (int d = 0; d < 3; ++d)
                        point(iPoint, d) = center[d], point(iPoint, d + 3) = integratedNormal[d];
                    point(iPoint, 6) = integratedArea;
                }
            }); });
        // Smoothing determines how many halo exchanges will be posted.
        if (options.collectivePreflight)
        {
            std::array<int, 2> passExtrema{options.nIterCellSmoothScale, -options.nIterCellSmoothScale};
            MPI::AllreduceInPlace<MPI::Reduction::Max>(passExtrema, mpi.comm);
            DNDS_check_throw_info(passExtrema[0] == -passExtrema[1], "FV smoothing pass counts differ across ranks");
        }
        data->halo.emplace(data->cells.BuildRowTransferPlan(communication));
        FiniteVolumePlan plan(data);
        if (options.nIterCellSmoothScale)
            ConstructCellSmoothScale(plan, data->cellSmoothScaleFactors, options.nIterCellSmoothScale);
        return plan;
    }
    void FiniteVolumePlan::Validate() const
    {
        DNDS_check_throw_info(_data && _data->stamp.Valid(), "finite-volume plan is stale; reconstruct metrics");
        _data->cells.Validate();
    }
    FiniteVolumeRead::FiniteVolumeRead(std::shared_ptr<const Detail::FiniteVolumeData> data, R::ExecutionContext context)
        : _data(std::move(data)),
          _meshRead(std::make_shared<MeshReadAccess>(_data->mesh->AcquireRead(MeshReadRequirements::CellFlux()))),
          _cellVolume(_data->cellVolume.Read(context)), _cellCent(_data->cellCent.Read(context)),
          _cellBary(_data->cellBary.Read(context)), _cellScale(_data->cellScale.Read(context)),
          _cellMaxLenScale(_data->cellMaxLenScale.Read(context)),
          _quadrature(_data->faceQuadrature.Read(context)), _radialMoment(_data->radialMoment.Read(context)),
          _faceCent(_data->faceCent.Read(context)), _faceMeanNorm(_data->faceMeanNorm.Read(context)),
          _faceArea(_data->faceArea.Read(context)),
          _periodicCodes(_data->facePeriodicCodes.Read(context)),
          _periodicParameters(_data->periodicParameters.Read(context)), _offsets(_data->faceOffsets.Read(context))
    {
        DNDS_check_throw_info(_data->stamp.Valid(), "finite-volume metrics changed before acquisition");
    }
    FiniteVolumeHostView FiniteVolumeRead::HostView() const
    {
        DNDS_check_throw_info(_data->cellVolume.GetDevice() == R::Device::Host(), "HostView requested from device FV metrics");
        return View();
    }
    FiniteVolumeHostView FiniteVolumeRead::View() const
    {
        return {_cellVolume.View<R::FixedRows<1>>(), _cellCent.View<R::FixedRows<3>>(), _cellBary.View<R::FixedRows<3>>(), _cellScale.View<R::FixedRows<1>>(), _cellMaxLenScale.View<R::FixedRows<1>>(), _offsets.View<R::FixedRows<1>>(), _quadrature.View<R::FixedRows<7>>(), _radialMoment.View<R::FixedRows<3>>(), _faceCent.View<R::FixedRows<3>>(), _faceMeanNorm.View<R::FixedRows<3>>(), _faceArea.View<R::FixedRows<1>>(), {_periodicCodes.View<R::FixedRows<1>, false>(), _periodicParameters.View<R::FixedRows<1>, false>()}};
    }
    FiniteVolumeRead FiniteVolumePlan::AcquireRead(R::ExecutionContext context) const
    {
        Validate();
        DNDS_check_throw_info(context.device == Device(), "FV metric read context differs from placement");
        return FiniteVolumeRead(_data, context);
    }
    const R::Array<real> &FiniteVolumePlan::CellVolumes() const { return _data->cellVolume; }
    const R::Array<real> &FiniteVolumePlan::CellCenters() const { return _data->cellCent; }
    const R::Array<real> &FiniteVolumePlan::CellBarycenters() const { return _data->cellBary; }
    const R::Array<real> &FiniteVolumePlan::CellScales() const { return _data->cellScale; }
    const R::Array<real> &FiniteVolumePlan::CellMaxLenScale() const { return _data->cellMaxLenScale; }
    const R::Array<real> &FiniteVolumePlan::CellSmoothScaleFactors() const { return _data->cellSmoothScaleFactors; }
    const R::Array<real> &FiniteVolumePlan::FaceQuadrature() const { return _data->faceQuadrature; }
    const R::Array<real> &FiniteVolumePlan::FaceCenters() const { return _data->faceCent; }
    const R::Array<real> &FiniteVolumePlan::FaceMeanNormals() const { return _data->faceMeanNorm; }
    const R::Array<real> &FiniteVolumePlan::FaceAreas() const { return _data->faceArea; }
    const R::Array<real> &FiniteVolumePlan::RadialMoments() const { return _data->radialMoment; }
    const R::Array<index> &FiniteVolumePlan::FaceOffsets() const { return _data->faceOffsets; }
    const R::Array<uint8_t> &FiniteVolumePlan::FacePeriodicCodes() const { return _data->facePeriodicCodes; }
    const R::Array<FacePeriodicParameters> &FiniteVolumePlan::PeriodicParameters() const { return _data->periodicParameters; }
    FiniteVolumeTopologyRead::FiniteVolumeTopologyRead(std::shared_ptr<const Detail::FiniteVolumeData> data)
        : _data(std::move(data)), _source(std::make_shared<CellFaceIncidenceRead>(_data->topology->AcquireRead()))
    {
        DNDS_check_throw_info(_data->stamp.Valid(), "FV topology changed before acquisition");
    }
    CellFaceIncidenceView FiniteVolumeTopologyRead::View() const
    {
        return _source->View();
    }
    FiniteVolumeTopologyRead FiniteVolumePlan::AcquireTopologyRead() const
    {
        Validate();
        return FiniteVolumeTopologyRead(_data);
    }
    MeshReadAccess FiniteVolumePlan::AcquireMeshRead(bool wallDistance) const
    {
        auto requirements = MeshReadRequirements::CellFlux();
        requirements.wallDistance = wallDistance;
        auto guard = _data->mesh->AcquireRead(requirements);
        Validate(); // The retained generation excludes mutation during validation.
        return guard;
    }
    R::Device FiniteVolumePlan::Device() const { return _data->cellVolume.GetDevice(); }
    FiniteVolumePlan FiniteVolumePlan::Placed(R::Device device, R::ExecutionContext context) const
    {
        Validate();
        if (device == Device())
            return *this;
        auto data = std::make_shared<Detail::FiniteVolumeData>(*_data, device, context);
        DNDS_check_throw_info(data->stamp.Valid(), "FV geometry changed during metric placement");
        return FiniteVolumePlan(std::move(data));
    }
    index FiniteVolumePlan::NumCell() const { return _data->cells.NumOwned(); }
    MPI_Comm FiniteVolumePlan::Communicator() const { return _data->communication->Communicator(); }
    index FiniteVolumePlan::NumCellProc() const { return _data->cells.NumLocal(); }
    index FiniteVolumePlan::NumFaceProc() const { return _data->faceOffsets.RowCount() - 1; }
    int FiniteVolumePlan::Dimension() const { return _data->dim; }
    bool FiniteVolumePlan::Axisymmetric() const { return _data->axisymmetric; }
    const EntitySpace &FiniteVolumePlan::Cells() const { return _data->cells; }
    EntityWorkPartition FiniteVolumePlan::CellWorkPartition() const
    {
        Validate();
        std::call_once(_data->partitionOnce, [&]
                       { if (!_data->workPartition) _data->workPartition.emplace(EntityWorkPartition::CellNodeCell(*_data->mesh)); });
        _data->workPartition->Validate();
        return *_data->workPartition;
    }
    FVField FiniteVolumePlan::BuildField(index width) const
    {
        Validate();
        DNDS_check_throw_info(width > 0, "FV field width must be positive");
        FVField field(_data->dofs, R::RowLayout::Dense(NumCell(), width), Device());
        // Runtime allocation publishes zero-initialized device buffers. Keep
        // the existing explicit CPU initialization/version behavior unchanged.
        if (Device() == R::Device::Host())
            Detail::Zero(field);
        return field;
    }
    FVField FiniteVolumePlan::BuildField(R::RowLayout prototype) const
    {
        return BuildField(std::move(prototype), Device());
    }
    FVField FiniteVolumePlan::BuildField(R::RowLayout prototype, R::Device device) const
    {
        Validate();
        DNDS_check_throw_info(!prototype.IsVariable() && prototype.Width() > 0, "FV fields require a uniform physical table");
        return FVField(_data->dofs, std::move(prototype), device);
    }
    void FiniteVolumePlan::ValidateField(const FVField &field, index width) const
    {
        Validate();
        field.Validate();
        DNDS_check_throw_info(field.Space().Same(_data->dofs) && index(field.Width()) == width && field.Device() == Device(),
                              "FV field belongs to another plan/order or has the wrong width/placement");
    }
    R::TransferBinding FiniteVolumePlan::BindHalo(FVField &field, R::TransferOptions options) const
    {
        _data->communication->LocalStage([&]
                                         { ValidateField(field, field.Width()); });
        const auto transport = options.provider ? R::TransferTransport::DevicePacked : R::TransferTransport::Direct;
        return R::PayloadPlan(*_data->halo, field.Owned().Layout(), field.Ghost().Layout(), R::ElementSchema::Arithmetic<real>())
            .Bind(field.Owned(), field.Ghost(), transport, std::move(options));
    }
    Algebra::DofHalo<> FiniteVolumePlan::BindTrackedHalo(FVField &field, R::TransferOptions options) const
    {
        _data->communication->LocalStage([&]
                                         { ValidateField(field, field.Width()); });
        const auto transport = options.provider ? R::TransferTransport::DevicePacked : R::TransferTransport::Direct;
        return Algebra::DofHalo<>(field, *_data->halo, std::move(options), transport);
    }
    real FiniteVolumePlan::GetCellVol(index iCell) const
    {
        Validate();
        auto read = _data->cellVolume.Read();
        return read.View<R::FixedRows<1>, true>()(iCell, 0);
    }
    tPoint FiniteVolumePlan::GetCellCent(index iCell) const
    {
        Validate();
        auto read = _data->cellCent.Read();
        const auto view = read.View<R::FixedRows<3>, true>();
        return {view(iCell, 0), view(iCell, 1), view(iCell, 2)};
    }
    tPoint FiniteVolumePlan::GetCellBary(index iCell) const
    {
        Validate();
        auto read = _data->cellBary.Read();
        const auto view = read.View<R::FixedRows<3>, true>();
        return {view(iCell, 0), view(iCell, 1), view(iCell, 2)};
    }
    real FiniteVolumePlan::GetCellScale(index iCell) const
    {
        Validate();
        auto read = _data->cellScale.Read();
        return read.View<R::FixedRows<1>, true>()(iCell, 0);
    }
    real FiniteVolumePlan::GetCellMaxLenScale(index iCell) const
    {
        Validate();
        auto read = _data->cellMaxLenScale.Read();
        return read.View<R::FixedRows<1>, true>()(iCell, 0);
    }
    Runtime::Array<real> FiniteVolumePlan::BuildCellMaxLenScale() const
    {
        Validate();
        DNDS_check_throw_info(Device() == R::Device::Host(), "BuildCellMaxLenScale requires host FV metrics");
        Runtime::Array<real> result(Runtime::RowLayout::Dense(NumCellProc(), 1));
        result.SetName("CFV/metrics/cellMaxLengthScale", DNDS_ALLOCATION_SITE());
        const auto read = _data->cellMaxLenScale.Read();
        auto write = result.Write();
        const auto src = read.View<R::FixedRows<1>>();
        const auto dst = write.View<R::FixedRows<1>>();
        OMP::ForRows(0, NumCellProc(), [&](index iCell)
                     { dst(iCell, 0) = src(iCell, 0); });
        return result;
    }
}
