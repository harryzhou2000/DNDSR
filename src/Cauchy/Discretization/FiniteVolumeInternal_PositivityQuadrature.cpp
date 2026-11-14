#include "Cauchy/Discretization/FiniteVolumeInternal.hpp"
#include "Geom/Quadratures/Quadrature.hpp"

namespace DNDS::Cauchy
{
    void FiniteVolume::Data::BuildPositivityQuadrature(bool limitedVR)
    {
        namespace R = Runtime;
        if (reconstruction)
            quadrature.faceBasis = reconstruction->SamplingFaceBasis();
        if (!settings.numerics.positivity && !limitedVR)
            return;
        const auto meshRead = mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        const auto topology = meshRead.HostView();
        const auto metricRead = metrics.AcquireRead();
        const auto metric = metricRead.HostView();
        const auto offsetsRead = stencil.offsets.Read();
        const auto samplesRead = stencil.samples.Read();
        const auto offsets = offsetsRead.View<R::FixedRows<1>>();
        const auto samples = samplesRead.View<R::FixedRows<1>>();
        const int sourceOrder = settings.numerics.RHSIntegrationOrder();
        // PP protects the same physical volume points used by source RHS.
        // This rule also applies when sources are inactive. There is no separate
        // PP quadrature choice or extra maximum-order interior grid.
        const std::array<int, 1> volumeOrders{sourceOrder};
        const bool shareSource = quadrature.source && quadrature.source->sampling;
        const index sourceCount = shareSource ? quadrature.source->sampling->QueryNumber() : 0;
        const index faceCount = samples.RowCount();
        std::vector<index> ppOffsets(metrics.NumCell() + 1, 0);
        for (index iCell = 0; iCell < metrics.NumCell(); ++iCell)
        {
            index count = offsets(iCell + 1, 0) - offsets(iCell, 0);
            for (int order : volumeOrders)
                count += Geom::Elem::Quadrature(topology.GetCellElement(iCell), order).GetNumPoints();
            ppOffsets[iCell + 1] = ppOffsets[iCell] + count;
            maximumPositivityPoints = std::max(maximumPositivityPoints, count);
        }
        const index count = ppOffsets.back(), extraCount = count - faceCount - sourceCount;
        DNDS_check_throw_info(extraCount >= 0, "PP source query coverage differs from volume rule");
        // Allocate exact query extents once. Maps use local query rows only;
        // there is no global-entity bitmap or coordinate-based coincidence test.
        const auto indices = [](index rows)
        { return R::Array<index>(R::RowLayout::Dense(rows, 1)); };
        R::Array<index> faceCells = indices(faceCount), faceBasisRows = indices(faceCount), faceOutputs = indices(faceCount);
        R::Array<index> extraCells = indices(extraCount), extraOutputs = indices(extraCount), sourceOutputs = indices(sourceCount);
        R::Array<real> facePoints(R::RowLayout::Dense(faceCount, 3)), extraPoints(R::RowLayout::Dense(extraCount, 3));
        stencil.positivityOffsets = indices(metrics.NumCell() + 1);
        stencil.positivityDisplacements = R::Array<real>(R::RowLayout::Dense(count, 3));
        {
            auto offsetWrite = stencil.positivityOffsets.Write();
            auto displacementWrite = stencil.positivityDisplacements.Write();
            auto faceCellWrite = faceCells.Write(), faceBasisWrite = faceBasisRows.Write(), faceOutputWrite = faceOutputs.Write();
            auto extraCellWrite = extraCells.Write(), extraOutputWrite = extraOutputs.Write(), sourceOutputWrite = sourceOutputs.Write();
            auto facePointWrite = facePoints.Write(), extraPointWrite = extraPoints.Write();
            const auto o = offsetWrite.View<R::FixedRows<1>>();
            const auto displacements = displacementWrite.View<R::FixedRows<3>>();
            const auto fc = faceCellWrite.View<R::FixedRows<1>>(), fb = faceBasisWrite.View<R::FixedRows<1>>(), fo = faceOutputWrite.View<R::FixedRows<1>>();
            const auto ec = extraCellWrite.View<R::FixedRows<1>>(), eo = extraOutputWrite.View<R::FixedRows<1>>(), so = sourceOutputWrite.View<R::FixedRows<1>>();
            const auto fp = facePointWrite.View<R::FixedRows<3>>(), ep = extraPointWrite.View<R::FixedRows<3>>();
            for (index iRow = 0; iRow < index(ppOffsets.size()); ++iRow)
                o(iRow, 0) = ppOffsets[iRow];
            index iExtra = 0, iSource = 0;
            for (index iCell = 0; iCell < metrics.NumCell(); ++iCell)
            {
                index iOutput = ppOffsets[iCell];
                const auto center = metric.GetCellCent(iCell);
                const auto appendDisplacement = [&](const Geom::tPoint &point)
                {
                    for (int d = 0; d < 3; ++d)
                        displacements(iOutput, d) = point[d] - center[d];
                };
                for (index entry = offsets(iCell, 0); entry < offsets(iCell + 1, 0); ++entry)
                {
                    const auto &sample = samples(entry, 0);
                    fc(entry, 0) = iCell;
                    fb(entry, 0) = 2 * sample.point + sample.side;
                    fo(entry, 0) = iOutput;
                    for (int d = 0; d < 3; ++d)
                        fp(entry, d) = sample.ownPoint[d];
                    appendDisplacement({sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]});
                    ++iOutput;
                }
                Geom::tSmallCoords coords;
                topology.GetCoordsOnCell(iCell, coords);
                const auto element = topology.GetCellElement(iCell);
                for (int order : volumeOrders)
                {
                    const Geom::Elem::Quadrature rule(element, order);
                    for (int iQ = 0; iQ < rule.GetNumPoints(); ++iQ)
                    {
                        const auto point = Geom::Elem::PPhysicsCoordD01Nj(coords,
                                                                          Geom::Elem::GetCachedShapeAtQuadrature(element.type, order, iQ));
                        appendDisplacement(point);
                        if (shareSource && order == sourceOrder)
                            so(iSource++, 0) = iOutput;
                        else
                        {
                            ec(iExtra, 0) = iCell;
                            eo(iExtra, 0) = iOutput;
                            for (int d = 0; d < 3; ++d)
                                ep(iExtra, d) = point[d];
                            ++iExtra;
                        }
                        ++iOutput;
                    }
                }
                DNDS_check_throw_info(iOutput == ppOffsets[iCell + 1], "PP query fill differs from counted rows");
            }
            DNDS_check_throw_info(iExtra == extraCount && iSource == sourceCount, "PP volume query partition is incomplete");
        }
        if (!reconstruction)
            return;
        Detail::QuadratureSampling queries;
        queries.rows = count;
        const int cachedDerivatives = settings.numerics.cfv.cacheDiffBase
                                          ? CFV::Math::TaylorBasis(metrics.Dimension(), settings.numerics.cfv.maxOrder).DerivativeNumber(settings.numerics.cfv.cacheDiffBaseOrder)
                                          : 0;
        if (!cachedDerivatives)
            faceBasisRows = indices(0);
        queries.segments.push_back(CFV::PolynomialSamplingPlan::FromSharedBasis(
                                       metrics, *reconstruction, faceCells, facePoints, quadrature.faceBasis, faceBasisRows, cachedDerivatives, false)
                                       .Named("Cauchy/quadrature/pp-face")
                                       .WithOutputRows(faceOutputs, count));
        if (extraCount)
            queries.segments.push_back(CFV::PolynomialSamplingPlan(*reconstruction, extraCells, extraPoints, false)
                                           .Named("Cauchy/quadrature/volume")
                                           .WithOutputRows(extraOutputs, count));
        if (shareSource)
            queries.segments.push_back(quadrature.source->sampling->WithDerivativeOrder(0).WithOutputRows(sourceOutputs, count));
        quadrature.positivity = std::move(queries);
    }
}
