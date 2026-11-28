#include "Cauchy/Discretization/BoundaryFluxReduction.hpp"
#include "DNDS/Core/OMP.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "DNDS/Runtime/Storage/Metadata.hpp"
#ifdef DNDS_USE_CUDA
#    include "Cauchy/Discretization/CUDA/BoundaryFluxKernels.hpp"
#endif

#include <algorithm>
#include <unordered_map>

namespace DNDS::Cauchy
{
    namespace R = Runtime;

    struct BoundaryFluxReduction::Data
    {
        R::ExecutionContext context;
        std::vector<int> selectedZones;
        R::Array<BoundaryOutputRow> selectedRows;
        R::MetadataBuffer<int> rowSlots;
        std::vector<real> hostResult;
        R::Array<real> deviceTotals;
        explicit Data(R::ExecutionContext where) : context(where) {}
    };

    BoundaryFluxReduction::BoundaryFluxReduction(R::ExecutionContext context)
        : _data(std::make_shared<Data>(context)) {}

    std::vector<real> BoundaryFluxReduction::EvaluateLocal(
        R::EagerEngine &engine,
        const R::ExecutionStream &stream,
        const std::vector<BoundaryOutputRow> &allBoundaryRows,
        const R::Array<real> &faceFlux,
        const R::Array<real> &boundaryMomentum,
        const R::Array<real> &faceQuadrature,
        const std::vector<int> &zones)
    {
        const auto retained = _data;
        const index components = faceFlux.Layout().Width();
        const index stride = components + 4;
        if (zones.empty())
            return {};

        DNDS_check_throw_info(engine.Owns(stream) &&
                                  stream.Context().device == retained->context.device &&
                                  (retained->context.device == R::Device::Host() ||
                                   stream.Context().stream == retained->context.stream) &&
                                  faceFlux.GetDevice() == retained->context.device &&
                                  boundaryMomentum.GetDevice() == retained->context.device &&
                                  faceQuadrature.GetDevice() == retained->context.device &&
                                  boundaryMomentum.Layout().Width() == 3,
                              "boundary flux reduction context or shape differs");

        if (retained->selectedZones != zones ||
            retained->selectedRows.GetDevice() != retained->context.device)
        {
            engine.Synchronize();
            std::unordered_map<int, int> zoneToSlot;
            zoneToSlot.reserve(zones.size());
            for (size_t i = 0; i < zones.size(); ++i)
                zoneToSlot.emplace(zones[i], int(i));

            std::vector<BoundaryOutputRow> hostRows;
            std::vector<int> hostSlots;
            for (const auto &row : allBoundaryRows)
            {
                const auto it = zoneToSlot.find(row.zone);
                if (it != zoneToSlot.end())
                {
                    hostRows.push_back(row);
                    hostSlots.push_back(it->second);
                }
            }
            R::Array<BoundaryOutputRow> staged(R::RowLayout::Dense(index(hostRows.size()), 1), R::Device::Host());
            {
                auto write = staged.Write();
                const auto view = write.View<R::FixedRows<1>>();
                for (index i = 0; i < index(hostRows.size()); ++i)
                    view(i, 0) = hostRows[size_t(i)];
            }
            retained->selectedRows = retained->context.device == R::Device::Host()
                                         ? std::move(staged)
                                         : staged.To(retained->context.device, retained->context);
            retained->selectedRows.SetName("boundaryFlux/selectedRows", DNDS_ALLOCATION_SITE());
            retained->rowSlots = R::MetadataBuffer<int>(hostSlots).CopyTo(retained->context.device, retained->context);
            retained->rowSlots.SetDiagnosticName("boundaryFlux/rowSlots", DNDS_ALLOCATION_SITE());
            retained->selectedZones = zones;
        }

        const index nZones = index(zones.size());
        retained->hostResult.assign(size_t(nZones * stride), real(0));

        if (retained->context.device == R::Device::Host())
        {
            R::SubmitArrays(
                engine, stream,
                [retained, nZones, components, stride](auto, auto rows, auto flux, auto momentum, auto quadrature)
                {
                    const auto slots = retained->rowSlots.HostView();
                    const index count = rows.RowCount();
                    auto accumulateFace = [&](index iRow, real *out)
                    {
                        const auto row = rows(iRow, 0);
                        const index base = index(slots[iRow]) * stride;
                        out[base] += row.area;
                        if (row.area == 0)
                            return;
                        for (index j = 0; j < components; ++j)
                            out[base + 1 + j] += flux(row.auxiliary, j);
                        Vector3 tangent{};
                        for (index iPoint = 0; iPoint < row.count; ++iPoint)
                        {
                            Vector3 n = row.normal;
                            if (!row.centerNormal)
                                for (int d = 0; d < 3; ++d)
                                    n[d] = quadrature(row.quadratureBegin + iPoint, 3 + d);
                            const auto weighted = momentum[row.boundaryBegin + iPoint];
                            real normalMomentum = 0;
                            for (int d = 0; d < 3; ++d)
                                normalMomentum += weighted[d] * n[d];
                            for (int d = 0; d < 3; ++d)
                                tangent[d] -= weighted[d] - normalMomentum * n[d];
                        }
                        for (int d = 0; d < 3; ++d)
                            out[base + 1 + components + d] += tangent[d];
                    };
                    const index threads = count >= 256 ? OMP::DistributedThreads() : 1;
                    if (threads <= 1)
                    {
                        for (index iRow = 0; iRow < count; ++iRow)
                            accumulateFace(iRow, retained->hostResult.data());
                    }
                    else
                    {
                        std::vector<std::vector<real>> partials(size_t(threads), std::vector<real>(size_t(nZones * stride), real(0)));
                        const index chunk = (count + threads - 1) / threads;
                        OMP::ForRows(0, threads, [&](index worker)
                                     {
                            const index begin = worker * chunk;
                            const index end = std::min(count, begin + chunk);
                            auto &local = partials[size_t(worker)];
                            for (index iRow = begin; iRow < end; ++iRow)
                                accumulateFace(iRow, local.data()); });
                        for (const auto &local : partials)
                            for (size_t k = 0; k < retained->hostResult.size(); ++k)
                                retained->hostResult[k] += local[k];
                    }
                },
                R::ReadAs<R::FixedRows<1>>(retained->selectedRows),
                R::ReadAs<R::TableRows>(faceFlux),
                R::ReadAs<R::FixedRows<3>>(boundaryMomentum),
                R::ReadAs<R::TableRows>(faceQuadrature));
            engine.Synchronize();
            return retained->hostResult;
        }

#ifdef DNDS_USE_CUDA
        if (retained->deviceTotals.RowCount() != nZones || retained->deviceTotals.Layout().Width() != stride)
        {
            engine.Synchronize();
            retained->deviceTotals = R::Array<real>(R::RowLayout::Dense(nZones, stride), retained->context.device);
            retained->deviceTotals.SetName("boundaryFlux/deviceTotals", DNDS_ALLOCATION_SITE());
        }
        R::SubmitArrays(
            engine, stream,
            [retained](auto context, auto rows, auto flux, auto momentum, auto quadrature, auto totals)
            {
                CUDA::AccumulateBoundaryFluxes(context, rows, retained->rowSlots.View(context.device),
                                               flux, momentum, quadrature, totals);
            },
            R::ReadAs<R::FixedRows<1>>(retained->selectedRows),
            R::ReadAs<R::TableRows>(faceFlux),
            R::ReadAs<R::FixedRows<3>>(boundaryMomentum),
            R::ReadAs<R::TableRows>(faceQuadrature),
            R::WriteAs<R::TableRows>(retained->deviceTotals));
        engine.Synchronize();
        const auto host = retained->deviceTotals.To(R::Device::Host(), stream.Context());
        const auto read = host.Read();
        const auto view = read.View<R::TableRows>();
        for (index s = 0; s < nZones; ++s)
            for (index k = 0; k < stride; ++k)
                retained->hostResult[size_t(s * stride + k)] = view(s, k);
        return retained->hostResult;
#else
        DNDS_check_throw_info(false, "CUDA boundary flux integration is not compiled");
        return {};
#endif
    }
}
