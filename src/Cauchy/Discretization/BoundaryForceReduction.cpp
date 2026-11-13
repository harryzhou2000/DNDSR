#include "Cauchy/Discretization/BoundaryForceReduction.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeBoundaryKernels.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "DNDS/Core/OMP.hpp"
#ifdef DNDS_USE_CUDA
#    include "Cauchy/Discretization/CUDA/BoundaryForceKernels.hpp"
#endif

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    struct BoundaryForceReduction::Data
    {
        R::ExecutionContext context;
        std::vector<int> selected;
        R::MetadataBuffer<int> zones;
        std::vector<Vector3> workerSums;
        Vector3 hostResult{};
        R::Array<real> partial, result;
        R::Array<unsigned int> completed;
        explicit Data(R::ExecutionContext where) : context(where) {}
    };
    BoundaryForceReduction::BoundaryForceReduction(R::ExecutionContext context) : _data(std::make_shared<Data>(context)) {}
    Vector3 BoundaryForceReduction::Evaluate(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                             const FiniteVolumeStencilStorage &stencil, const R::Array<real> &momentum, const std::vector<int> &zones)
    {
        const auto retained = _data;
        DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == retained->context.device &&
                                  stream.Context().stream == retained->context.stream && momentum.GetDevice() == retained->context.device &&
                                  momentum.RowCount() == stencil.boundaryEntries.RowCount(),
                              "boundary force reduction context or shape differs");
        if (retained->selected != zones || retained->zones.GetDevice() != retained->context.device)
        {
            engine.Synchronize();
            retained->zones = R::MetadataBuffer<int>(zones).CopyTo(retained->context.device, retained->context);
            retained->zones.SetDiagnosticName("boundaryForce/zones", DNDS_ALLOCATION_SITE());
            retained->selected = zones;
        }
        if (retained->context.device == R::Device::Host())
        {
            retained->workerSums.resize(OMP::DistributedThreads());
            R::SubmitArrays(engine, stream, [retained](auto, auto samples, auto entries, auto values)
                            {
                FiniteVolumeStencilView view;
                view.samples = samples; view.boundaryEntries = entries;
                retained->hostResult = CPU::AccumulateBoundaryForce(view, values, retained->zones.HostView(),
                    {retained->workerSums.data(), index(retained->workerSums.size())}); }, R::ReadAs<R::FixedRows<1>>(stencil.samples), R::ReadAs<R::FixedRows<1>>(stencil.boundaryEntries), R::ReadAs<R::FixedRows<3>>(momentum));
            engine.Synchronize();
            return retained->hostResult;
        }
#ifdef DNDS_USE_CUDA
        const auto partialRows = CUDA::BoundaryForceScratchRows(momentum.RowCount());
        if (retained->partial.RowCount() != partialRows)
        {
            engine.Synchronize();
            retained->partial = R::Array<real>(R::RowLayout::Dense(partialRows, 3), retained->context.device);
            retained->result = R::Array<real>(R::RowLayout::Dense(1, 3), retained->context.device);
            retained->completed = R::Array<unsigned int>(R::RowLayout::Dense(1, 1), retained->context.device);
            retained->partial.SetName("boundaryForce/partials", DNDS_ALLOCATION_SITE());
            retained->result.SetName("boundaryForce/result", DNDS_ALLOCATION_SITE());
            retained->completed.SetName("boundaryForce/completedBlocks", DNDS_ALLOCATION_SITE());
        }
        DNDS_check_throw_info(retained->zones.GetDevice() == retained->context.device, "boundary zone metadata residency differs");
        const auto zoneRead = std::make_shared<R::ReadAccess>(retained->zones.Read(retained->context));
        R::SubmitArrays(engine, stream, [retained, zoneRead](auto context, auto samples, auto entries, auto values, auto partial, auto result, auto completed)
                        {
            const auto metadata = zoneRead->View<int>();
            CUDA::AccumulateBoundaryForce(context, samples, entries, values, {metadata.data, index(retained->zones.size())}, partial, result, completed); }, R::ReadAs<R::FixedRows<1>>(stencil.samples), R::ReadAs<R::FixedRows<1>>(stencil.boundaryEntries), R::ReadAs<R::FixedRows<3>>(momentum), R::WriteAs<R::FixedRows<3>>(retained->partial), R::WriteAs<R::FixedRows<3>>(retained->result), R::WriteAs<R::FixedRows<1>>(retained->completed));
        engine.Synchronize();
        zoneRead->Complete(R::Completion::FromExternalCompletion(retained->context));
        const auto host = retained->result.To(R::Device::Host(), stream.Context());
        const auto read = host.Read();
        return {read.View()(0, 0), read.View()(0, 1), read.View()(0, 2)};
#else
        DNDS_check_throw_info(false, "CUDA boundary-force reduction is not compiled");
        return {};
#endif
    }
}
