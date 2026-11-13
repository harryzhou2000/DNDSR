#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeReconstructionKernels.hpp"
#include "Cauchy/State/CPU/FieldKernels.hpp"
#include "DNDS/Runtime/Execution/EagerArrayBatch.hpp"

namespace DNDS::Cauchy
{
    void FiniteVolume::Workspace::SubmitReconstructionNorm(
        R::EagerEngine &engine, const R::ExecutionStream &stream, const EquationDofs &current)
    {
        DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == owner->metrics.Device() &&
                                  current.Device() == owner->metrics.Device() && current.PhysicalFieldNumber() == index(fields.size()),
                              "reconstruction norm stream, placement or grouping differs");
        if (!reconstructionNorm)
        {
            const index rows = owner->metrics.NumCell();
            const index partials = owner->metrics.Device() == R::Device::Host() ? OMP::DistributedThreads() : std::max<index>(1, std::min<index>(256, (rows + 127) / 128));
            reconstructionNorm.emplace(ReconstructionNormStorage{
                R::Array<real>(R::RowLayout::Dense(partials, 2), owner->metrics.Device()),
                R::Array<real>(R::RowLayout::Dense(1, 2), owner->metrics.Device())});
            reconstructionNorm->partial.SetName("Cauchy/reconstructionNormPartials", DNDS_ALLOCATION_SITE());
            reconstructionNorm->result.SetName("Cauchy/reconstructionNorm", DNDS_ALLOCATION_SITE());
        }
        std::vector<R::ArrayOperand<real, false, R::UniformTableRows>> reads;
        for (index iField = 0; iField < current.PhysicalFieldNumber(); ++iField)
        {
            DNDS_check_throw_info(fields[iField].reconstruction, "reconstruction norm has no iteration workspace");
            const auto &a = current.PhysicalField(iField).Owned();
            const auto &b = fields[iField].reconstruction->temporary.Owned();
            DNDS_check_throw_info(a.RowCount() == owner->metrics.NumCell() && b.RowCount() == a.RowCount() &&
                                      a.Layout().Width() == b.Layout().Width(),
                                  "reconstruction norm field shape differs");
            reads.push_back(R::ReadAs<R::UniformTableRows>(a));
            reads.push_back(R::ReadAs<R::UniformTableRows>(b));
        }
        std::vector<R::ArrayOperand<real, true, R::FixedRows<2>>> writes{
            R::WriteAs<R::FixedRows<2>>(reconstructionNorm->partial), R::WriteAs<R::FixedRows<2>>(reconstructionNorm->result)};
        R::SubmitArrayBatch(engine, stream, [](R::ExecutionContext context, auto inputs, auto outputs)
                            {
            if (context.device == R::Device::Host())
            {
                std::array<real, 2> norms{};
                for (index iField = 0; iField < inputs.size() / 2; ++iField)
                    CPU::AccumulateReconstructionUpdateNorm(inputs[2 * iField], inputs[2 * iField + 1], {norms.data(), 2}, outputs[0]);
                for (int j = 0; j < 2; ++j) outputs[1](0, j) = norms[j];
            }
            else
            {
#ifdef DNDS_USE_CUDA
                DNDS_check_throw_info(inputs.size() == 2, "CUDA reconstruction norm requires one combined field");
                CUDA::ReconstructionUpdateNorm(context, inputs[0], inputs[1], outputs[0], outputs[1]);
#else
                DNDS_check_throw_info(false, "CUDA reconstruction norms are not built");
#endif
            } }, std::move(reads), std::move(writes));
    }
    std::array<real, 2> FiniteVolume::Workspace::ReadReconstructionNorm() const
    {
        DNDS_check_throw_info(reconstructionNorm && reconstructionNorm->result.Storage().IsIdle(),
                              "reconstruction norms require a completed segment");
        const auto host = reconstructionNorm->result.To(R::Device::Host());
        const auto read = host.Read();
        return {read.View()(0, 0), read.View()(0, 1)};
    }
}
