#pragma once
#include "DNDS/Runtime/EagerArray.hpp"

namespace DNDS::Runtime::Demo
{
    using State = Array<double, FixedRows<5>>;
    DNDS_DEVICE_CALLABLE inline void Cell(ArrayView<const double> owned, ArrayView<const double> ghost,
                                          ArrayView<double> output, size_t i)
    {
        for (size_t j = 0; j < 5; ++j)
        {
            const double left = i ? owned(i - 1, j) : ghost(0, j);
            const double right = i + 1 < owned.Rows() ? owned(i + 1, j) : ghost(1, j);
            output(i, j) = .5 * owned(i, j) + .25 * (left + right);
        }
    }
#ifdef DNDS_USE_CUDA
    void StencilCUDA(EagerEngine &, const ExecutionStream &, State, State, State, bool);
#endif
    inline void Stencil(EagerEngine &engine, const ExecutionStream &stream, State owned, State ghost, State output, bool core)
    {
        DNDS_check_throw_info(owned.Rows() >= 3 && ghost.Rows() == 2 && output.Rows() == owned.Rows(), "mock stencil shape mismatch");
#ifdef DNDS_USE_CUDA
        if (stream.Context().device.space == MemorySpace::CUDA)
            return StencilCUDA(engine, stream, std::move(owned), std::move(ghost), std::move(output), core);
#endif
        if (core)
            SubmitArrays(engine, stream, [](ExecutionContext, ArrayView<const double> u, ArrayView<double> v)
                         { for (size_t i = 1; i + 1 < u.Rows(); ++i) Cell(u, {}, v, i); }, Read(owned), Write(output));
        else
            SubmitArrays(engine, stream, [](ExecutionContext, ArrayView<const double> u, ArrayView<const double> g, ArrayView<double> v)
                         { Cell(u, g, v, 0); Cell(u, g, v, u.Rows() - 1); }, Read(owned), Read(ghost), ReadWrite(output));
    }
}
