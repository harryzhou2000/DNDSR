#pragma once

#include "DNDS/Runtime/Array/ArrayView.hpp"
#include <memory>

namespace DNDS::Algebra
{
    /// Each row stores one column-major square matrix (or its singular values).
    /// Views are borrowed; the enclosing operation retains all allocations.
    struct BatchedSVDViews
    {
        Runtime::ArrayView<real> matrix, singularValues, leftVectors, rightVectors;
    };

    /** @brief Fixed-shape FP64 CUDA Jacobi-SVD provider, without hidden copies.
     * @details Construction queries workspace size using already allocated
     * prototype views; it does not read their numerical contents. SubmitBorrowed
     * enqueues on the construction stream, with caller-owned workspace and
     * per-matrix info. There are no host waits or host numerical inputs.
     * The operation must retain this provider and all views until completion.
     * One provider belongs to one execution stream; use another provider for an
     * independent lane. CUDA resource destruction follows retirement of work.
     * Numerical nonconvergence is returned in device info, not an exception;
     * invalid wrappers/provider launch failures use ordinary host error policy.
     */
    class BatchedSVD
    {
        struct Data;
        std::unique_ptr<Data> _data;

    public:
        BatchedSVD(Runtime::ExecutionContext context, int dimension, BatchedSVDViews prototype);
        ~BatchedSVD();
        BatchedSVD(const BatchedSVD &) = delete;
        BatchedSVD &operator=(const BatchedSVD &) = delete;
        index WorkspaceElements() const;
        void SubmitBorrowed(Runtime::ExecutionContext context, BatchedSVDViews values,
                            Runtime::ArrayView<real> workspace, Runtime::ArrayView<int> info);
    };
}
