#pragma once

#include "DNDS/Runtime/Array/ArrayView.hpp"
#include <memory>

namespace DNDS::Algebra
{
    /// Fixed linear cycle parameters. No residual-based inner stopping or inner
    /// Krylov method: ordinary left GMRES requires a fixed linear action.
    struct AmgXSettings
    {
        int cycles = 1;
        int preSweeps = 2, postSweeps = 2, coarseSweeps = 8;
        int maxLevels = 20, minCoarseRows = 16;
        real relaxation = 0.75;
    };

    /** Single-device FP64 AMG for one rank's owned-owned principal matrix.
     * DNDS owns the global operator and every MPI operation. This adapter never
     * receives a communicator. Scalar CSR includes all entries of coupled blocks;
     * callers must not discard off-rank contributions from their global operator.
     *
     * Host wrappers below borrow views for the duration of the call. Callers hold
     * leases/queued operand reservations and this object through completion.
     * AmgX has no application-stream argument: calls wait for the supplied CUDA
     * stream, run the provider, then synchronize its device before returning.
     * This is an explicit host boundary, not a captureable native-stream kernel.
     * The scheduler must drain pending host-progressed communication before this
     * boundary; a device-wide provider join must not wait on MPI progress that
     * only the currently blocked controller could perform.
     * Provider failures throw here; production collective solver catches abort.
     * A pattern/value change is forbidden during a GMRES solve by the caller.
     */
    class AmgXPreconditioner
    {
        struct State;
        std::unique_ptr<State> _state;

    public:
        static bool Available() noexcept;
        explicit AmgXPreconditioner(Runtime::Device device, AmgXSettings settings = {});
        ~AmgXPreconditioner();
        AmgXPreconditioner(const AmgXPreconditioner &) = delete;
        AmgXPreconditioner &operator=(const AmgXPreconditioner &) = delete;

        /// Copies host scalar CSR indices into private Runtime Arrays. Columns
        /// must be sorted, unique and local, and each row must contain a diagonal.
        /// Zero-row ranks use offsets={0}, columns={} and perform an empty apply.
        void SetPattern(Runtime::ArrayView<const int> offsets,
                        Runtime::ArrayView<const int> columns);
        /// Device scalar values ordered like columns. Upload/rebuild is outside
        /// the Krylov solve. Repeated calls reuse the matrix's published pattern.
        void UpdateValuesBorrowed(Runtime::ExecutionContext context,
                                  Runtime::ArrayView<const real> values);
        /// Apply from zero with fixed cycles. Input/output are compact scalar
        /// tables (N x 1), never aliased. Completion is host-visible on return.
        void ApplyBorrowed(Runtime::ExecutionContext context,
                           Runtime::ArrayView<const real> residual,
                           Runtime::ArrayView<real> correction);
        Runtime::Device GetDevice() const;
        index RowNumber() const;
    };
}
