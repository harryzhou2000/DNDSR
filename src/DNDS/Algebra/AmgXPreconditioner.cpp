#include "DNDS/Algebra/AmgXPreconditioner.hpp"
#include "DNDS/Runtime/Array/Array.hpp"
#include <cmath>
#include <limits>
#include <mutex>
#include <sstream>
#ifdef DNDS_USE_AMGX
#    include <amgx_c.h>
#    include <cuda_runtime_api.h>
#endif

namespace DNDS::Algebra
{
    namespace
    {
        template <class T>
        void CheckScalarTable(Runtime::ArrayView<T> view, Runtime::Device device)
        {
            view.ValidateDevice(device);
            DNDS_check_throw_info(view.kind == Runtime::RowLayoutKind::Table && !view.offsets && !view.sizes &&
                                      view.width == 1 && view.stride == 1 && (!view.rows || view.data),
                                  "AmgX requires compact scalar tables");
        }
#ifdef DNDS_USE_AMGX
        // The provider has process-global initialization and configuration state.
        // Serialize adapter calls, not numerical work in unrelated DNDS streams.
        struct ProviderRuntime
        {
            std::mutex mutex;
            unsigned users = 0;
        };
        ProviderRuntime &Provider()
        {
            static ProviderRuntime provider;
            return provider;
        }
        void Check(AMGX_RC code)
        {
            if (code == AMGX_RC_OK)
                return;
            char message[1024]{};
            AMGX_get_error_string(code, message, sizeof(message));
            DNDS_check_throw_info(false, std::string("AmgX: ") + message);
        }
        void CheckCUDA(cudaError_t code)
        {
            DNDS_check_throw_info(code == cudaSuccess, std::string("AmgX CUDA boundary: ") + cudaGetErrorString(code));
        }
        struct DeviceScope
        {
            int previous = -1;
            explicit DeviceScope(int ordinal)
            {
                CheckCUDA(cudaGetDevice(&previous));
                CheckCUDA(cudaSetDevice(ordinal));
            }
            ~DeviceScope()
            {
                if (cudaSetDevice(previous) != cudaSuccess)
                    std::abort();
            }
        };
        std::string Configuration(const AmgXSettings &s)
        {
            // NOSOLVER uses fixed coarsest_sweeps of the same Jacobi smoother.
            // monitor_residual=0 prevents tolerance-dependent nonlinear action.
            std::ostringstream text;
            text << R"({"config_version":2,"determinism_flag":1,"solver":{"solver":"AMG",)"
                 << R"("algorithm":"AGGREGATION","selector":"SIZE_2","cycle":"V",)"
                 << R"("smoother":"BLOCK_JACOBI","coarse_solver":"NOSOLVER","monitor_residual":0,)"
                 << R"("print_solve_stats":0,"print_grid_stats":0,"max_iters":)" << s.cycles
                 << R"(,"presweeps":)" << s.preSweeps << R"(,"postsweeps":)" << s.postSweeps
                 << R"(,"coarsest_sweeps":)" << s.coarseSweeps << R"(,"max_levels":)" << s.maxLevels
                 << R"(,"min_coarse_rows":)" << s.minCoarseRows << R"(,"relaxation_factor":)" << s.relaxation << "}}";
            return text.str();
        }
#endif
    }

    struct AmgXPreconditioner::State
    {
        Runtime::Device device;
        Runtime::Array<int> offsets, columns;
        index rows = 0, entries = 0;
        bool pattern = false, uploaded = false, ready = false;
#ifdef DNDS_USE_AMGX
        bool initialized = false;
        AMGX_config_handle config = nullptr;
        AMGX_resources_handle resources = nullptr;
        AMGX_matrix_handle matrix = nullptr;
        AMGX_vector_handle rhs = nullptr, solution = nullptr;
        AMGX_solver_handle solver = nullptr;
        ~State()
        {
            if (!initialized)
                return;
            std::lock_guard<std::mutex> lock(Provider().mutex);
            // Destruction is rank-local and cannot throw or start MPI cleanup.
            int previous = -1;
            if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(device.ordinal) != cudaSuccess)
                std::abort();
            auto release = [](AMGX_RC rc)
            { if (rc != AMGX_RC_OK) std::abort(); };
            if (solver)
                release(AMGX_solver_destroy(solver));
            if (solution)
                release(AMGX_vector_destroy(solution));
            if (rhs)
                release(AMGX_vector_destroy(rhs));
            if (matrix)
                release(AMGX_matrix_destroy(matrix));
            if (resources)
                release(AMGX_resources_destroy(resources));
            if (config)
                release(AMGX_config_destroy(config));
            if (!--Provider().users)
                release(AMGX_finalize());
            if (cudaSetDevice(previous) != cudaSuccess)
                std::abort();
        }
#endif
    };

    bool AmgXPreconditioner::Available() noexcept
    {
#ifdef DNDS_USE_AMGX
        return true;
#else
        return false;
#endif
    }
    AmgXPreconditioner::AmgXPreconditioner(Runtime::Device device, AmgXSettings settings)
        : _state(std::make_unique<State>())
    {
        DNDS_check_throw_info(Available(), "AmgX requested but DNDS_USE_AMGX is disabled");
        DNDS_check_throw_info(device.IsValid() && device.space == Runtime::MemorySpace::CUDA, "AmgX needs one CUDA device");
        DNDS_check_throw_info(settings.cycles > 0 && settings.preSweeps >= 0 && settings.postSweeps >= 0 &&
                                  settings.coarseSweeps > 0 && settings.maxLevels > 0 && settings.minCoarseRows > 0 &&
                                  std::isfinite(settings.relaxation) && settings.relaxation > 0 && settings.relaxation <= 1,
                              "invalid fixed-cycle AmgX settings");
        _state->device = device;
#ifdef DNDS_USE_AMGX
        std::lock_guard<std::mutex> lock(Provider().mutex);
        DeviceScope scope(device.ordinal);
        if (!Provider().users)
            Check(AMGX_initialize());
        ++Provider().users;
        _state->initialized = true;
        Check(AMGX_config_create(&_state->config, Configuration(settings).c_str()));
        Check(AMGX_resources_create(&_state->resources, _state->config, nullptr, 1, &device.ordinal));
        Check(AMGX_matrix_create(&_state->matrix, _state->resources, AMGX_mode_dDDI));
        Check(AMGX_vector_create(&_state->rhs, _state->resources, AMGX_mode_dDDI));
        Check(AMGX_vector_create(&_state->solution, _state->resources, AMGX_mode_dDDI));
        Check(AMGX_solver_create(&_state->solver, _state->resources, AMGX_mode_dDDI, _state->config));
#endif
    }
    AmgXPreconditioner::~AmgXPreconditioner() = default;
    Runtime::Device AmgXPreconditioner::GetDevice() const { return _state->device; }
    index AmgXPreconditioner::RowNumber() const { return _state->rows; }

    void AmgXPreconditioner::SetPattern(Runtime::ArrayView<const int> offsets, Runtime::ArrayView<const int> columns)
    {
        CheckScalarTable(offsets, Runtime::Device::Host());
        CheckScalarTable(columns, Runtime::Device::Host());
        DNDS_check_throw_info(offsets.rows >= 1 && offsets.rows - 1 <= std::numeric_limits<int>::max() &&
                                  columns.rows <= std::numeric_limits<int>::max(),
                              "AmgX scalar CSR exceeds int32 range");
        const index rows = offsets.rows - 1;
        DNDS_check_throw_info(offsets(0, 0) == 0 && offsets(rows, 0) == columns.rows, "invalid AmgX CSR endpoints");
        for (index iRow = 0; iRow < rows; ++iRow)
        {
            const int begin = offsets(iRow, 0), end = offsets(iRow + 1, 0);
            DNDS_check_throw_info(begin >= 0 && begin <= end && end <= columns.rows, "invalid AmgX CSR row");
            bool diagonal = false;
            int previous = -1;
            for (index j = begin; j < end; ++j)
            {
                const int column = columns(j, 0);
                DNDS_check_throw_info(column > previous && column < rows, "AmgX requires sorted unique local columns");
                previous = column;
                diagonal |= column == iRow;
            }
            DNDS_check_throw_info(diagonal, "AmgX row is missing its diagonal");
        }
        Runtime::Array<int> ownOffsets(Runtime::RowLayout::Dense(offsets.rows, 1));
        Runtime::Array<int> ownColumns(Runtime::RowLayout::Dense(columns.rows, 1));
        {
            auto a = ownOffsets.Write(), b = ownColumns.Write();
            for (index iRow = 0; iRow < offsets.rows; ++iRow)
                a.View()(iRow, 0) = offsets(iRow, 0);
            for (index iRow = 0; iRow < columns.rows; ++iRow)
                b.View()(iRow, 0) = columns(iRow, 0);
        }
        ownOffsets.SetName("AmgX/scalarCSR/offsets", DNDS_ALLOCATION_SITE());
        ownColumns.SetName("AmgX/scalarCSR/columns", DNDS_ALLOCATION_SITE());
        _state->offsets = std::move(ownOffsets);
        _state->columns = std::move(ownColumns);
        _state->rows = rows;
        _state->entries = columns.rows;
        _state->pattern = true;
        _state->uploaded = _state->ready = false;
    }

    void AmgXPreconditioner::UpdateValuesBorrowed(Runtime::ExecutionContext context, Runtime::ArrayView<const real> values)
    {
        DNDS_check_throw_info(context.device == GetDevice() && _state->pattern, "AmgX update needs its device and pattern");
        CheckScalarTable(values, GetDevice());
        DNDS_check_throw_info(values.rows == _state->entries, "AmgX scalar value count mismatch");
        _state->ready = false;
#ifdef DNDS_USE_AMGX
        std::lock_guard<std::mutex> lock(Provider().mutex);
        DeviceScope scope(context.device.ordinal);
        CheckCUDA(cudaStreamSynchronize(static_cast<cudaStream_t>(context.stream)));
        if (_state->rows)
        {
            if (_state->uploaded)
                Check(AMGX_matrix_replace_coefficients(_state->matrix, int(_state->rows), int(_state->entries), values.data, nullptr));
            else
            {
                auto offsets = _state->offsets.Read(), columns = _state->columns.Read();
                Check(AMGX_matrix_upload_all(_state->matrix, int(_state->rows), int(_state->entries), 1, 1,
                                             offsets.View().data, columns.View().data, values.data, nullptr));
                Check(AMGX_vector_bind(_state->rhs, _state->matrix));
                Check(AMGX_vector_bind(_state->solution, _state->matrix));
            }
            Check(AMGX_solver_setup(_state->solver, _state->matrix));
            CheckCUDA(cudaDeviceSynchronize());
        }
        _state->uploaded = _state->ready = true;
#endif
    }
    void AmgXPreconditioner::ApplyBorrowed(Runtime::ExecutionContext context, Runtime::ArrayView<const real> residual,
                                           Runtime::ArrayView<real> correction)
    {
        DNDS_check_throw_info(context.device == GetDevice() && _state->ready, "AmgX apply needs its device and completed setup");
        CheckScalarTable(residual, GetDevice());
        CheckScalarTable(correction, GetDevice());
        DNDS_check_throw_info(residual.rows == _state->rows && correction.rows == _state->rows &&
                                  (!_state->rows || residual.data != correction.data),
                              "AmgX vector shape or alias mismatch");
#ifdef DNDS_USE_AMGX
        std::lock_guard<std::mutex> lock(Provider().mutex);
        DeviceScope scope(context.device.ordinal);
        CheckCUDA(cudaStreamSynchronize(static_cast<cudaStream_t>(context.stream)));
        if (!_state->rows)
            return;
        Check(AMGX_vector_upload(_state->rhs, int(_state->rows), 1, residual.data));
        Check(AMGX_vector_set_zero(_state->solution, int(_state->rows), 1));
        Check(AMGX_solver_solve_with_0_initial_guess(_state->solver, _state->rhs, _state->solution));
        AMGX_SOLVE_STATUS status;
        Check(AMGX_solver_get_status(_state->solver, &status));
        DNDS_check_throw_info(status == AMGX_SOLVE_SUCCESS || status == AMGX_SOLVE_NOT_CONVERGED,
                              "AmgX fixed-cycle preconditioner failed or diverged");
        Check(AMGX_vector_download(_state->solution, correction.data));
        CheckCUDA(cudaDeviceSynchronize());
#endif
    }
}
