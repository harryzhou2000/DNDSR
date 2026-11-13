#include "Cauchy/Implicit/LocalAMGPreconditioner.hpp"
#include "Cauchy/Implicit/CUDA/LocalAMGKernels.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include <algorithm>
#include <climits>

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    struct LocalAMGPreconditioner::State
    {
        std::shared_ptr<const Algebra::DofSpace::Partition> partition;
        std::shared_ptr<Algebra::AmgXPreconditioner> provider;
        R::Array<index> offsets, sources;
        R::Array<real> values, rhs, correction;
        R::ArrayStorage::Shared diagonalIdentity, offDiagonalIdentity;
        index components;
        uint64_t diagonalEpoch = 0, offDiagonalEpoch = 0;
        bool current = false, full, nativeBlock;
    };
    LocalAMGPreconditioner::LocalAMGPreconditioner(const LinearProblemContext &problem, Algebra::AmgXSettings settings)
        : _state(std::make_shared<State>())
    {
        problem.Validate();
        DNDS_check_throw_info(problem.Device().space == R::MemorySpace::CUDA && problem.HasAssembledOffDiagonal() &&
                                  problem.Components() >= 5 && problem.Components() <= 7,
                              "local AmgX requires assembled CUDA combined-AoS 5/6/7 blocks");
        const index rows = index(problem.Space().OwnedRows()), n = problem.Components();
        DNDS_check_throw_info(rows <= INT_MAX / n, "AmgX scalar row numbering exceeds int32");
        auto &data = *_state;
        data.partition = problem.Space().Pin();
        data.components = n;
        // Keep flow-five components in one aggregate. Scalar expansion allows
        // coarsening to mix physically unrelated components (including the
        // exactly-zero transverse equation in planar flow), producing a poor
        // preconditioner. Pinned AmgX LOW_DEG supports native 5, but not 6/7.
        data.nativeBlock = n == 5;
        data.full = problem.DiagonalMode() == DiagonalStorage::Block;
        data.diagonalIdentity = problem.Diagonal().SharedStorage();
        data.offDiagonalIdentity = problem.OffDiagonal().Values().SharedStorage();
        const auto columns = problem.Columns();
        std::vector<index> offsets{0}, sources;
        std::vector<int> providerOffsets{0}, providerColumns;
        index globalBlock = 0;
        for (index iCell = 0; iCell < rows; ++iCell)
        {
            std::vector<std::pair<index, index>> local{{iCell, -1}};
            for (index k = 0; k < index(columns[iCell].size()); ++k)
                if (columns[iCell][k] < rows)
                    local.emplace_back(columns[iCell][k], globalBlock + k);
            globalBlock += index(columns[iCell].size());
            std::sort(local.begin(), local.end());
            DNDS_check_throw_info(index(providerColumns.size()) <= INT_MAX - index(local.size()) * n * n,
                                  "AmgX scalar entry numbering exceeds int32");
            for (const auto &block : local)
                sources.push_back(block.second);
            offsets.push_back(index(sources.size()));
            for (index i = 0; i < (data.nativeBlock ? 1 : n); ++i)
            {
                for (const auto &block : local)
                    if (data.nativeBlock)
                        providerColumns.push_back(int(block.first));
                    else
                        for (index j = 0; j < n; ++j)
                            providerColumns.push_back(int(block.first * n + j));
                providerOffsets.push_back(int(providerColumns.size()));
            }
        }
        const auto make = [](const auto &values)
        {
            using T = typename std::decay_t<decltype(values)>::value_type;
            R::Array<T> array(R::RowLayout::Dense(values.size(), 1));
            auto write = array.Write();
            for (index i = 0; i < index(values.size()); ++i)
                write.View()(i, 0) = values[i];
            return array;
        };
        const auto hostOffsets = make(providerOffsets), hostColumns = make(providerColumns);
        const auto o = hostOffsets.Read(), c = hostColumns.Read();
        data.provider = std::make_shared<Algebra::AmgXPreconditioner>(problem.Device(), settings);
        data.provider->SetPattern(o.View(), c.View(), data.nativeBlock ? int(n) : 1);
        data.offsets = make(offsets).To(problem.Device());
        data.sources = make(sources).To(problem.Device());
        data.values = R::Array<real>(R::RowLayout::Dense(index(sources.size()) * n * n, 1), problem.Device());
        data.rhs = R::Array<real>(R::RowLayout::Dense(0, 1), problem.Device());
        data.correction = R::Array<real>(data.rhs.Layout(), problem.Device());
        data.offsets.SetName("Cauchy/localAMG/blockOffsets", DNDS_ALLOCATION_SITE());
        data.sources.SetName("Cauchy/localAMG/sourceBlocks", DNDS_ALLOCATION_SITE());
        data.values.SetName("Cauchy/localAMG/matrixValues", DNDS_ALLOCATION_SITE());
        data.rhs.SetName("Cauchy/localAMG/rhs", DNDS_ALLOCATION_SITE());
        data.correction.SetName("Cauchy/localAMG/correction", DNDS_ALLOCATION_SITE());
    }
    void LocalAMGPreconditioner::Update(R::EagerEngine &engine, const R::ExecutionStream &stream, const LinearProblemContext &problem)
    {
        auto &data = *_state;
        DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == data.provider->GetDevice() &&
                                  problem.Space().Pin() == data.partition && problem.HasAssembledOffDiagonal() && problem.Components() == data.components &&
                                  problem.Diagonal().SharedStorage() == data.diagonalIdentity && problem.OffDiagonal().Values().SharedStorage() == data.offDiagonalIdentity,
                              "local AmgX matrix/lane identity changed");
        const uint64_t diagonalEpoch = problem.Diagonal().MutationEpoch(), offEpoch = problem.OffDiagonal().Values().MutationEpoch();
        if (data.current && data.diagonalEpoch == diagonalEpoch && data.offDiagonalEpoch == offEpoch)
            return;
        data.current = false;
        // Finish host-progressed transfers before any AmgX device-wide wait.
        engine.Synchronize();
#ifdef DNDS_USE_CUDA
        R::SubmitArrays(engine, stream, [pin = _state](R::ExecutionContext context, auto offsets, auto sources, auto diagonal, auto blocks, auto values)
                        { CUDA::PackLocalAMGMatrix(context, offsets, sources, diagonal, blocks, values, int(pin->components), pin->full, pin->nativeBlock); }, R::ReadAs<R::FixedRows<1>>(data.offsets), R::ReadAs<R::FixedRows<1>>(data.sources), R::ReadAs<R::TableRows>(problem.Diagonal()), R::ReadAs<R::TableRows>(problem.OffDiagonal().Values()), R::WriteAs<R::TableRows>(data.values));
        engine.Synchronize();
        const auto read = data.values.Read(stream.Context());
        data.provider->UpdateValuesBorrowed(stream.Context(), read.View());
        data.diagonalEpoch = diagonalEpoch;
        data.offDiagonalEpoch = offEpoch;
        data.current = true;
#else
        DNDS_check_throw_info(false, "CUDA local AmgX packing is not compiled");
#endif
    }
    void LocalAMGPreconditioner::Apply(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                       const R::Array<real> &rhs, R::Array<real> &output)
    {
        auto &data = *_state;
        DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == data.provider->GetDevice() && data.current &&
                                  !rhs.Storage().Overlaps(output.Storage()),
                              "local AmgX input/lane aliases or matrix is stale");
        for (const auto *array : {&rhs, static_cast<const R::Array<real> *>(&output)})
            DNDS_check_throw_info(array->GetDevice() == data.provider->GetDevice() && array->Layout().Kind() == R::RowLayoutKind::Table &&
                                      array->RowCount() == index(data.partition->owned) && array->Layout().Width() == data.components,
                                  "local AmgX vector shape differs");
        engine.Synchronize();
#ifdef DNDS_USE_CUDA
        // Compact AoS can be borrowed directly as a scalar table; padded AoS
        // uses reserved device scratch. No physical field is copied to host.
        const bool compact = rhs.Layout().Stride() == data.components && output.Layout().Stride() == data.components;
        if (!compact)
        {
            if (data.rhs.RowCount() != index(data.partition->owned) * data.components)
            {
                data.rhs = R::Array<real>(R::RowLayout::Dense(index(data.partition->owned) * data.components, 1), data.provider->GetDevice());
                data.correction = R::Array<real>(data.rhs.Layout(), data.provider->GetDevice());
                data.rhs.SetName("Cauchy/localAMG/rhs", DNDS_ALLOCATION_SITE());
                data.correction.SetName("Cauchy/localAMG/correction", DNDS_ALLOCATION_SITE());
            }
            R::SubmitArrays(engine, stream, [](R::ExecutionContext context, auto source, auto target)
                            { CUDA::PackLocalAMGVector(context, source, target); }, R::ReadAs<R::TableRows>(rhs), R::WriteAs<R::TableRows>(data.rhs));
            engine.Synchronize();
        }
        {
            const auto read = (compact ? rhs : data.rhs).Read(stream.Context());
            auto write = (compact ? output : data.correction).Write(stream.Context());
            auto source = read.View();
            auto target = write.View();
            if (compact)
            {
                source.rows *= data.components;
                source.width = source.stride = 1;
                target.rows *= data.components;
                target.width = target.stride = 1;
            }
            data.provider->ApplyBorrowed(stream.Context(), source, target);
        }
        if (!compact)
        {
            R::SubmitArrays(engine, stream, [](R::ExecutionContext context, auto source, auto target)
                            { CUDA::UnpackLocalAMGVector(context, source, target); }, R::ReadAs<R::TableRows>(data.correction), R::WriteAs<R::TableRows>(output));
            engine.Synchronize();
        }
#else
        DNDS_check_throw_info(false, "CUDA local AmgX application is not compiled");
#endif
    }
}
