#pragma once

#include "Geom/Transition/ReorderPlan.hpp"
#include "DNDS/Runtime/Field.hpp"
#include "DNDS/Runtime/Exchange.hpp"
#include "DNDS/Runtime/RowTransfer.hpp"
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <optional>

namespace DNDS::Geom
{
    namespace FieldTransferDetail
    {
        /// Agree on a canonical earlier payload without assuming every rank's
        /// local alias inventory is the same.
        inline int AgreeAliasCandidate(const MPIInfo &mpi, int candidate, size_t relationCount)
        {
            DNDS_check_throw_info(relationCount <= size_t(INT_MAX), "alias agreement range exceeds MPI int range");
            const auto local = candidate >= 0 ? std::optional<int>(candidate) : std::nullopt;
            return MPI::AgreeOptionalInteger(mpi, local, int(relationCount)).value_or(-1);
        }

        /// Every rank runs local work before agreement and enters the next
        /// collective only after all ranks succeeded. Payload/route constructors
        /// retain their own collective schema and extent checks.
        template <class F>
        void LocalStage(const MPIInfo &mpi, F &&action)
        {
            int valid = 1;
            std::exception_ptr failure;
            try
            {
                action();
            }
            catch (...)
            {
                valid = 0;
                failure = std::current_exception();
            }
            MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, mpi.comm);
            // Preserve the originating error after collective agreement. Other
            // ranks still fail before entering the next payload operation.
            if (failure)
                std::rethrow_exception(failure);
            DNDS_check_throw_info(valid, "field preparation failed before collective payload; source retained");
        }

        template <class Value>
        struct Prepared final : PreparedCompanion
        {
            Value *destination;
            Value next;
            Prepared(Value &source, Value value) : destination(&source), next(std::move(value)) {}
            void Publish() noexcept override
            {
                static_assert(std::is_nothrow_move_assignable_v<Value>, "field publication must not throw");
                *destination = std::move(next);
            }
        };
    }

    /// Extract the exact correspondence once, then share it across native Arrays
    /// (including separate SoA components). No geometry or interpolation in DNDS.
    inline Runtime::RowTransferPlan NativeRowTransfer(const PermutationTransfer &transfer, const MPIInfo &mpi)
    {
        // Agree on the mode before taking branches with different collectives.
        int mode = transfer.isLocalOnly, minMode = 0, maxMode = 0;
        MPI_Allreduce(&mode, &minMode, 1, MPI_INT, MPI_MIN, mpi.comm);
        MPI_Allreduce(&mode, &maxMode, 1, MPI_INT, MPI_MAX, mpi.comm);
        DNDS_check_throw_info(minMode == maxMode, "native row transfer modes disagree");
        std::vector<std::vector<index>> send, receive;
        std::vector<int> sendCounts, receiveCounts;
        index destinationRows = 0;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            const index rows = index(transfer.size());
            send.resize(mpi.size);
            receive.resize(mpi.size);
            if (transfer.isLocalOnly)
            {
                DNDS_check_throw_info(transfer.localOld2New.size() == rows, "invalid native local permutation extent");
                std::vector<bool> seen(rows, false);
                send[mpi.rank].reserve(rows);
                receive[mpi.rank].reserve(rows);
                for (index i = 0; i < rows; ++i)
                {
                    const auto target = transfer.localOld2New[i];
                    DNDS_check_throw_info(target >= 0 && size_t(target) < rows && !seen[size_t(target)], "invalid native local permutation");
                    seen[size_t(target)] = true;
                    send[mpi.rank].push_back(i);
                    receive[mpi.rank].push_back(target);
                }
                destinationRows = rows;
            }
            else
            {
                DNDS_check_throw_info(transfer.pushStart.size() == size_t(mpi.size + 1) &&
                                          transfer.newGlobalOffsets.size() == size_t(mpi.size + 1) &&
                                          transfer.pushStart.front() == 0 && transfer.pushIndex.size() == rows &&
                                          transfer.pushStart.back() == transfer.size(), "invalid native distributed route extent");
                sendCounts.resize(mpi.size);
                receiveCounts.resize(mpi.size);
                std::vector<bool> seen(size_t(rows), false);
                for (int peer = 0; peer < mpi.size; ++peer)
                {
                    const auto first = transfer.pushStart[peer], last = transfer.pushStart[peer + 1];
                    DNDS_check_throw_info(first >= 0 && last >= first && size_t(last) <= rows && last - first <= INT_MAX,
                                          "invalid native push offsets");
                    DNDS_check_throw_info(transfer.newGlobalOffsets[peer] >= 0 &&
                                              transfer.newGlobalOffsets[peer + 1] >= transfer.newGlobalOffsets[peer],
                                          "invalid native destination offsets");
                    sendCounts[peer] = int(last - first);
                    send[peer].reserve(size_t(sendCounts[peer]));
                    for (DNDS::index j = first; j < last; ++j)
                    {
                        const auto row = transfer.pushIndex[size_t(j)];
                        DNDS_check_throw_info(row >= 0 && size_t(row) < rows && !seen[size_t(row)] &&
                                                  transfer.targetRanks[size_t(row)] == peer, "invalid native push row");
                        seen[size_t(row)] = true;
                        send[peer].push_back(row);
                    }
                }
                destinationRows = transfer.newGlobalOffsets[mpi.rank + 1] - transfer.newGlobalOffsets[mpi.rank];
            } });
        if (!transfer.isLocalOnly)
        {
            MPI_Alltoall(sendCounts.data(), 1, MPI_INT, receiveCounts.data(), 1, MPI_INT, mpi.comm);
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                // ArrayTransformer push receives contiguous sender-rank buckets,
                // retaining each sender's pushIndex order within its bucket.
                index next = 0;
                for (int peer = 0; peer < mpi.size; ++peer)
                {
                    receive[peer].reserve(size_t(receiveCounts[peer]));
                    for (int j = 0; j < receiveCounts[peer]; ++j)
                        receive[peer].push_back(next++);
                }
                DNDS_check_throw_info(next == destinationRows, "native destination offsets disagree with received rows"); });
        }
        return Runtime::RowTransferPlan::Known(std::make_shared<Runtime::CommunicationContext>(mpi.comm),
                                               index(transfer.size()), destinationRows, std::move(send), std::move(receive));
    }

    template <class T, class L = Runtime::RowLayout>
    CompanionEntry RegisterArray(Runtime::Array<T> &array, EntityKind kind, std::string name,
                                 Runtime::ElementSchema schema = Runtime::ElementSchema::Arithmetic<T>(),
                                 Runtime::ExecutionContext context = Runtime::ExecutionContext::Host(),
                                 CompanionDisposition disposition = CompanionDisposition::TransferIfReordered)
    {
        CompanionEntry entry{kind, {}, std::move(name)};
        entry.disposition = disposition;
        entry.preflight = [&array, schema, context](const PermutationTransfer &transfer)
        {
            DNDS_check_throw_info(array.RowCount() == size_t(transfer.size()) && array.Storage().IsIdle() && array.IsValid() &&
                                      schema.Extent() == sizeof(T) && schema.Compatible(Runtime::ElementSchema::TypeIdentity<T>()) &&
                                      context.device == array.GetDevice() &&
                                      (context.device != Runtime::Device::Host() || !context.stream),
                                  "native Array migration requires idle valid compatible owned rows/context");
        };
        entry.attachment = [&array, schema]()
        {
            DNDS_check_throw_info(array.IsValid() && array.Storage().IsIdle(), "native Array attachment requires valid idle storage");
            return CompanionAttachment{schema.Identity() + ":" + Runtime::ElementSchema::TypeIdentity<T>() + ":" +
                                           std::to_string(static_cast<int>(array.GetDevice().space)),
                                       {},
                                       false,
                                       0,
                                       0,
                                       &array,
                                       array.Identity(),
                                       array.Version(),
                                       array.ShapeVersion(),
                                       size_t(array.RowCount()),
                                       size_t(array.RowCount()),
                                       0};
        };
        entry.prepare = [&array, schema, context](const PermutationTransfer &transfer, const MPIInfo &mpi)
        {
            auto route = NativeRowTransfer(transfer, mpi);
            std::optional<Runtime::RowLayout> layout;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                        layout.emplace(route.DestinationLayout(array.Layout()));
                        if constexpr (!std::is_same_v<L, Runtime::RowLayout>)
                            layout = Runtime::RowLayout::Dense(layout->RowCount(), L::Default().Width()); });
            std::unique_ptr<FieldTransferDetail::Prepared<Runtime::Array<T, L>>> next;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            { next = std::make_unique<FieldTransferDetail::Prepared<Runtime::Array<T, L>>>(array,
                                                                                                                           Runtime::Array<T, L>(*layout, array.GetDevice())); });
            Runtime::PayloadPlan(route, array.Layout(), next->next.Layout(), schema).Bind(array, next->next).Begin(context).Wait();
            return next;
        };
        auto validate = entry.preflight;
        auto prepare = entry.prepare;
        entry.fn = [validate, prepare, disposition](const PermutationTransfer &transfer, const MPIInfo &mpi)
        {
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                DNDS_check_throw_info(disposition == CompanionDisposition::TransferIfReordered, "field disposition prohibits reorder");
                validate(transfer); });
            auto next = prepare(transfer, mpi);
            next->Publish();
        };
        return entry;
    }

    /// Adapt an owned native field to the existing mesh permutation machinery.
    /// Stable IDs, states and time histories are ordinary registered fields.
    /// The initial implementation is host-only; ghosts are rebuilt separately.
    /// The field reference must outlive the synchronous reorder call.
    template <class T>
    CompanionEntry RegisterField(Runtime::Field<T> &field, EntityKind kind, std::string name,
                                 CompanionDisposition disposition = CompanionDisposition::TransferIfReordered)
    {
        CompanionEntry entry{kind, {}, std::move(name)};
        entry.disposition = disposition;
        entry.preflight = [&field, kind](const PermutationTransfer &transfer)
        {
            std::string expected = entityKindName(kind);
            std::transform(expected.begin(), expected.end(), expected.begin(), [](unsigned char c)
                           { return char(std::tolower(c)); });
            DNDS_check_throw_info(field.Values().GetDevice() == Runtime::Device::Host() &&
                                      field.Layout().rows == size_t(transfer.size()) &&
                                      field.GetDistribution().ghosts == 0 &&
                                      field.GetDistribution().owned == field.Layout().rows &&
                                      field.GetDistribution().epoch != UINT64_MAX && field.Values().IsIdle() &&
                                      field.Values().IsValid() && field.Association() == expected,
                                  "native field migration requires idle valid host owned rows with matching entity association");
        };
        entry.attachment = [&field, kind]()
        {
            std::string expected = entityKindName(kind);
            std::transform(expected.begin(), expected.end(), expected.begin(), [](unsigned char c)
                           { return char(std::tolower(c)); });
            DNDS_check_throw_info(field.Association() == expected, "field association differs from attached entity kind");
            DNDS_check_throw_info(field.Values().IsValid() && field.Values().IsIdle(), "native Field attachment requires valid idle storage");
            return CompanionAttachment{Runtime::ElementSchema::TypeIdentity<T>() + ":" + std::to_string(field.Layout().components) + ":" +
                                           std::to_string(field.Layout().stride),
                                       field.Association(), true,
                                       field.GetDistribution().domain, field.GetDistribution().epoch, &field,
                                       field.Values().Identity(), field.Values().Version(), 0, field.Layout().rows,
                                       field.GetDistribution().owned, field.GetDistribution().ghosts};
        };
        entry.prepare = [&field](const PermutationTransfer &transfer, const MPIInfo &mpi)
        {
            // Move source addresses with the mesh's exact correspondence,
            // including local permutation and distributed receive order.
            auto route = NativeRowTransfer(transfer, mpi);
            auto layout = field.Layout();
            layout.rows = route.DestinationRows();
            auto distribution = field.GetDistribution();
            distribution.owned = layout.rows;
            ++distribution.epoch;
            std::unique_ptr<FieldTransferDetail::Prepared<Runtime::Field<T>>> next;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            { next = std::make_unique<FieldTransferDetail::Prepared<Runtime::Field<T>>>(field,
                                                                                                                        Runtime::Field<T>(layout, distribution, field.Association())); });
            // Dense Field includes stride padding in its exact row
            // history. Byte schema retains compatibility for POD fields.
            std::optional<Runtime::Array<uint8_t>> source, destination;
            std::optional<Runtime::ElementSchema> bytes;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                        const auto width = CheckedSize::Multiply(layout.stride, sizeof(T));
                        source.emplace(Runtime::Array<uint8_t>::FromStorage(Runtime::RowLayout::Dense(field.Layout().rows, width), field.Values()));
                        destination.emplace(Runtime::Array<uint8_t>::FromStorage(Runtime::RowLayout::Dense(layout.rows, width), next->next.Values()));
                        bytes.emplace(Runtime::ElementSchema::Arithmetic<uint8_t>()); });
            Runtime::PayloadPlan(route, source->Layout(), destination->Layout(), *bytes)
                .Bind(*source, *destination)
                .Begin()
                .Wait();
            return next;
        };
        auto validate = entry.preflight;
        auto prepare = entry.prepare;
        entry.fn = [validate, prepare, disposition](const PermutationTransfer &transfer, const MPIInfo &mpi)
        {
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                DNDS_check_throw_info(disposition == CompanionDisposition::TransferIfReordered, "field disposition prohibits reorder");
                validate(transfer); });
            auto next = prepare(transfer, mpi);
            next->Publish();
        };
        return entry;
    }
}
