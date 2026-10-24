#pragma once

#include "ReorderPlan.hpp"
#include "DNDS/Runtime/Field.hpp"
#include "DNDS/Runtime/Exchange.hpp"
#include "DNDS/Runtime/RowTransfer.hpp"
#include <algorithm>
#include <cctype>
#include <optional>

namespace DNDS::Geom
{
    namespace FieldTransferDetail
    {
        /// Every rank runs local work before agreement and enters the next
        /// collective only after all ranks succeeded. Payload/route constructors
        /// retain their own collective schema and extent checks.
        template <class F>
        void LocalStage(const MPIInfo &mpi, F &&action)
        {
            int valid = 1;
            try
            {
                action();
            }
            catch (...)
            {
                valid = 0;
            }
            MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, mpi.comm);
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
        ArrayAdjacencyPair<2> owners;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            owners.InitPair("NativeRowTransfer.owners", mpi);
            owners.father->Resize(transfer.size());
            for (DNDS::index i = 0; i < transfer.size(); ++i)
            {
                owners(i, 0) = mpi.rank;
                owners(i, 1) = i;
            } });
        transfer.transferRows(owners, mpi);
        std::vector<Runtime::RowRequest> requests;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            for (DNDS::index i = 0; i < owners.father->Size(); ++i)
                requests.push_back({int(owners(i, 0)), owners(i, 1), size_t(i)}); });
        return {mpi.comm, size_t(transfer.size()), requests.size(), requests};
    }

    template <class T, class L>
    CompanionEntry RegisterArray(Runtime::Array<T, L> &array, EntityKind kind, std::string name,
                                 Runtime::ElementSchema schema = Runtime::ElementSchema::Arithmetic<T>(),
                                 Runtime::ExecutionContext context = Runtime::ExecutionContext::Host(),
                                 CompanionDisposition disposition = CompanionDisposition::TransferIfReordered)
    {
        CompanionEntry entry{kind, {}, std::move(name)};
        entry.disposition = disposition;
        entry.preflight = [&array, schema, context](const PermutationTransfer &transfer)
        {
            DNDS_check_throw_info(array.Rows() == size_t(transfer.size()) && array.Storage().IsIdle() && array.IsValid() &&
                                      schema.Extent() == sizeof(T) && schema.Compatible(Runtime::ElementSchema::TypeIdentity<T>()) &&
                                      context.device == array.GetDevice() &&
                                      (context.device != Runtime::Device::Host() || !context.stream),
                                  "native Array migration requires idle valid compatible owned rows/context");
        };
        entry.attachment = [&array, schema]()
        { return CompanionAttachment{schema.Identity() + ":" + Runtime::ElementSchema::TypeIdentity<T>() + ":" +
                                         std::to_string(static_cast<int>(array.GetDevice().space)),
                                     {},
                                     false,
                                     0,
                                     0,
                                     &array}; };
        entry.prepare = [&array, schema, context](const PermutationTransfer &transfer, const MPIInfo &mpi)
        {
            auto route = NativeRowTransfer(transfer, mpi);
            std::optional<Runtime::RowLayout> layout;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                        layout.emplace(route.DestinationLayout(array.Layout()));
                        if constexpr (!std::is_same_v<L, Runtime::RowLayout>)
                            layout = Runtime::RowLayout::Dense(layout->Rows(), L::Default().Width()); });
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
            return CompanionAttachment{Runtime::ElementSchema::TypeIdentity<T>() + ":" + std::to_string(field.Layout().components) + ":" +
                                           std::to_string(field.Layout().stride),
                                       field.Association(), true,
                                       field.GetDistribution().domain, field.GetDistribution().epoch, &field};
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
