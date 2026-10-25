#include "NativeMeshLegacyArchive.hpp"
#include "FieldTransfer.hpp"
#include "DNDS/Runtime/RowBuilder.hpp"
#include "NativeMeshBuilder.hpp"
#include "NativeMeshArchive.hpp"
#include "DNDS/Serializer/SerializerJSON.hpp"
#include <regex>
#include <cstring>

namespace DNDS::Geom
{
    namespace
    {
        namespace S = Serializer;
        struct Path
        {
            S::SerializerBaseSSP serializer;
            std::string old;
            explicit Path(S::SerializerBaseSSP value) : serializer(std::move(value)), old(serializer->GetCurrentPath()) {}
            ~Path() { serializer->GoToPath(old); }
        };
        void Check(bool value, const char *message) { DNDS_check_throw_info(value, message); }
        template <class T>
        Runtime::Array<T> Decode(const MPIInfo &mpi, S::SerializerBaseSSP serializer, const std::string &base,
                                 const std::string &pair, S::ArrayGlobalOffset partition, std::optional<size_t> width = {})
        {
            const bool perRank = serializer->IsPerRank();
            index rows = 0, start = 0, stride = 0;
            int dynamic = 0;
            bool csr = false, padded = false;
            std::vector<size_t> sizes;
            S::ArrayGlobalOffset rowOffset = S::ArrayGlobalOffset_Unknown, dataOffset = S::ArrayGlobalOffset_Unknown;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                serializer->GoToPath(base);
                serializer->GoToPath(pair);
                serializer->GoToPath("father");
                std::string signature;
                serializer->ReadString("array_sig", signature);
                std::smatch match;
                Check(std::regex_match(signature, match, std::regex("(CSR|TABLE_StaticFixed|TABLE_Fixed|TABLE_StaticMax|TABLE_Max)__([0-9]+)_(-?[0-9]+)_(-?[0-9]+)_(-?[0-9]+)")), "legacy array signature invalid");
                Check(std::stoll(match[2]) == int64_t(sizeof(T)), "legacy array POD ABI differs");
                const auto kind = match[1].str();
                const auto rs = std::stoll(match[3]), rm = std::stoll(match[4]), alignment = std::stoll(match[5]);
                Check(alignment == NoAlign, "aligned legacy primary records are unsupported");
                serializer->ReadInt("row_size_dynamic", dynamic);
                csr = kind == "CSR";
                padded = kind == "TABLE_Max" || kind == "TABLE_StaticMax";
                if (csr) Check(rs == NonUniformSize && rm == NonUniformSize, "legacy CSR signature invalid");
                else
                {
                    stride = padded ? (rm >= 0 ? rm : dynamic) : (rs >= 0 ? rs : dynamic);
                    Check(stride >= 0 && stride <= INT32_MAX, "legacy array stride invalid");
                }
                if (perRank)
                {
                    Check(!(partition == S::ArrayGlobalOffset_EvenSplit), "legacy per-rank archive does not support repartition");
                    serializer->ReadIndex("size", rows);
                }
                else
                {
                    index global = 0;
                    serializer->ReadIndex("sizeGlobal", global);
                    Check(global >= 0, "legacy global row count invalid");
                    if (partition == S::ArrayGlobalOffset_EvenSplit)
                    {
                        const auto range = EvenSplitRange(mpi.rank, mpi.size, global);
                        start = range.first; rows = range.second - range.first;
                    }
                    else
                    {
                        std::vector<index> counts;
                        auto offset = S::ArrayGlobalOffset{mpi.size, 0};
                        serializer->ReadIndexVector("size", counts, offset);
                        Check(counts.size() == size_t(mpi.size), "legacy writer row counts mismatch");
                        index total = 0;
                        for (int rank = 0; rank < mpi.size; ++rank)
                        {
                            Check(counts[size_t(rank)] >= 0 && counts[size_t(rank)] <= INT64_MAX - total, "legacy row counts invalid");
                            if (rank < mpi.rank) start += counts[size_t(rank)];
                            total += counts[size_t(rank)];
                        }
                        Check(total == global, "legacy global row count mismatch");
                        rows = counts[size_t(mpi.rank)];
                    }
                    rowOffset = S::ArrayGlobalOffset{rows, start};
                }
                Check(rows >= 0 && uint64_t(rows) < SIZE_MAX, "legacy row count invalid"); });
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                if (csr)
                {
                    ssp<host_device_vector<index>> offsets;
                    auto selected = perRank ? S::ArrayGlobalOffset{rows, 0} : rowOffset;
                    dataOffset = serializer->ReadSharedRowStartVector("pRowStart", offsets, selected);
                    Check(offsets && offsets->size() == size_t(rows) + 1 && (*offsets)[0] == 0, "legacy CSR offsets invalid");
                    for (size_t row = 0; row < size_t(rows); ++row)
                    {
                        Check((*offsets)[row + 1] >= (*offsets)[row], "legacy CSR offsets decrease");
                        const auto size = (*offsets)[row + 1] - (*offsets)[row];
                        Check(size <= INT32_MAX && (!width || size_t(size) == *width), "legacy CSR row width invalid");
                        sizes.push_back(size_t(size));
                    }
                }
                else
                {
                    Check(!rows || stride <= INT64_MAX / rows, "legacy data count overflow");
                    Check(!start || stride <= INT64_MAX / start, "legacy data offset overflow");
                    if (!perRank) dataOffset = S::ArrayGlobalOffset{rows * stride, start * stride};
                    if (padded)
                    {
                        ssp<host_device_vector<rowsize>> rowSizes;
                        auto selected = rowOffset;
                        serializer->ReadSharedRowsizeVector("pRowSizes", rowSizes, selected);
                        Check(rowSizes && rowSizes->size() == size_t(rows), "legacy padded sizes invalid");
                        for (auto size : *rowSizes)
                        { Check(size >= 0 && size <= stride && (!width || size_t(size) == *width), "legacy padded row invalid"); sizes.push_back(size_t(size)); }
                    }
                    else
                    {
                        Check(!width || size_t(stride) == *width, "legacy fixed width mismatch");
                        sizes.assign(size_t(rows), size_t(stride));
                    }
                } });
            std::vector<T> values;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                auto offset = dataOffset;
                if constexpr (std::is_same_v<T, index>) serializer->ReadIndexVector("data", values, offset);
                else if constexpr (std::is_same_v<T, real>)
                {
                    if (!std::dynamic_pointer_cast<S::SerializerJSON>(serializer)) serializer->ReadRealVector("data", values, offset);
                    else
                    {
                        if (offset.isDist()) offset = offset * index(sizeof(T));
                        index count = 0;
                        serializer->ReadUint8Array("data", nullptr, count, offset);
                        Check(count >= 0 && count % index(sizeof(T)) == 0, "legacy real bytes truncated");
                        values.resize(size_t(count / index(sizeof(T))));
                        uint8_t dummy = 0;
                        serializer->ReadUint8Array("data", values.empty() ? &dummy : reinterpret_cast<uint8_t *>(values.data()), count, offset);
                    }
                }
                else
                {
                    if (offset.isDist())
                    {
                        Check(offset.size() <= INT64_MAX / index(sizeof(T)) && offset.offset() <= INT64_MAX / index(sizeof(T)), "legacy POD count overflow");
                        offset = offset * index(sizeof(T));
                    }
                    index count = 0;
                    serializer->ReadUint8Array("data", nullptr, count, offset);
                    Check(count >= 0 && count % index(sizeof(T)) == 0, "legacy POD bytes truncated");
                    values.resize(size_t(count / index(sizeof(T))));
                    uint8_t dummy = 0;
                    serializer->ReadUint8Array("data", values.empty() ? &dummy : reinterpret_cast<uint8_t *>(values.data()), count, offset);
                }
                size_t required = 0;
                if (csr) for (auto size : sizes) { Check(size <= SIZE_MAX - required, "legacy CSR count overflow"); required += size; }
                else required = size_t(rows) * size_t(stride);
                Check(values.size() == required, "legacy data extent mismatch"); });
            std::optional<Runtime::Array<T>> output;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
            Runtime::RowBuilder builder(size_t(rows), sizeof(T));
            for (size_t row = 0; row < size_t(rows); ++row) builder.ResizeRow(row, sizes[row]);
            size_t offset = 0;
            for (size_t row = 0; row < size_t(rows); ++row)
            {
                auto destination = builder.Row<T>(row);
                std::copy_n(values.data() + offset, sizes[row], destination.data);
                offset += csr ? sizes[row] : size_t(stride);
            }
            output.emplace(Runtime::Array<T>::FromSharedStorage(builder.Commit())); });
            return std::move(*output);
        }
    }

    std::shared_ptr<const NativeMeshState> ReadCompatibleNativeMeshArchive(
        const MPIInfo &mpi, S::SerializerBaseSSP serializer, const std::string &name,
        S::ArrayGlobalOffset partition, int ghostLayers)
    {
        int schema = 0;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            Check(bool(serializer), "compatible archive serializer missing");
            Path restore(serializer);
            serializer->GoToPath(name);
            const auto keys = serializer->ListCurrentPath();
            const bool native = keys.count("metadata_json"), legacy = keys.count("mesh");
            Check(native != legacy, "mesh archive schema missing or ambiguous");
            schema = native ? 1 : 2; });
        int expected = schema;
        MPI_Bcast(&expected, 1, MPI_INT, 0, mpi.comm);
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { Check(schema == expected, "mesh archive schemas differ across ranks"); });
        return schema == 1 ? ReadNativeMeshArchive(mpi, std::move(serializer), name, partition, ghostLayers)
                           : NativeMeshLegacyArchive::Read(mpi, std::move(serializer), name, partition, ghostLayers);
    }

    NativeOwnedMeshData NativeMeshLegacyArchive::ReadOwned(const MPIInfo &mpi, S::SerializerBaseSSP serializer,
                                                           const std::string &name, S::ArrayGlobalOffset partition)
    {
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            Check(bool(serializer) && name.size() <= size_t(INT_MAX), "legacy archive serializer/name invalid");
            Check(partition == S::ArrayGlobalOffset_Unknown || partition == S::ArrayGlobalOffset_EvenSplit, "unsupported legacy archive partition");
            if (!serializer->IsPerRank()) Check(serializer->getMPI() == mpi, "legacy serializer communicator mismatch"); });
        std::string expectedName = name;
        int expectedSize = mpi.rank == 0 ? int(name.size()) : 0;
        MPI_Bcast(&expectedSize, 1, MPI_INT, 0, mpi.comm);
        expectedName.resize(size_t(expectedSize));
        MPI_Bcast(expectedName.data(), expectedSize, MPI_CHAR, 0, mpi.comm);
        int signature = int(serializer->IsPerRank()) + 2 * int(partition == S::ArrayGlobalOffset_EvenSplit), expectedSignature = signature;
        MPI_Bcast(&expectedSignature, 1, MPI_INT, 0, mpi.comm);
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { Check(expectedName == name && signature == expectedSignature, "legacy read schedule differs across ranks"); });
        Path restore(serializer);
        NativeOwnedMeshData data;
        std::string base;
        index writers = 0;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            serializer->GoToPath(name);
            base = serializer->GetCurrentPath();
            std::string schema;
            index dimension = 0, rank = mpi.rank;
            int periodic = 0;
            serializer->ReadString("mesh", schema);
            serializer->ReadIndex("dim", dimension);
            serializer->ReadIndex("MPISize", writers);
            serializer->ReadInt("isPeriodic", periodic);
            if (serializer->IsPerRank()) serializer->ReadIndex("MPIRank", rank);
            Check(schema == "UnstructuredMesh" && dimension >= 1 && dimension <= 3 && writers > 0 &&
                  (periodic == 0 || periodic == 1), "legacy mesh metadata invalid");
            Check((partition == S::ArrayGlobalOffset_EvenSplit && !serializer->IsPerRank()) || writers == mpi.size,
                  "legacy matching partition requires writer rank count");
            Check(!serializer->IsPerRank() || rank == mpi.rank, "legacy per-rank archive rank mismatch");
            data.dimension = int(dimension); data.periodic = periodic; });
        std::array<index, 3> metadata{data.dimension, data.periodic, writers}, expected = metadata;
        MPI_Bcast(expected.data(), int(expected.size()), DNDS_MPI_INDEX, 0, mpi.comm);
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { Check(metadata == expected, "legacy metadata differs across ranks"); });
        data.coords = Decode<real>(mpi, serializer, base, "coords", partition, 3);
        data.cellInfo = Decode<ElemInfo>(mpi, serializer, base, "cellElemInfo", partition, 1);
        data.bndInfo = Decode<ElemInfo>(mpi, serializer, base, "bndElemInfo", partition, 1);
        data.relations[size_t(MeshRelation::CellNodes)] = Decode<index>(mpi, serializer, base, "cell2node", partition);
        data.relations[size_t(MeshRelation::BoundaryNodes)] = Decode<index>(mpi, serializer, base, "bnd2node", partition);
        data.ids.emplace(EntityKind::Node, Decode<index>(mpi, serializer, base, "node2nodeOrig", partition, 1));
        data.ids.emplace(EntityKind::Cell, Decode<index>(mpi, serializer, base, "cell2cellOrig", partition, 1));
        data.ids.emplace(EntityKind::Bnd, Decode<index>(mpi, serializer, base, "bnd2bndOrig", partition, 1));
        if (data.periodic)
        {
            data.cellNodesPbi = Decode<NodePeriodicBits>(mpi, serializer, base, "cell2nodePbi", partition);
            data.bndNodesPbi = Decode<NodePeriodicBits>(mpi, serializer, base, "bnd2nodePbi", partition);
            serializer->GoToPath(base);
            serializer->GoToPath("periodicInfo");
            for (int frame = 1; frame <= 3; ++frame)
            {
                std::vector<real> rotation, center, translation;
                FieldTransferDetail::LocalStage(mpi, [&]
                                                {
                    auto offset = S::ArrayGlobalOffset_One;
                    serializer->ReadRealVector("rotation" + std::to_string(frame), rotation, offset);
                    offset = S::ArrayGlobalOffset_One;
                    serializer->ReadRealVector("rotationCenter" + std::to_string(frame), center, offset);
                    offset = S::ArrayGlobalOffset_One;
                    serializer->ReadRealVector("translation" + std::to_string(frame), translation, offset);
                    Check(rotation.size() == 9 && center.size() == 3 && translation.size() == 3, "legacy periodic frame width invalid");
                    std::copy(rotation.begin(), rotation.end(), data.periodicInfo.rotation[size_t(frame)].map().data());
                    std::copy(center.begin(), center.end(), data.periodicInfo.rotationCenter[size_t(frame)].map().data());
                    std::copy(translation.begin(), translation.end(), data.periodicInfo.translation[size_t(frame)].map().data());
                    Check(data.periodicInfo.rotation[size_t(frame)].map().allFinite() && data.periodicInfo.rotationCenter[size_t(frame)].map().allFinite() &&
                          data.periodicInfo.translation[size_t(frame)].map().allFinite(), "legacy periodic frame nonfinite"); });
            }
        }
        const auto nodes = EntitySpace::CreateOwned(EntityKind::Node, mpi, index(data.coords->Rows()));
        data.spaces.emplace(EntityKind::Node, nodes);
        data.spaces.emplace(EntityKind::Cell, nodes.CreateSibling(EntityKind::Cell, index(data.cellInfo->Rows())));
        data.spaces.emplace(EntityKind::Bnd, nodes.CreateSibling(EntityKind::Bnd, index(data.bndInfo->Rows())));
        NativeMeshInput input{mpi, data.dimension, data.spaces.at(EntityKind::Cell), nodes, data.spaces.at(EntityKind::Bnd),
                              *data.relations[size_t(MeshRelation::CellNodes)], *data.relations[size_t(MeshRelation::BoundaryNodes)],
                              *data.coords, *data.cellInfo, *data.bndInfo, data.ids};
        input.periodic = data.periodic;
        input.cellNodesPbi = data.cellNodesPbi;
        input.boundaryNodesPbi = data.bndNodesPbi;
        input.periodicInfo = data.periodicInfo;
        auto prepared = NativeMeshBuilder::Build(input, {1, true});
        return NativeMeshAssembler::CaptureOwned(*prepared, mpi);
    }

    std::shared_ptr<const NativeMeshState> NativeMeshLegacyArchive::Read(const MPIInfo &mpi, S::SerializerBaseSSP serializer,
                                                                         const std::string &name, S::ArrayGlobalOffset partition,
                                                                         int ghostLayers)
    {
        return NativeMeshAssembler::Build(mpi, ReadOwned(mpi, std::move(serializer), name, partition), ghostLayers);
    }
}
