#pragma once

#include "Array.hpp"
#include "DNDS/ArrayTransformer.hpp"
#include <cstring>
#include <optional>

namespace DNDS::Runtime
{
    template <class T>
    std::string ArraySchema()
    {
        if constexpr (std::is_same_v<T, bool>)
            return "bool";
        if constexpr (std::is_floating_point_v<T>)
            return "float" + std::to_string(sizeof(T) * 8);
        if constexpr (std::is_integral_v<T>)
            return std::string(std::is_signed_v<T> ? "int" : "uint") + std::to_string(sizeof(T) * 8);
        return typeid(T).name(); // Supply a stable explicit schema for application PODs.
    }
    namespace Detail
    {
        inline void IOCheck(bool valid, const MPIInfo &mpi, bool perRank, const char *message)
        {
            int all = valid;
            if (!perRank)
                MPI_Allreduce(MPI_IN_PLACE, &all, 1, MPI_INT, MPI_MIN, mpi.comm);
            DNDS_check_throw_info(all, message);
        }
        inline void IOAgree(const std::string &value, const MPIInfo &mpi, bool perRank)
        {
            if (perRank)
                return;
            IOCheck(value.size() <= INT32_MAX, mpi, false, "checkpoint metadata too long");
            int size = mpi.rank == 0 ? int(value.size()) : 0;
            MPI_Bcast(&size, 1, MPI_INT, 0, mpi.comm);
            std::string root(size, '\0');
            if (mpi.rank == 0)
                root = value;
            MPI_Bcast(root.data(), size, MPI_CHAR, 0, mpi.comm);
            IOCheck(value == root, mpi, false, "checkpoint name/schema differs across ranks");
        }
        struct SerializerPath
        {
            Serializer::SerializerBaseSSP serializer;
            std::string path;
            explicit SerializerPath(Serializer::SerializerBaseSSP value) : serializer(std::move(value)), path(serializer->GetCurrentPath()) {}
            ~SerializerPath() { serializer->GoToPath(path); }
        };
    }

    /// Checkpoint logical rows through the existing distributed CSR serializer.
    /// I/O explicitly stages CUDA data on host; padding is not serialized.
    /// Homogeneous ABI bytes plus schema metadata; no implied portable POD codec.
    /// This common codec serves C++, Python structured dtypes and device snapshots.
    /// @pre Finish asynchronous producers before taking the snapshot. Use the
    /// same serializer mode, name/schema and collective call order on all ranks
    /// (including empty ranks); per-rank serializers are independent instead.
    /// The application supplies a stable POD schema/ABI and checkpoints matching
    /// mesh IDs and field epochs. Separate field writes are not a transaction.
    inline void WriteArrayStorage(const Buffer &storage, const RowLayout &layout, size_t elementBytes,
                                  Serializer::SerializerBaseSSP serializer, const std::string &name,
                                  const MPIInfo &mpi, const std::string &schema)
    {
        DNDS_check_throw_info(serializer, "null Array serializer");
        Detail::IOAgree(name + ":" + schema + ":" + std::to_string(elementBytes), mpi, serializer->IsPerRank());
        bool valid = elementBytes && layout.Rows() <= size_t(INT64_MAX) && !schema.empty();
        valid = valid && storage.Bytes() / elementBytes == layout.Elements() && storage.Bytes() % elementBytes == 0;
        for (size_t i = 0; i < layout.Rows(); ++i)
            valid = valid && layout.RowSize(i) <= size_t(INT32_MAX) / elementBytes;
        Detail::IOCheck(valid, mpi, serializer->IsPerRank(), "Array checkpoint row/count exceeds serializer range");

        // Retain one stable snapshot through the collective write. A conflicting
        // rank-local access must be rejected on all participants before I/O.
        Buffer host;
        std::optional<ReadAccess> read;
        try
        {
            if (storage.GetDevice() != Device::Host())
            {
                host = Buffer::Allocate(storage.Bytes());
                Copy(storage, host, {storage.GetDevice(), nullptr}).Wait();
            }
            read.emplace(host.Identity() ? host.Read() : storage.Read());
        }
        catch (...)
        {
            valid = false;
        }
        Detail::IOCheck(valid, mpi, serializer->IsPerRank(), "Array checkpoint cannot acquire a host snapshot");
        DNDS::ParArray<uint8_t, NonUniformSize> staging(mpi);
        staging.Resize(static_cast<DNDS::index>(layout.Rows()));
        auto view = read->View<uint8_t>();
        for (size_t i = 0; i < layout.Rows(); ++i)
        {
            const size_t width = layout.RowSize(i);
            staging.ResizeRow(i, static_cast<DNDS::rowsize>(width * elementBytes));
            if (width)
                std::memcpy(staging[i], view.data + layout.Offset(i) * elementBytes, width * elementBytes);
        }
        staging.Compress();
        Detail::SerializerPath guard(serializer);
        serializer->CreatePath(name);
        serializer->GoToPath(name);
        serializer->WriteInt("runtime_array_format", 2);
        serializer->WriteIndex("element_bytes", elementBytes);
        serializer->WriteInt("writer_ranks", mpi.size);
        serializer->WriteString("element_schema", schema);
        staging.WriteSerializer(serializer, "rows", Serializer::ArrayGlobalOffset_Parts);
    }

    template <class T, class L>
    void WriteArray(const Array<T, L> &array, Serializer::SerializerBaseSSP serializer,
                    const std::string &name, const MPIInfo &mpi, const std::string &schema = ArraySchema<T>())
    {
        WriteArrayStorage(array.Storage(), array.Layout(), sizeof(T), serializer, name, mpi, schema);
    }

    struct ArrayArchive
    {
        RowLayout layout;
        Buffer storage;
    };

    /// Unknown reads the saved rank partition; EvenSplit or {rows,start} reads
    /// directly from the global checkpoint (including a different MPI size).
    /// The result uses packed ragged rows; a fixed policy restores fixed rows.
    inline ArrayArchive ReadArrayStorage(Serializer::SerializerBaseSSP serializer, const std::string &name,
                                         const MPIInfo &mpi, Serializer::ArrayGlobalOffset offset,
                                         Device device, size_t elementBytes, const std::string &schema)
    {
        DNDS_check_throw_info(serializer, "null Array serializer");
        Detail::IOAgree(name + ":" + schema + ":" + std::to_string(elementBytes), mpi, serializer->IsPerRank());
        Detail::SerializerPath guard(serializer);
        serializer->GoToPath(name);
        int format = 0, ranks = 0;
        DNDS::index bytes = 0;
        std::string storedSchema;
        serializer->ReadInt("runtime_array_format", format);
        serializer->ReadIndex("element_bytes", bytes);
        serializer->ReadInt("writer_ranks", ranks);
        serializer->ReadString("element_schema", storedSchema);
        bool valid = format == 2 && elementBytes && bytes == elementBytes && storedSchema == schema;
        valid = valid && (offset.isDist() ? offset.size() >= 0 : offset == Serializer::ArrayGlobalOffset_Unknown || offset == Serializer::ArrayGlobalOffset_EvenSplit);
        valid = valid && (!(offset == Serializer::ArrayGlobalOffset_Unknown) || ranks == mpi.size);
        valid = valid && (!serializer->IsPerRank() || offset == Serializer::ArrayGlobalOffset_Unknown);
        Detail::IOCheck(valid, mpi, serializer->IsPerRank(), "Array checkpoint schema or read partition mismatch");
        DNDS::ParArray<uint8_t, NonUniformSize> staging(mpi);
        staging.ReadSerializer(serializer, "rows", offset);
        std::vector<size_t> lengths(size_t(staging.Size()));
        for (size_t i = 0; i < lengths.size(); ++i)
        {
            valid = valid && staging.RowSize(i) % elementBytes == 0;
            lengths[i] = staging.RowSize(i) / elementBytes;
        }
        auto layout = RowLayout::Ragged(lengths);
        Detail::IOCheck(valid, mpi, serializer->IsPerRank(), "Array checkpoint row shape mismatch");
        auto host = Buffer::Allocate(CheckedSize::Multiply(layout.Elements(), elementBytes));
        {
            auto write = host.Write();
            auto view = write.View<uint8_t>();
            for (size_t i = 0; i < lengths.size(); ++i)
                if (lengths[i])
                    std::memcpy(view.data + layout.Offset(i) * elementBytes, staging[i], lengths[i] * elementBytes);
        }
        if (device == Device::Host())
            return {std::move(layout), std::move(host)};
        auto result = Buffer::Allocate(host.Bytes(), device);
        Copy(host, result, {device, nullptr}).Wait();
        return {std::move(layout), std::move(result)};
    }

    template <class T, class L = RowLayout>
    Array<T, L> ReadArray(Serializer::SerializerBaseSSP serializer, const std::string &name,
                          const MPIInfo &mpi, Serializer::ArrayGlobalOffset offset = Serializer::ArrayGlobalOffset_Unknown,
                          Device device = Device::Host(), const std::string &schema = ArraySchema<T>())
    {
        auto archive = ReadArrayStorage(serializer, name, mpi, offset, device, sizeof(T), schema);
        if constexpr (!std::is_same_v<L, RowLayout>)
        {
            const auto width = L::Default().Width();
            bool valid = true;
            for (size_t i = 0; i < archive.layout.Rows(); ++i)
                valid = valid && archive.layout.RowSize(i) == width;
            Detail::IOCheck(valid, mpi, serializer->IsPerRank(), "fixed Array checkpoint width mismatch");
            archive.layout = RowLayout::Dense(archive.layout.Rows(), width);
        }
        return Array<T, L>::FromStorage(std::move(archive.layout), std::move(archive.storage));
    }
}
