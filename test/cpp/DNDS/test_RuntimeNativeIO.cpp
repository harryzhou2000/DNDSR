#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "DNDS/Runtime/ArrayIO.hpp"
#include "DNDS/Runtime/RowBuilder.hpp"
#include "DNDS/ArrayTransformer.hpp"
#include "DNDS/Serializer/SerializerH5.hpp"
#include "DNDS/Serializer/SerializerJSON.hpp"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <unistd.h>

namespace R = DNDS::Runtime;
namespace S = DNDS::Serializer;
using I = DNDS::index;

namespace
{
    template <class Backend>
    struct DirectSpy : Backend
    {
        using Backend::Backend;
        std::vector<const uint8_t *> written;
        std::vector<uint8_t *> restored;
        size_t legacyWrites = 0;
        void WriteByteSegments(const std::string &name, const uint8_t *data, I bytes,
                               const std::vector<S::ByteMemorySegment> &segments, S::ArrayGlobalOffset offset) override
        {
            written.push_back(data);
            Backend::WriteByteSegments(name, data, bytes, segments, offset);
        }
        void ReadByteSegments(const std::string &name, uint8_t *data, I bytes,
                              const std::vector<S::ByteMemorySegment> &segments, S::ArrayGlobalOffset &offset) override
        {
            restored.push_back(data);
            Backend::ReadByteSegments(name, data, bytes, segments, offset);
        }
        void WriteUint8Array(const std::string &name, const uint8_t *data, I bytes, S::ArrayGlobalOffset offset) override
        {
            ++legacyWrites;
            Backend::WriteUint8Array(name, data, bytes, offset);
        }
    };

    std::string File(const DNDS::MPIInfo &mpi, const std::string &tag, bool json)
    {
        int pid = int(getpid());
        MPI_Bcast(&pid, 1, MPI_INT, 0, mpi.comm);
        return (std::filesystem::temp_directory_path() /
                ("dnds-native-io-" + std::to_string(pid) + "-" + tag +
                 (json ? "-rank" + std::to_string(mpi.rank) + ".json" : ".h5")))
            .string();
    }

    std::vector<I> Counts(const DNDS::MPIInfo &mpi)
    {
        std::vector<I> result(size_t(mpi.size));
        for (int rank = 0; rank < mpi.size; ++rank)
            result[size_t(rank)] = rank == mpi.size - 1 && mpi.size > 1 ? 0 : 3 + rank;
        return result;
    }

    I Prefix(const std::vector<I> &counts, int rank)
    {
        I value = 0;
        for (int i = 0; i < rank; ++i)
            value += counts[size_t(i)];
        return value;
    }

    template <class T, class Backend>
    void WriteDirect(const R::Array<T> &source, const std::shared_ptr<DirectSpy<Backend>> &serializer,
                     const std::string &name, const DNDS::MPIInfo &mpi)
    {
        auto retained = source.Storage().Read();
        const auto pointer = retained.template View<uint8_t>().data;
        const auto before = serializer->written.size();
        R::WriteArray(source, serializer, name, mpi);
        REQUIRE(serializer->written.size() == before + 1);
        CHECK(serializer->written.back() == pointer);
        CHECK(serializer->legacyWrites == 0);
    }

    template <class T, class L = R::RowLayout, class Backend>
    R::Array<T> ReadDirect(const std::shared_ptr<DirectSpy<Backend>> &serializer, const std::string &name,
                           const DNDS::MPIInfo &mpi, S::ArrayGlobalOffset partition = S::ArrayGlobalOffset_Unknown)
    {
        const auto before = serializer->restored.size();
        auto result = R::ReadArray<T, L>(serializer, name, mpi, partition);
        REQUIRE(serializer->restored.size() == before + 1);
        auto retained = result.Storage().Read();
        CHECK(serializer->restored.back() == retained.template View<uint8_t>().data);
        CHECK(serializer->legacyWrites == 0);
        return result;
    }

    void Fill(R::Array<double> &array, I first)
    {
        auto values = array.Write();
        for (size_t row = 0; row < array.RowCount(); ++row)
            for (size_t slot = 0; slot < array.RowSize(row); ++slot)
                values.At(row, slot) = double((first + I(row)) * 100 + I(slot)) + 0.25;
    }

    void CheckValues(const R::Array<double> &array, I first, bool variable)
    {
        auto values = array.Read();
        for (size_t row = 0; row < array.RowCount(); ++row)
        {
            CHECK(array.RowSize(row) == (variable ? size_t((first + I(row)) % 4) : 3));
            for (size_t slot = 0; slot < array.RowSize(row); ++slot)
                CHECK(values.At(row, slot) == double((first + I(row)) * 100 + I(slot)) + 0.25);
        }
    }

    template <class Backend>
    void Shapes(const DNDS::MPIInfo &mpi, bool json, const std::shared_ptr<DirectSpy<Backend>> &serializer)
    {
        const auto counts = Counts(mpi);
        const I first = Prefix(counts, mpi.rank), total = Prefix(counts, mpi.size);
        const size_t rows = size_t(counts[size_t(mpi.rank)]);
        std::vector<size_t> lengths(rows);
        for (size_t row = 0; row < rows; ++row)
            lengths[row] = size_t((first + I(row)) % 4);
        R::Array<double> dense(R::RowLayout::Dense(rows, 3, 7));
        R::Array<double, R::CSRRows> csr(R::RowLayout::Ragged(lengths));
        R::Array<double, R::PaddedRows> padded(R::RowLayout::Padded(lengths, 9));
        R::Array<double, R::FixedRows<3, 5>> fixed(R::RowLayout::Dense(rows, 3, 5));
        R::Array<double, R::StaticMaxRows<8>> maximum(R::RowLayout::Padded(lengths, 8));
        {
            auto raw = dense.Storage().Write();
            const auto bytes = raw.View<uint8_t>();
            if (bytes.size)
                std::memset(bytes.data, 0xd7, bytes.size);
        }
        Fill(dense, first);
        Fill(csr, first);
        Fill(padded, first);
        Fill(fixed, first);
        Fill(maximum, first);
        const auto file = File(mpi, "shapes", json);
        serializer->OpenFile(file, false);
        WriteDirect(dense, serializer, "dense", mpi);
        WriteDirect(csr, serializer, "csr", mpi);
        WriteDirect(padded, serializer, "padded", mpi);
        WriteDirect(fixed, serializer, "fixed", mpi);
        WriteDirect(maximum, serializer, "maximum", mpi);
        R::Array<double> empty(R::RowLayout::Dense(0, 3, 5));
        WriteDirect(empty, serializer, "empty", mpi);
#ifdef DNDS_USE_CUDA
        auto device = dense.To(R::Device::CUDA(0));
        const auto payloadCalls = serializer->written.size();
        CHECK_THROWS(R::WriteArray(device, serializer, "device_rejected", mpi));
        CHECK(serializer->written.size() == payloadCalls);
        CHECK(serializer->ListCurrentPath().count("device_rejected") == 0);
#endif
        serializer->CloseFile();
        {
            auto raw = dense.Storage().Read();
            const auto bytes = raw.View<uint8_t>();
            for (size_t row = 0; row < rows; ++row)
                for (size_t byte = 3 * sizeof(double); byte < 7 * sizeof(double); ++byte)
                    CHECK(bytes.data[row * 7 * sizeof(double) + byte] == 0xd7);
        }
        serializer->OpenFile(file, true);
        serializer->GoToPath("dense");
        CHECK(serializer->ListCurrentPath().count("row_offsets") == 0);
        I width = -1;
        serializer->ReadIndex("row_width", width);
        CHECK(width == 3);
        serializer->GoToPath("/");
        auto a = ReadDirect<double, R::FixedRows<3, 5>>(serializer, "dense", mpi);
        CHECK(a.Layout().Stride() == 5);
        CheckValues(a, first, false);
        {
            auto raw = a.Storage().Read();
            const auto bytes = raw.template View<uint8_t>();
            for (size_t row = 0; row < rows; ++row)
                for (size_t byte = 3 * sizeof(double); byte < 5 * sizeof(double); ++byte)
                    CHECK(bytes.data[row * 5 * sizeof(double) + byte] == 0);
        }
        CheckValues(ReadDirect<double>(serializer, "fixed", mpi), first, false);
        CheckValues(ReadDirect<double, R::CSRRows>(serializer, "csr", mpi), first, true);
        CheckValues(ReadDirect<double, R::PaddedRows>(serializer, "padded", mpi), first, true);
        auto maxRead = ReadDirect<double, R::StaticMaxRows<8>>(serializer, "maximum", mpi);
        CHECK(maxRead.Layout().Stride() == 8);
        CheckValues(maxRead, first, true);
        const auto before = serializer->restored.size();
        CHECK_THROWS((R::ReadArray<double, R::FixedRows<2>>(serializer, "dense", mpi)));
        CHECK(serializer->restored.size() == before);
        auto emptyRead = ReadDirect<double, R::FixedRows<3, 5>>(serializer, "empty", mpi);
        CHECK(emptyRead.RowCount() == 0);
        CHECK(emptyRead.Layout().Width() == 3);
        CHECK_THROWS((R::ReadArray<double, R::FixedRows<2>>(serializer, "empty", mpi)));
        CHECK_THROWS(R::ReadArray<double>(serializer, "dense", mpi, S::ArrayGlobalOffset_Unknown, R::Device::CUDA(0)));
        serializer->CloseFile();
        if (!json)
        {
            serializer->OpenFile(file, true);
            const auto calls = serializer->restored.size();
            // A bad row window or static policy on just one participant must
            // reject all ranks before any collective payload read is posted.
            const S::ArrayGlobalOffset window{1, mpi.rank == 0 ? total : 0};
            CHECK_THROWS(R::ReadArray<double>(serializer, "dense", mpi, window));
            CHECK(serializer->restored.size() == calls);
            if (mpi.rank == 0)
                CHECK_THROWS((R::ReadArray<double, R::FixedRows<2>>(serializer, "dense", mpi)));
            else
                CHECK_THROWS((R::ReadArray<double, R::FixedRows<3>>(serializer, "dense", mpi)));
            CHECK(serializer->restored.size() == calls);
            CheckValues(ReadDirect<double>(serializer, "dense", mpi, {1, total - 1}), total - 1, false);
            CheckValues(ReadDirect<double>(serializer, "csr", mpi, {1, total - 1}), total - 1, true);
            const auto interval = DNDS::EvenSplitRange(mpi.rank, mpi.size, total);
            auto even = ReadDirect<double>(serializer, "maximum", mpi, S::ArrayGlobalOffset_EvenSplit);
            CHECK(even.RowCount() == size_t(interval.second - interval.first));
            CheckValues(even, interval.first, true);
            serializer->CloseFile();
            if (mpi.rank == 0)
            {
                const DNDS::MPIInfo single(MPI_COMM_SELF);
                auto reader = std::make_shared<DirectSpy<S::SerializerH5>>(single);
                reader->OpenFile(file, true);
                auto gathered = ReadDirect<double>(reader, "dense", single, S::ArrayGlobalOffset_EvenSplit);
                CHECK(gathered.RowCount() == size_t(total));
                CheckValues(gathered, 0, false);
                if (mpi.size > 1)
                    CHECK_THROWS(R::ReadArray<double>(reader, "dense", single));
                reader->CloseFile();
            }
        }
        MPI_Barrier(mpi.comm);
        if (json || mpi.rank == 0)
            std::filesystem::remove(file);
    }

    struct Mixed
    {
        uint16_t tag;
        int32_t zone;
        double energy;
        int64_t ids[2];
    };

    void RegisterMixed()
    {
        R::ElementSchema::Register<Mixed>(R::ElementSchema::Record("native-io.mixed.v1", sizeof(Mixed),
                                                                   {{"tag", offsetof(Mixed, tag), R::ElementScalar::U16},
                                                                    {"zone", offsetof(Mixed, zone), R::ElementScalar::I32},
                                                                    {"energy", offsetof(Mixed, energy), R::ElementScalar::F64},
                                                                    {"ids", offsetof(Mixed, ids), R::ElementScalar::I64, 2}},
                                                                   "NativeIOMixed", alignof(Mixed)));
    }

    void FillMixed(Mixed &value, I global, size_t slot)
    {
        std::memset(&value, 0xa5, sizeof(value));
        value.tag = uint16_t(0x1234 + global + I(slot));
        value.zone = int32_t(-1234567 - 17 * global - I(slot));
        value.energy = 0.125 + double(global * 8 + I(slot));
        value.ids[0] = INT64_C(0x123456789abcdef) + global;
        value.ids[1] = -INT64_C(0x11223344556677) - I(slot);
    }

    void ReverseMixed(Mixed &value)
    {
        // Independent fixture encoding: no production SwapWords invocation.
        auto reverse = [](void *pointer, size_t bytes)
        {
            auto begin = static_cast<uint8_t *>(pointer);
            std::reverse(begin, begin + bytes);
        };
        reverse(&value.tag, 2);
        reverse(&value.zone, 4);
        reverse(&value.energy, 8);
        reverse(&value.ids[0], 8);
        reverse(&value.ids[1], 8);
    }

    void SetH5Order(const std::string &file, int order)
    {
        const hid_t handle = H5Fopen(file.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
        REQUIRE(handle >= 0);
        const hid_t attribute = H5Aopen(handle, "runtime_array_byte_order", H5P_DEFAULT);
        REQUIRE(attribute >= 0);
        CHECK(H5Awrite(attribute, H5T_NATIVE_INT, &order) >= 0);
        CHECK(H5Aclose(attribute) >= 0);
        CHECK(H5Fclose(handle) >= 0);
    }

    template <class Backend>
    void MixedEndian(const DNDS::MPIInfo &mpi, bool json, const std::shared_ptr<DirectSpy<Backend>> &serializer)
    {
        RegisterMixed();
        const auto counts = Counts(mpi);
        const I first = Prefix(counts, mpi.rank);
        const size_t rows = size_t(counts[size_t(mpi.rank)]);
        std::vector<size_t> lengths(rows);
        for (size_t row = 0; row < rows; ++row)
            lengths[row] = row % 3 == 1 ? 0 : 2 + row % 2;
        R::Array<Mixed> dense(R::RowLayout::Dense(rows, 2, 4)), csr(R::RowLayout::Ragged(lengths));
        for (auto *array : {&dense, &csr})
        {
            auto write = array->Write();
            for (size_t row = 0; row < rows; ++row)
                for (size_t slot = 0; slot < array->RowSize(row); ++slot)
                {
                    FillMixed(write.At(row, slot), first + I(row), slot);
                    ReverseMixed(write.At(row, slot));
                }
        }
        const auto file = File(mpi, "endian", json);
        serializer->OpenFile(file, false);
        WriteDirect(dense, serializer, "dense", mpi);
        WriteDirect(csr, serializer, "csr", mpi);
        serializer->CloseFile();
        const int opposite = 1 - R::Detail::NativeByteOrder();
        MPI_Barrier(mpi.comm);
        if (json)
        {
            nlohmann::json document;
            {
                std::ifstream input(file);
                input >> document;
            }
            document["runtime_array_byte_order"] = opposite;
            {
                std::ofstream output(file);
                output << document;
            }
        }
        else if (mpi.rank == 0)
            SetH5Order(file, opposite);
        MPI_Barrier(mpi.comm);
        serializer->OpenFile(file, true);
        for (const auto &name : {"dense", "csr"})
        {
            auto result = ReadDirect<Mixed>(serializer, name, mpi);
            REQUIRE(result.RowCount() == rows);
            auto read = result.Read();
            for (size_t row = 0; row < rows; ++row)
            {
                CHECK(result.RowSize(row) == (std::string(name) == "dense" ? 2 : lengths[row]));
                for (size_t slot = 0; slot < result.RowSize(row); ++slot)
                {
                    Mixed expected;
                    FillMixed(expected, first + I(row), slot);
                    CHECK(std::memcmp(&read.At(row, slot), &expected, sizeof(Mixed)) == 0);
                }
            }
        }
        serializer->CloseFile();
        MPI_Barrier(mpi.comm);
        if (json || mpi.rank == 0)
            std::filesystem::remove(file);
    }
}

TEST_CASE("Native payload uses caller buffers across dense CSR padded fixed and static-max shapes")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    SUBCASE("H5")
    {
        auto serializer = std::make_shared<DirectSpy<S::SerializerH5>>(mpi);
        serializer->SetCollectiveRW(true, true);
        Shapes(mpi, false, serializer);
    }
    SUBCASE("JSON")
    {
        auto serializer = std::make_shared<DirectSpy<S::SerializerJSON>>();
        serializer->SetUseCodecOnUint8(true); // Native records bypass legacy codec staging.
        Shapes(mpi, true, serializer);
    }
}

TEST_CASE("JSON byte segments support in-memory DOM and retain file mode checks")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto memory = std::make_shared<S::SerializerJSON>();
    memory->CreatePath("/");
    const uint8_t sourceBytes[]{1, 2, 3, 99, 4, 5};
    const std::vector<S::ByteMemorySegment> segments{{0, 3}, {4, 2}};
    memory->WriteByteSegments("bytes", sourceBytes, 6, segments, S::ArrayGlobalOffset_Unknown);
    uint8_t restoredBytes[6]{};
    auto offset = S::ArrayGlobalOffset_Unknown;
    memory->ReadByteSegments("bytes", restoredBytes, 6, segments, offset);
    for (size_t i : {size_t(0), size_t(1), size_t(2), size_t(4), size_t(5)})
        CHECK(restoredBytes[i] == sourceBytes[i]);

    R::Array<double> source(R::RowLayout::Ragged({2, 0, 1}));
    {
        auto write = source.Write();
        write.At(0, 0) = 1.25;
        write.At(0, 1) = -3.5;
        write.At(2, 0) = 9.75;
    }
    R::WriteArray(source, memory, "array", mpi);
    const auto restarted = R::ReadArray<double>(memory, "array", mpi);
    CHECK(restarted.RowCount() == 3);
    CHECK(restarted.RowSize(0) == 2);
    CHECK(restarted.RowSize(1) == 0);
    CHECK(restarted.RowSize(2) == 1);
    auto values = restarted.Read();
    CHECK(values.At(0, 0) == 1.25);
    CHECK(values.At(0, 1) == -3.5);
    CHECK(values.At(2, 0) == 9.75);

    const auto file = File(mpi, "byte-mode", true);
    auto writer = std::make_shared<S::SerializerJSON>();
    writer->OpenFile(file, false);
    CHECK_THROWS(writer->ReadByteSegments("missing", restoredBytes, 6, segments, offset));
    writer->WriteByteSegments("bytes", sourceBytes, 6, segments, S::ArrayGlobalOffset_Unknown);
    writer->CloseFile();

    auto reader = std::make_shared<S::SerializerJSON>();
    reader->OpenFile(file, true);
    CHECK_THROWS(reader->WriteByteSegments("bytes", sourceBytes, 6, segments, S::ArrayGlobalOffset_Unknown));
    std::fill(std::begin(restoredBytes), std::end(restoredBytes), uint8_t(0));
    offset = S::ArrayGlobalOffset_Unknown;
    reader->ReadByteSegments("bytes", restoredBytes, 6, segments, offset);
    for (size_t i : {size_t(0), size_t(1), size_t(2), size_t(4), size_t(5)})
        CHECK(restoredBytes[i] == sourceBytes[i]);
    reader->CloseFile();
    MPI_Barrier(mpi.comm);
    std::filesystem::remove(file);
}

TEST_CASE("Opposite-endian mixed records swap individual declared words and preserve object padding")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    SUBCASE("H5")
    {
        auto serializer = std::make_shared<DirectSpy<S::SerializerH5>>(mpi);
        serializer->SetCollectiveRW(true, true);
        MixedEndian(mpi, false, serializer);
    }
    SUBCASE("JSON")
    {
        auto serializer = std::make_shared<DirectSpy<S::SerializerJSON>>();
        MixedEndian(mpi, true, serializer);
    }
}

TEST_CASE("Malformed native records reject before payload fill and permit pristine retry")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    RegisterMixed();
    const auto file = File(mpi, "malformed", true);
    auto serializer = std::make_shared<DirectSpy<S::SerializerJSON>>();
    R::Array<Mixed> source(R::RowLayout::Ragged({2, 0, 3}));
    {
        auto write = source.Write();
        for (size_t row = 0; row < source.RowCount(); ++row)
            for (size_t slot = 0; slot < source.RowSize(row); ++slot)
                FillMixed(write.At(row, slot), I(row), slot);
    }
    serializer->OpenFile(file, false);
    WriteDirect(source, serializer, "record", mpi);
    serializer->CloseFile();
    nlohmann::json pristine;
    {
        std::ifstream input(file);
        input >> pristine;
    }
    for (int failure = 0; failure < 7; ++failure)
    {
        auto malformed = pristine;
        auto &record = malformed["record"];
        if (failure == 0)
            record["row_counts"][0] = 4;
        if (failure == 1)
            record["row_offsets"][1] = -1;
        if (failure == 2)
            record["element_schema"] = "native-io.other.v1";
        if (failure == 3)
            record["element_definition"] = R::ElementSchema::Record("native-io.mixed.v1", sizeof(Mixed),
                                                                    {{"tag", offsetof(Mixed, tag), R::ElementScalar::U16}, {"zone", offsetof(Mixed, zone), R::ElementScalar::U32}, {"energy", offsetof(Mixed, energy), R::ElementScalar::F64}, {"ids", offsetof(Mixed, ids), R::ElementScalar::I64, 2}},
                                                                    "NativeIOMixed", alignof(Mixed))
                                               .Definition();
        if (failure == 4)
            malformed["runtime_array_byte_order"] = 7;
        if (failure == 5)
            record["element_bytes"] = sizeof(Mixed) + 1;
        if (failure == 6)
            record["elements_global"] = 1;
        {
            std::ofstream output(file);
            output << malformed;
        }
        serializer->OpenFile(file, true);
        const auto before = serializer->restored.size();
        CHECK_THROWS(R::ReadArray<Mixed>(serializer, "record", mpi));
        CHECK(serializer->restored.size() == before);
        CHECK(serializer->GetCurrentPath().empty());
        serializer->CloseFile();
        {
            std::ofstream output(file);
            output << pristine;
        }
        serializer->OpenFile(file, true);
        auto restored = ReadDirect<Mixed>(serializer, "record", mpi);
        CHECK(restored.RowCount() == 3);
        CHECK(restored.RowSize(2) == 3);
        serializer->CloseFile();
    }
    std::filesystem::remove(file);
}

TEST_CASE("Malformed H5 native metadata rejects collectively without truncated CSR association")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto serializer = std::make_shared<DirectSpy<S::SerializerH5>>(mpi);
    serializer->SetCollectiveRW(true, true);
    const auto file = File(mpi, "bad-h5", false);
    R::Array<double> source(R::RowLayout::Ragged({2, 0, 3}));
    Fill(source, 3 * mpi.rank);
    for (int failure = 0; failure < 6; ++failure)
    {
        CAPTURE(failure);
        serializer->OpenFile(file, false);
        R::WriteArray(source, serializer, "record", mpi);
        serializer->CloseFile();
        if (mpi.rank == 0)
        {
            const auto handle = H5Fopen(file.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
            REQUIRE(handle >= 0);
            if (failure == 0 || failure == 1)
            {
                const auto group = H5Gopen2(handle, "/record", H5P_DEFAULT);
                REQUIRE(group >= 0);
                const auto attr = H5Aopen(group, failure == 0 ? "writer_ranks" : "rows_global", H5P_DEFAULT);
                REQUIRE(attr >= 0);
                const I invalid = failure == 0 ? mpi.size + 1 : INT64_MAX;
                CHECK(H5Awrite(attr, H5T_NATIVE_INT64, &invalid) >= 0);
                CHECK(H5Aclose(attr) >= 0);
                CHECK(H5Gclose(group) >= 0);
            }
            else
            {
                const char *path = failure == 4 ? "/record/row_counts" : "/record/row_offsets";
                const auto dataset = H5Dopen2(handle, path, H5P_DEFAULT);
                REQUIRE(dataset >= 0);
                const auto space = H5Dget_space(dataset);
                hsize_t count = 0;
                CHECK(H5Sget_simple_extent_dims(space, &count, nullptr) >= 0);
                std::vector<I> values(size_t(count), 0);
                CHECK(H5Dread(dataset, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) >= 0);
                if (failure == 2)
                    values.front() = 1; // Lost prefix must not silently shift row zero.
                if (failure == 3)
                    --values.back(); // Lost suffix must not truncate the last row.
                if (failure == 4)
                    values.front() = -1;
                if (failure != 5)
                    CHECK(H5Dwrite(dataset, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) >= 0);
                CHECK(H5Sclose(space) >= 0);
                CHECK(H5Dclose(dataset) >= 0);
                if (failure == 5)
                {
                    CHECK(H5Ldelete(handle, path, H5P_DEFAULT) >= 0);
                    --count;
                    const auto shortSpace = H5Screate_simple(1, &count, nullptr);
                    const auto shorter = H5Dcreate2(handle, path, H5T_NATIVE_INT64, shortSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                    REQUIRE(shorter >= 0);
                    CHECK(H5Dwrite(shorter, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) >= 0);
                    CHECK(H5Dclose(shorter) >= 0);
                    CHECK(H5Sclose(shortSpace) >= 0);
                }
            }
            CHECK(H5Fclose(handle) >= 0);
        }
        MPI_Barrier(mpi.comm);
        serializer->OpenFile(file, true);
        const auto calls = serializer->restored.size();
        CHECK_THROWS(R::ReadArray<double>(serializer, "record", mpi, S::ArrayGlobalOffset_EvenSplit));
        CHECK(serializer->restored.size() == calls);
        serializer->CloseFile();
    }
    serializer->OpenFile(file, false);
    R::WriteArray(source, serializer, "record", mpi);
    serializer->CloseFile();
    serializer->OpenFile(file, true);
    {
        auto restored = ReadDirect<double>(serializer, "record", mpi);
        auto read = restored.Read();
        REQUIRE(restored.RowCount() == 3);
        for (size_t row = 0; row < restored.RowCount(); ++row)
            for (size_t column = 0; column < restored.RowSize(row); ++column)
                CHECK(read.At(row, column) == double((3 * mpi.rank + I(row)) * 100 + I(column)) + 0.25);
    }
    serializer->CloseFile();
    MPI_Barrier(mpi.comm);
    if (mpi.rank == 0)
        std::filesystem::remove(file);
}

TEST_CASE("Native collective checkpoint rejects one rank's moved or unpublished Array before file mutation")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto serializer = std::make_shared<DirectSpy<S::SerializerH5>>(mpi);
    serializer->SetCollectiveRW(true, true);
    const auto file = File(mpi, "unpublished-source", false);
    R::Array<double> source(R::RowLayout::Dense(3, 3));
    Fill(source, 3 * mpi.rank);
    serializer->OpenFile(file, false);
    for (bool editing : {false, true})
    {
        std::optional<R::Array<double>> saved;
        std::optional<R::ExclusiveRows> editor;
        if (mpi.rank == 0)
        {
            if (editing)
                editor.emplace(source.EditRows());
            else
                saved.emplace(std::move(source));
        }
        CHECK_THROWS_WITH(R::WriteArray(source, serializer, "rejected", mpi),
                          doctest::Contains("live published source on every rank"));
        CHECK(serializer->written.empty());
        CHECK(serializer->ListCurrentPath().count("rejected") == 0);
        if (editor)
            editor->Commit();
        if (saved)
            source = std::move(*saved);
    }
    WriteDirect(source, serializer, "valid", mpi);
    serializer->CloseFile();
    serializer->OpenFile(file, true);
    const auto restored = ReadDirect<double>(serializer, "valid", mpi);
    CheckValues(restored, 3 * mpi.rank, false);
    serializer->CloseFile();
    MPI_Barrier(mpi.comm);
    if (mpi.rank == 0)
        std::filesystem::remove(file);
}

TEST_CASE("Version two legacy byte CSR remains readable through direct native allocation")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    const auto counts = Counts(mpi);
    const size_t rows = size_t(counts[size_t(mpi.rank)]);
    const I first = Prefix(counts, mpi.rank);
    for (int mode = 0; mode < 6; ++mode)
    {
        const bool json = mode % 3 != 0;
        const auto file = File(mpi, "legacy", json);
        S::SerializerBaseSSP serializer;
        if (json)
        {
            auto writer = std::make_shared<S::SerializerJSON>();
            writer->SetUseCodecOnUint8(mode % 3 == 2);
            serializer = writer;
        }
        else
            serializer = std::make_shared<S::SerializerH5>(mpi);
        // Test fixture only: reproduce the historical v2 writer exactly.
        DNDS::ParArray<uint8_t, DNDS::NonUniformSize> legacy(mpi);
        legacy.Resize(I(rows));
        for (size_t row = 0; row < rows; ++row)
        {
            const size_t width = size_t((first + I(row)) % 4);
            legacy.ResizeRow(I(row), DNDS::rowsize(width * sizeof(double)));
            for (size_t slot = 0; slot < width; ++slot)
            {
                const double value = double((first + I(row)) * 100 + I(slot)) + 0.25;
                std::memcpy(legacy[I(row)] + slot * sizeof(double), &value, sizeof(value));
            }
        }
        legacy.Compress();
        serializer->OpenFile(file, false);
        serializer->CreatePath("legacy");
        serializer->GoToPath("legacy");
        serializer->WriteInt("runtime_array_format", 2);
        serializer->WriteIndex("element_bytes", sizeof(double));
        serializer->WriteInt("writer_ranks", mpi.size);
        const auto legacyIdentity = mode >= 3 ? std::string("old.application.history.f64.v1") : R::ArraySchema<double>();
        serializer->WriteString("element_schema", legacyIdentity);
        legacy.WriteSerializer(serializer, "rows", S::ArrayGlobalOffset_Parts);
        serializer->CloseFile();
        serializer->OpenFile(file, true);
        auto result = R::ReadArray<double>(serializer, "legacy", mpi, S::ArrayGlobalOffset_Unknown, R::Device::Host(), legacyIdentity);
        CHECK(result.RowCount() == rows);
        CheckValues(result, first, true);
        serializer->CloseFile();
        MPI_Barrier(mpi.comm);
        if (json || mpi.rank == 0)
            std::filesystem::remove(file);
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const int result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
