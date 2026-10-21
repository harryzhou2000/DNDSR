#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "DNDS/Runtime/ArrayIO.hpp"
#include "DNDS/Serializer/SerializerH5.hpp"
#include "DNDS/Serializer/SerializerJSON.hpp"
#include <filesystem>
#include <unistd.h>

using namespace DNDS::Runtime;
namespace S = DNDS::Serializer;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    doctest::Context ctx;
    ctx.applyCommandLine(argc, argv);
    int result = ctx.run();
    MPI_Finalize();
    return result;
}

TEST_CASE("Native parallel checkpoint matching partitions and direct read")
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    int pid = getpid();
    MPI_Bcast(&pid, 1, MPI_INT, 0, mpi.comm);
    const auto file = "__runtime_array_" + std::to_string(pid) + ".h5";
    const size_t localRows = mpi.rank == 0 ? 3 : 0;
    Array<double> a(RowLayout::Ragged(localRows ? std::vector<size_t>{2, 0, 3} : std::vector<size_t>{}));
    if (localRows)
    {
        auto write = a.Write();
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < a.RowSize(i); ++j)
                write.At(i, j) = i * 10 + j;
    }
    auto serializer = std::make_shared<S::SerializerH5>(mpi);
    serializer->SetCollectiveRW(true, true);
    serializer->OpenFile(file, false);
    WriteArray(a, serializer, "state", mpi);
#ifdef DNDS_USE_CUDA
    WriteArray(a.To(Device::CUDA(0)), serializer, "gpu_state", mpi);
#endif
    serializer->CloseFile();
    serializer->OpenFile(file, true);
    auto matching = ReadArray<double>(serializer, "state", mpi);
    CHECK(matching.Rows() == localRows);
    if (localRows)
        CHECK(matching.Read().At(2, 2) == 22);
#ifdef DNDS_USE_CUDA
    auto deviceRead = ReadArray<double>(serializer, "gpu_state", mpi, S::ArrayGlobalOffset_Unknown, Device::CUDA(0));
    if (localRows)
        CHECK(deviceRead.To(Device::Host()).Read().At(2, 2) == 22);
#endif
    serializer->CloseFile();
    serializer->OpenFile(file, true);
    // Each rank may directly request any contiguous row partition, including overlap.
    auto direct = ReadArray<double>(serializer, "state", mpi, {1, 2});
    CHECK(direct.Rows() == 1);
    CHECK(direct.Read().At(0, 2) == 22);
    serializer->CloseFile();
    serializer->OpenFile(file, true);
    auto even = ReadArray<double>(serializer, "state", mpi, S::ArrayGlobalOffset_EvenSplit);
    int64_t total = even.Rows();
    MPI_Allreduce(MPI_IN_PLACE, &total, 1, MPI_INT64_T, MPI_SUM, mpi.comm);
    CHECK(total == 3);
    CHECK_THROWS(ReadArray<int32_t>(serializer, "state", mpi));
    serializer->CloseFile();
    serializer.reset();
    // Change communicator size: direct partition read is independent of writer ranks.
    if (mpi.rank == 0)
    {
        DNDS::MPIInfo single(MPI_COMM_SELF);
        auto reader = std::make_shared<S::SerializerH5>(single);
        reader->OpenFile(file, true);
        auto all = ReadArray<double>(reader, "state", single, S::ArrayGlobalOffset_EvenSplit);
        CHECK(all.Rows() == 3);
        CHECK(all.Read().At(2, 1) == 21);
        if (mpi.size > 1)
            CHECK_THROWS(ReadArray<double>(reader, "state", single));
        reader->CloseFile();
    }
    MPI_Barrier(mpi.comm);
    if (mpi.rank == 0)
        std::filesystem::remove(file);
}

TEST_CASE("Native JSON checkpoint padded rows and POD schema")
{
    struct Info
    {
        int id;
        double x;
    };
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    const auto file = "__runtime_json_" + std::to_string(getpid()) + ".json";
    auto serializer = std::make_shared<S::SerializerJSON>();
    Array<Info> a(RowLayout::Dense(2, 1, 3));
    a.Write().At(1, 0) = {mpi.rank, 0.25};
    serializer->OpenFile(file, false);
    WriteArray(a, serializer, "metadata", mpi, "metadata.v1");
    serializer->CloseFile();
    serializer->OpenFile(file, true);
    auto b = ReadArray<Info, FixedRows<1>>(serializer, "metadata", mpi, S::ArrayGlobalOffset_Unknown, Device::Host(), "metadata.v1");
    CHECK(b.Layout().Stride() == 1);
    CHECK(b.Read().At(1, 0).id == mpi.rank);
    CHECK(b.Read().At(1, 0).x == 0.25);
    serializer->CloseFile();
    std::filesystem::remove(file);
}
