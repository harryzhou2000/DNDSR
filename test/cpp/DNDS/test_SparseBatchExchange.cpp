#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "DNDS/Runtime/SparseBatchExchange.hpp"

namespace R = DNDS::Runtime;
using I = DNDS::index;
struct BatchRecord
{
    I origin, token;
    double value;
};
namespace DNDS::Runtime
{
    template <>
    struct ElementSchemaProvider<BatchRecord>
    {
        static ElementSchema Get()
        {
            return ElementSchema::Record("test.batch-record", sizeof(BatchRecord),
                                         {{"origin", offsetof(BatchRecord, origin), ElementScalar::I64},
                                          {"token", offsetof(BatchRecord, token), ElementScalar::I64},
                                          {"value", offsetof(BatchRecord, value), ElementScalar::F64}});
        }
    };
}

TEST_CASE("Sparse fixed-record batches fan out and return variable proposal counts")
{
    int rank = 0, peers = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &peers);
    auto context = std::make_shared<R::CommunicationContext>(MPI_COMM_WORLD);
    R::Array<BatchRecord> source(R::RowLayout::Dense(5, 1));
    {
        auto write = source.Write();
        for (I iRow = 0; iRow < 5; ++iRow)
            write.View()(iRow, 0) = {rank, iRow, 0.25 + rank * 10 + double(iRow)};
    }
    std::vector<std::vector<I>> send(static_cast<size_t>(peers));
    send[rank].push_back(1);
    send[(rank + 1) % peers].push_back(4);
    send[(rank + 1) % peers].push_back(0);
    send[(rank + peers - 1) % peers].push_back(0);
    auto plan = R::SparseBatchExchange::Build(context, 5, send);
    auto incoming = plan.Exchange(source);
    R::Array<BatchRecord> proposals(incoming.Layout());
    std::vector<std::vector<I>> replies(static_cast<size_t>(peers));
    {
        auto read = incoming.Read();
        auto write = proposals.Write();
        for (int peer = 0; peer < peers; ++peer)
            for (I iRow = plan.ReceiveOffsets()[peer]; iRow < plan.ReceiveOffsets()[peer + 1]; ++iRow)
            {
                const auto record = read.View()(iRow, 0);
                CHECK(record.origin == peer);
                CHECK(record.value == doctest::Approx(0.25 + peer * 10 + double(record.token)));
                write.View()(iRow, 0) = record;
                // Some exact geometric candidates produce no proposals.
                if (record.token % 2 == 0)
                    replies[peer].push_back(iRow);
            }
    }
    auto back = R::SparseBatchExchange::Build(context, incoming.RowCount(), std::move(replies));
    auto received = back.Exchange(proposals);
    auto read = received.Read();
    CHECK(received.RowCount() == 3);
    for (I iRow = 0; iRow < received.RowCount(); ++iRow)
    {
        CHECK(read.View()(iRow, 0).origin == rank);
        CHECK(read.View()(iRow, 0).token % 2 == 0);
    }
}

TEST_CASE("Empty batch peers complete and invalid record layouts fail collectively")
{
    int rank = 0, peers = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &peers);
    auto context = std::make_shared<R::CommunicationContext>(MPI_COMM_WORLD);
    std::vector<std::vector<I>> empty(static_cast<size_t>(peers));
    auto plan = R::SparseBatchExchange::Build(context, 0, empty);
    R::Array<double> source(R::RowLayout::Dense(0, 1));
    CHECK(plan.Exchange(source).RowCount() == 0);
    auto invalid = empty;
    if (rank == 0)
        invalid[0].push_back(0);
    CHECK_THROWS(R::SparseBatchExchange::Build(context, 0, invalid));
    if (peers > 1)
    {
        R::Array<double> wrong(R::RowLayout::Dense(0, rank == 0 ? 1 : 2));
        CHECK_THROWS(plan.Exchange(wrong));
        CHECK_THROWS(plan.Exchange(source, rank));
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int result;
    {
        doctest::Context context(argc, argv);
        result = context.run();
    }
    MPI_Finalize();
    return result;
}
