#include "DNDS/Array.hpp"
#include "DNDS/ArrayTransformer.hpp"

#include <cstdlib>
using namespace DNDS;

// TODO: test large/ test other types/ test size renew
void test_CSR()
{

    MPIInfo mpi;
    mpi.setWorld();
    DNDS_assert(mpi.size <= 2);
    // Debug::MPIDebugHold(mpi);
    using tArray = ParArray<real, NonUniformSize>;
    std::shared_ptr<tArray> A = std::make_shared<tArray>();
    std::shared_ptr<tArray> A_Ghost = std::make_shared<tArray>();
    auto &Ar = *A;
    auto &Ar_Ghost = *A_Ghost;
    ArrayTransformer<real, NonUniformSize> A_Trans;
    Ar.setMPI(mpi), Ar_Ghost.setMPI(mpi);
    std::cout << "C" << std::endl;
    Ar.Resize(4, 3);
    DNDS_assert(Ar.IfCompressed() == false);
    Ar.ResizeRow(mpi.rank + 1, 2);
    if (mpi.rank == 0)
    {
        Ar(0, 0) = Ar(0, 1) = Ar(0, 2) = 1;
        Ar(1, 0) = Ar(1, 1) = 2;
        Ar(2, 0) = Ar(2, 1) = Ar(2, 2) = 31;
        Ar(3, 0) = Ar(3, 1) = Ar(3, 2) = 2;
    }
    else
    {
        Ar(0, 0) = Ar(0, 1) = Ar(0, 2) = 44;
        Ar(1, 0) = Ar(1, 1) = Ar(1, 2) = 444;
        Ar(2, 0) = Ar(2, 1) = 4444;
        Ar(3, 0) = Ar(3, 1) = Ar(3, 2) = 44444;
    }
    std::cout << "B" << std::endl;
    Ar.Compress();
    A_Trans.setFatherSon(A, A_Ghost);
    A_Trans.createFatherGlobalMapping();
    std::vector<int> pullingIndexG;
    if (mpi.rank == 0)
    {
        pullingIndexG.push_back(1);
        pullingIndexG.push_back(3);
    }
    else
    {
        pullingIndexG.push_back(0);
        pullingIndexG.push_back(2);
        pullingIndexG.push_back(4);
        pullingIndexG.push_back(6);
    }
    std::cout << "A" << std::endl;
    A_Trans.createGhostMapping(pullingIndexG);
    A_Trans.createMPITypes();
    DNDS_assert(A_Trans.pPullTypeVec);
    A_Trans.pullOnce();
    MPISerialDo(mpi, [&]()
                {
        std::cout << "rank " << mpi.rank << std::endl;
        std::cout << Ar_Ghost << std::endl; });

    // "
    // C
    // C
    // B
    // B
    // A
    // A
    // rank 0
    // 2       2
    // 2       2       2

    // rank 1
    // 1       1       1
    // 31      31      31
    // 44      44      44
    // 4444    4444
    // "
}

void test_CSR_LARGE()
{
    MPIInfo mpi;
    mpi.setWorld();
    // DNDS_assert(mpi.size <= 2);
    // Debug::MPIDebugHold(mpi);
    using tArray = ParArray<real, NonUniformSize>;
    std::shared_ptr<tArray> A = std::make_shared<tArray>();
    std::shared_ptr<tArray> A_Ghost = std::make_shared<tArray>();
    auto &Ar = *A;
    auto &Ar_Ghost = *A_Ghost;
    ArrayTransformer<real, NonUniformSize> A_Trans;
    Ar.setMPI(mpi), Ar_Ghost.setMPI(mpi);
    std::cout << "C" << std::endl;
    Ar.Resize(mpi.rank % 2 * 20000 + 30000);
    DNDS_assert(Ar.IfCompressed() == false);
    Ar.createGlobalMapping();
    for (DNDS::index i = 0; i < Ar.Size(); i++)
    {
        Ar.ResizeRow(i, i % 3 + 1);
        for (rowsize j = 0; j < Ar.RowSize(i); j++)
            Ar(i, j) = Ar.pLGlobalMapping->operator()(mpi.rank, i);
    }
    std::vector<int> pullingIndexG;
    srand(mpi.rank);
    for (int i = 0; i < 1000; i++)
        pullingIndexG.push_back(rand() % Ar.pLGlobalMapping->globalSize());
    A_Trans.setFatherSon(A, A_Ghost);
    A_Trans.createGhostMapping(pullingIndexG);
    A_Trans.createMPITypes();
    DNDS_assert(A_Trans.pPullTypeVec);
    A_Trans.pullOnce();
    MPISerialDo(mpi,
                [&]()
                {
                    std::cout << "rank " << mpi.rank << std::endl;
                    for (DNDS::index i = 0; i < Ar_Ghost.Size(); i++)
                    {
                        DNDS::index globalIndex = A_Trans.pLGhostMapping->operator()(-1, i + Ar.Size());
                        //! this is a call convention to acquire ghost_local -> global
                        std::cout << globalIndex << ": ";
                        for (rowsize j = 0; j < Ar_Ghost.RowSize(i); j++)
                        {
                            std::cout << Ar_Ghost(i, j) << "\t";
                            DNDS_assert(globalIndex == Ar_Ghost(i, j));
                        }
                        std::cout << std::endl;
                        
                    }
                });
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    // test_CSR();
    test_CSR_LARGE();

    MPI_Finalize();

    return 0;
}