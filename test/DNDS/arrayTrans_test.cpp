#include "DNDS/Array.hpp"
#include "DNDS/ArrayTransformer.hpp"

#include <cstdlib>
using namespace DNDS;

std::vector<double> argD;

/*
running:
mpirun -np 4 valgrind --suppressions=/usr/share/openmpi/openmpi-valgrind.supp --log-file=log_valgrind.log test/arrayTrans_test.exe 1000 1000 100
*/

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

void test_CSR_MAX_LARGE()
{
    int localSiz = 2000;
    int localSizB = 3000;
    int pullSize = 100;
    if (argD.size() == 3)
    {
        localSiz = argD[0];
        localSizB = argD[1];
        pullSize = argD[2];
    }
    MPIInfo mpi;
    mpi.setWorld();
// DNDS_assert(mpi.size <= 2);
// Debug::MPIDebugHold(mpi);

//
#define TCOMP_N 2

#if TCOMP_N == 0
    using tComp = real;
#elif TCOMP_N == 1
    using tComp = std::array<real, 2>;
#elif TCOMP_N == 2
    using tComp = Eigen::Matrix<real, 3, 3>;
#endif

    // using tArray = ParArray<tComp, NonUniformSize>;
    // using tArray = ParArray<tComp, NonUniformSize, 3>;
    // using tArray = ParArray<tComp, NonUniformSize, DynamicSize>;
    using tArray = ParArray<tComp, DynamicSize>;
    // using tArray = ParArray<tComp, 3>;

    std::shared_ptr<tArray> A = std::make_shared<tArray>();
    std::shared_ptr<tArray> A_Ghost = std::make_shared<tArray>();
    auto &Ar = *A;
    auto &Ar_Ghost = *A_Ghost;
    ArrayTransformerType<tArray>::Type A_Trans;
    Ar.setMPI(mpi), Ar_Ghost.setMPI(mpi);
    Ar.AssertConsistent(), Ar_Ghost.AssertConsistent();
    std::cout << "C" << std::endl;
    Ar.Resize(mpi.rank % 2 * localSiz + localSizB, 3); // max 3 or preallocate 3
    DNDS_assert((Ar.GetDataLayout() == CSR && Ar.IfCompressed()) == false);
    Ar.createGlobalMapping();
    for (DNDS::index i = 0; i < Ar.Size(); i++)
    {
        if (tArray::rs == NonUniformSize)
            Ar.ResizeRow(i, Ar.pLGlobalMapping->operator()(mpi.rank, i) % 3 + 1);
        for (rowsize j = 0; j < Ar.RowSize(i); j++)
#if TCOMP_N == 0
            Ar(i, j) = Ar.pLGlobalMapping->operator()(mpi.rank, i);
#elif TCOMP_N == 1
            Ar(i, j)[0] = Ar(i, j)[1] = Ar.pLGlobalMapping->operator()(mpi.rank, i);
#elif TCOMP_N == 2
            Ar(i, j).setIdentity(), Ar(i, j) *= Ar.pLGlobalMapping->operator()(mpi.rank, i);
#endif
    }
    std::vector<int> pullingIndexG;
    srand(mpi.rank);
    for (int i = 0; i < pullSize; i++)
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
#if TCOMP_N == 0
                            std::cout << Ar_Ghost(i, j) << "\t", DNDS_assert(globalIndex == Ar_Ghost(i, j));
#elif TCOMP_N == 1
                    std::cout << Ar_Ghost(i, j)[0] << ", " << Ar_Ghost(i, j)[1] << "\t", DNDS_assert(globalIndex == Ar_Ghost(i, j)[0]);
#elif TCOMP_N == 2
                    std::cout << Ar_Ghost(i, j) << "\n\n", DNDS_assert(globalIndex == Ar_Ghost(i, j)(1, 1));
#endif
                        }
                        std::cout << std::endl;
                    }
                });
    // Expected (non uniform):
    // rank 1:
    // a: a a a
    // b: b b
    // c: c c
    // d: d
    // e: e e e
    // ...
    // rank 2:
    // ...
}

int main(int argc, char *argv[])
{
    DNDS_assert_info(BasicType_To_MPIIntType<uint64_t[4][5]>().second == 20, "BasicType_To_MPIIntType() bad");
    DNDS_assert_info(BasicType_To_MPIIntType<uint64_t[4][5]>().first == MPI_UINT64_T, "BasicType_To_MPIIntType() bad");

    MPI_Init(&argc, &argv);

    for (int i = 1; i < argc; i++)
    {
        double v = std::atof(argv[i]);
        argD.push_back(v);
    }

    // test_CSR();
    test_CSR_MAX_LARGE();

    MPI_Finalize();

    return 0;
}