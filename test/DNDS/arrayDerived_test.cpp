#include "DNDS/ArrayDerived/ArrayAdjacency.hpp"
#include "DNDS/ArrayDerived/ArrayEigenVector.hpp"
#include "DNDS/ArrayDerived/ArrayEigenUniMatrixBatch.hpp"

#include <cstdlib>
using namespace DNDS;

std::vector<double> argD;

/*
running:
valgrind --log-file=log_valgrind.log test/arrayDerived_test.exe
*/

void test_ADJ()
{
    int nRow = 1000;
    if (argD.size() == 1)
    {
        nRow = int(argD[0]);
    }

    // ArrayAdjacency<3> A_adj;
    // ArrayAdjacency<DynamicSize> A_adj;
    // ArrayAdjacency<NonUniformSize, DynamicSize> A_adj;
    // ArrayAdjacency<NonUniformSize, 3> A_adj;
    ArrayAdjacency<NonUniformSize, NonUniformSize> A_adj;

    A_adj.Resize(nRow, 3);

    if (A_adj._dataLayout == CSR || isTABLE_Max(A_adj._dataLayout))
    {
        for (DNDS::index i = 0; i < A_adj.Size(); i++)
            A_adj.ResizeRow(i, i % 3 + 1);
    }
    for (DNDS::index i = 0; i < A_adj.Size(); i++)
    {
        for (rowsize j = 0; j < A_adj.RowSize(i); j++)
            A_adj[i][j] = i;
    }
    A_adj.Compress();
    std::cout << A_adj << std::endl;
}

void test_Vec()
{
    int nRow = 1000;
    if (argD.size() == 1)
    {
        nRow = int(argD[0]);
    }

    // ArrayEigenVector<3> A_v;
    // ArrayEigenVector<DynamicSize> A_v;
    // ArrayEigenVector<NonUniformSize, 3> A_v;
    // ArrayEigenVector<NonUniformSize, DynamicSize> A_v;
    ArrayEigenVector<NonUniformSize> A_v;

    A_v.Resize(nRow, 3);

    if (A_v._dataLayout == CSR || isTABLE_Max(A_v._dataLayout))
    {
        for (DNDS::index i = 0; i < A_v.Size(); i++)
            A_v.ResizeRow(i, i % 3 + 1);
    }
    for (DNDS::index i = 0; i < A_v.Size(); i++)
    {
        A_v[i].setLinSpaced(i, i + 0.5);
    }
    A_v.Compress();
    for (DNDS::index i = 0; i < A_v.Size(); i++)
    {
        std::cout << A_v[i].transpose() << std::endl;
    }
}

void test_UniMatBatch()
{
    int nRow = 1000;
    if (argD.size() == 1)
    {
        nRow = int(argD[0]);
    }

    // ArrayEigenUniMatrixBatch<Eigen::Dynamic, Eigen::Dynamic> A_UMB;
    // ArrayEigenUniMatrixBatch<3, Eigen::Dynamic> A_UMB;
    // ArrayEigenUniMatrixBatch<Eigen::Dynamic, 4> A_UMB;
    ArrayEigenUniMatrixBatch<3, 4> A_UMB;

    A_UMB.ResizeMatrix(3, 4);
    A_UMB.Resize(nRow, -1, -1);

    for (DNDS::index i = 0; i < A_UMB.Size(); i++)
        A_UMB.ResizeBatch(i, i % 3 + 1);

    for (DNDS::index i = 0; i < A_UMB.Size(); i++)
    {
        for (rowsize j = 0; j < A_UMB.BatchSize(i); j++)
            A_UMB(i, j).setIdentity(), A_UMB(i, j) *= i;
    }
    A_UMB.Compress();
    for (DNDS::index i = 0; i < A_UMB.Size(); i++)
    {
        std::cout << i << ": \n";
        auto Batch = A_UMB(i);
        for (auto &j : Batch)
            std::cout << j << std::endl;
    }
}

int main(int argc, char *argv[])
{
    // ! Disable MPI call to help serial mem check
    // MPI_Init(&argc, &argv);

    for (int i = 1; i < argc; i++)
    {
        double v = std::atof(argv[i]);
        argD.push_back(v);
    }

    // test_ADJ();
    // test_Vec();
    test_UniMatBatch();

    // MPI_Finalize();

    return 0;
}