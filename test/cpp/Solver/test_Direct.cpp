/**
 * @file test_Direct.cpp
 * @brief Unit tests for block-sparse LU factorization in Solver/Direct.hpp.
 *
 * Tests a 2D periodic Laplacian with 2x2 blocks on an NxN grid:
 *   A = 5-point stencil with periodic boundary (4 neighbors per cell).
 *   Each cell (i,j) connects to (i±1,j) and (i,j±1) with wrap-around.
 *   Diag = 4*I + delta*diag(i,2i), off-diag = -I (non-symmetric diag
 *   to exercise the full LU, not just LDLT).
 *
 * Verifies:
 *   - SerialSymLUStructure: symbolic factorization with fill-in
 *   - LocalLUBase: MatMul (A*x == b), InPlaceDecompose, Solve (A^{-1}b == x)
 *   - ILU(0): approximate solve, residual reduction
 *   - Complete LU (iluCode=-1): exact solve
 *   - ILU(1): intermediate fill, better than ILU(0)
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include "DNDS/Core/Defines.hpp"
#include "DNDS/MPI/MPI.hpp"
#include "Solver/Direct.hpp"
#include "DNDS/Algebra/BlockMatrixStorage.hpp"
#include "Solver/NativeBlockLU.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <vector>
#include <memory>
#include <cmath>
#include <optional>

using namespace DNDS;

static constexpr int BS = 2; // block size
using Block = Eigen::Matrix<real, BS, BS>;
using BVec = Eigen::Vector<real, BS>;

// ===================================================================
// Concrete LocalLU derived class for 2x2 dense blocks
// ===================================================================
struct TestBlockLU;
using tVec = std::vector<BVec>;
using tBase = Direct::LocalLUBase<TestBlockLU, Block, tVec>;

struct TestBlockLU : public tBase
{
    std::vector<Block> diag;
    std::vector<std::vector<Block>> lower;
    std::vector<std::vector<Block>> upper;

    TestBlockLU(ssp<Direct::SerialSymLUStructure> s, int N)
        : tBase(s), diag(N), lower(N), upper(N)
    {
        for (DNDS::index i = 0; i < N; i++)
        {
            diag[i].setZero();
            lower[i].resize(s->lowerTriStructure[i].size());
            upper[i].resize(s->upperTriStructure[i].size());
            for (auto &b : lower[i])
                b.setZero();
            for (auto &b : upper[i])
                b.setZero();
        }
    }

    Block &GetDiag(DNDS::index i) { return diag[i]; }
    Block &GetLower(DNDS::index i, int ij) { return lower[i][ij]; }
    Block &GetUpper(DNDS::index i, int ij) { return upper[i][ij]; }
    Block InvertDiag(const Block &v) { return v.inverse(); }
};

// ===================================================================
// 2D periodic grid: NxN cells, cell (r,c) has 4 neighbors with wrap
// ===================================================================
static constexpr int G = 4; // grid side length => G*G = 16 cells
static const int NCells = G * G;

static DNDS::index cellIdx(int r, int c)
{
    return ((r % G) + G) % G * G + ((c % G) + G) % G;
}

static std::vector<std::vector<DNDS::index>> build2DPeriodicAdj()
{
    std::vector<std::vector<DNDS::index>> adj(NCells);
    for (int r = 0; r < G; r++)
        for (int c = 0; c < G; c++)
        {
            DNDS::index me = cellIdx(r, c);
            adj[me].push_back(cellIdx(r - 1, c));
            adj[me].push_back(cellIdx(r + 1, c));
            adj[me].push_back(cellIdx(r, c - 1));
            adj[me].push_back(cellIdx(r, c + 1));
        }
    return adj;
}

/// Fill 2D periodic Laplacian with non-symmetric diagonal perturbation.
/// Diag_i = 4*I + delta * diag(i, 2*i)  (makes the diagonal non-symmetric).
/// Off-diag entries corresponding to original adjacency = -I.
/// Fill-in entries (from ILU(k>0) or complete LU) = 0 (zeroed by constructor).
/// Uses cell2cellFaceVLocal2FullRowPos to identify which entries in
/// lower/upper correspond to original neighbors.
static void fill2DLaplacian(TestBlockLU &lu, double delta = 0.1)
{
    auto adj = build2DPeriodicAdj();
    Block I2 = Block::Identity();
    auto &symLU = *lu.symLU;

    for (int i = 0; i < NCells; i++)
    {
        Block D = 4.0 * I2;
        D(0, 0) += delta * i;
        D(1, 1) += delta * 2 * i;
        lu.diag[i] = D;

        // Only fill original adjacency entries with -I; fill-in stays 0
        auto &c2cPos = symLU.cell2cellFaceVLocal2FullRowPos[i];
        for (int ic2c = 0; ic2c < (int)adj[i].size(); ic2c++)
        {
            int pos = c2cPos[ic2c]; // 0 = diag, 1..nLower = lower, nLower+1.. = upper
            if (pos == 0)
                continue; // diagonal (self)
            int nLower = symLU.lowerTriStructure[i].size();
            if (pos <= nLower)
                lu.lower[i][pos - 1] = -I2;
            else
                lu.upper[i][pos - nLower - 1] = -I2;
        }
    }
}

static MPIInfo g_mpi;

// ===================================================================
// Tests
// ===================================================================

TEST_CASE("Direct 2D periodic: symbolic factorization has fill-in")
{
    auto symLU = std::make_shared<Direct::SerialSymLUStructure>(g_mpi, NCells);
    auto adj = build2DPeriodicAdj();
    symLU->ObtainSymmetricSymbolicFactorization(adj, {0, NCells}, -1); // complete LU

    // Complete LU on a 2D grid has fill-in: some rows should have > 4 entries
    int maxLower = 0;
    for (int i = 0; i < NCells; i++)
        maxLower = std::max(maxLower, (int)symLU->lowerTriStructure[i].size());

    std::cout << "[2D-sym] maxLower entries = " << maxLower
              << " (> 4 means fill-in)" << std::endl;
    CHECK(maxLower > 2); // definitely more than the original 2 lower neighbors
}

TEST_CASE("Direct symbolic factorization ignores self edges")
{
    std::vector<std::vector<DNDS::index>> adj = {
        {0, 1},
        {0, 1},
    };

    auto symLU = std::make_shared<Direct::SerialSymLUStructure>(g_mpi, 2);
    symLU->ObtainSymmetricSymbolicFactorization(adj, {0, 2}, 0);

    CHECK(symLU->lowerTriStructure[0].empty());
    CHECK(symLU->upperTriStructure[1].empty());
    REQUIRE(symLU->cell2cellFaceVLocal2FullRowPos[0].size() == 2);
    REQUIRE(symLU->cell2cellFaceVLocal2FullRowPos[1].size() == 2);
    CHECK(symLU->cell2cellFaceVLocal2FullRowPos[0][0] == 0);
    CHECK(symLU->cell2cellFaceVLocal2FullRowPos[1][1] == 0);
}

TEST_CASE("Direct 2D periodic: MatMul is correct")
{
    auto symLU = std::make_shared<Direct::SerialSymLUStructure>(g_mpi, NCells);
    auto adj = build2DPeriodicAdj();
    symLU->ObtainSymmetricSymbolicFactorization(adj, {0, NCells}, 0);

    TestBlockLU lu(symLU, NCells);
    fill2DLaplacian(lu);

    // Build a known x vector
    tVec x(NCells), Ax(NCells);
    for (int i = 0; i < NCells; i++)
    {
        x[i] = BVec::Zero();
        x[i](0) = std::sin(0.5 * i);
        x[i](1) = std::cos(0.3 * i);
        Ax[i] = BVec::Zero();
    }

    lu.MatMul(x, Ax);

    // Verify by manual computation for cell 0
    int r0 = 0, c0 = 0;
    BVec expected = lu.diag[0] * x[0];
    // 4 neighbors of cell 0 (periodic: left=G-1, right=1, up=(G-1)*G, down=G)
    DNDS::index neighbors[] = {cellIdx(r0 - 1, c0), cellIdx(r0 + 1, c0),
                               cellIdx(r0, c0 - 1), cellIdx(r0, c0 + 1)};
    for (auto nb : neighbors)
        expected -= x[nb]; // off-diag = -I

    real err0 = (Ax[0] - expected).norm();
    CHECK(err0 < 1e-14);

    // Check all entries are finite
    for (int i = 0; i < NCells; i++)
        CHECK_FALSE(Ax[i].hasNaN());
}

TEST_CASE("Direct 2D periodic: complete LU solve is exact")
{
    auto symLU = std::make_shared<Direct::SerialSymLUStructure>(g_mpi, NCells);
    auto adj = build2DPeriodicAdj();
    symLU->ObtainSymmetricSymbolicFactorization(adj, {0, NCells}, -1); // complete LU

    // Build A for MatMul, then decompose a copy for Solve
    TestBlockLU luMat(symLU, NCells);
    fill2DLaplacian(luMat);

    tVec x_known(NCells), b(NCells), x_solved(NCells);
    for (int i = 0; i < NCells; i++)
    {
        x_known[i] = BVec::Zero();
        x_known[i](0) = std::sin(0.7 * i + 0.1);
        x_known[i](1) = std::cos(0.4 * i + 0.2);
        b[i] = BVec::Zero();
        x_solved[i] = BVec::Zero();
    }

    luMat.MatMul(x_known, b); // b = A * x_known

    TestBlockLU luSolve(symLU, NCells);
    fill2DLaplacian(luSolve);
    luSolve.InPlaceDecompose();
    luSolve.Solve(b, x_solved);

    real maxErr = 0;
    for (int i = 0; i < NCells; i++)
        maxErr = std::max(maxErr, (x_solved[i] - x_known[i]).norm());

    std::cout << "[2D-LU-full] maxErr = " << std::scientific << maxErr << std::endl;
    CHECK(maxErr < 1e-10);
}

TEST_CASE("Direct 2D periodic: ILU(0) is approximate but reduces residual")
{
    auto symLU0 = std::make_shared<Direct::SerialSymLUStructure>(g_mpi, NCells);
    auto adj = build2DPeriodicAdj();
    symLU0->ObtainSymmetricSymbolicFactorization(adj, {0, NCells}, 0); // ILU(0)

    // Build b = A * x_known using the original (non-decomposed) matrix
    auto symLUorig = std::make_shared<Direct::SerialSymLUStructure>(g_mpi, NCells);
    symLUorig->ObtainSymmetricSymbolicFactorization(adj, {0, NCells}, 0);
    TestBlockLU luOrig(symLUorig, NCells);
    fill2DLaplacian(luOrig);

    tVec x_known(NCells), b(NCells), x_ilu(NCells), Ax_ilu(NCells);
    for (int i = 0; i < NCells; i++)
    {
        x_known[i] = BVec::Zero();
        x_known[i](0) = std::sin(0.7 * i + 0.1);
        x_known[i](1) = std::cos(0.4 * i + 0.2);
        b[i] = BVec::Zero();
        x_ilu[i] = BVec::Zero();
        Ax_ilu[i] = BVec::Zero();
    }

    luOrig.MatMul(x_known, b);

    // ILU(0) solve
    TestBlockLU luILU(symLU0, NCells);
    fill2DLaplacian(luILU);
    luILU.InPlaceDecompose();
    luILU.Solve(b, x_ilu);

    // Compute residual ||b - A*x_ilu||
    // Need a fresh non-decomposed matrix for MatMul
    auto symLUmul = std::make_shared<Direct::SerialSymLUStructure>(g_mpi, NCells);
    symLUmul->ObtainSymmetricSymbolicFactorization(adj, {0, NCells}, 0);
    TestBlockLU luMul(symLUmul, NCells);
    fill2DLaplacian(luMul);
    luMul.MatMul(x_ilu, Ax_ilu);

    real residualNorm = 0, bNorm = 0;
    for (int i = 0; i < NCells; i++)
    {
        residualNorm += (b[i] - Ax_ilu[i]).squaredNorm();
        bNorm += b[i].squaredNorm();
    }
    residualNorm = std::sqrt(residualNorm);
    bNorm = std::sqrt(bNorm);

    real relResidual = residualNorm / bNorm;
    std::cout << "[2D-ILU0] relResidual = " << std::scientific << relResidual << std::endl;

    // ILU(0) on a 2D periodic grid is NOT exact (has fill-in), but should reduce residual
    CHECK(relResidual < 0.5);   // significant reduction from initial residual of ~1
    CHECK(relResidual > 1e-14); // not exact
}

/// Use ILU as preconditioner in fixed-point iteration: x_{k+1} = x_k + M^{-1}(b - Ax_k).
/// Returns error norm after nIter iterations.
static real iluPrecIterSolve(int iluCode, const tVec &b, const tVec &x_known, int nIter)
{
    auto adj = build2DPeriodicAdj();

    // Build non-decomposed A for MatMul
    auto symA = std::make_shared<Direct::SerialSymLUStructure>(g_mpi, NCells);
    symA->ObtainSymmetricSymbolicFactorization(adj, {0, NCells}, 0);
    TestBlockLU luA(symA, NCells);
    fill2DLaplacian(luA);

    // Build ILU preconditioner M
    auto symM = std::make_shared<Direct::SerialSymLUStructure>(g_mpi, NCells);
    symM->ObtainSymmetricSymbolicFactorization(adj, {0, NCells}, iluCode);
    TestBlockLU luM(symM, NCells);
    fill2DLaplacian(luM);
    luM.InPlaceDecompose();

    tVec x(NCells), r(NCells), z(NCells), Ax(NCells);
    for (int i = 0; i < NCells; i++)
        x[i] = BVec::Zero();

    for (int iter = 0; iter < nIter; iter++)
    {
        luA.MatMul(x, Ax);
        for (int i = 0; i < NCells; i++)
            r[i] = b[i] - Ax[i];
        luM.Solve(r, z);
        for (int i = 0; i < NCells; i++)
            x[i] += z[i];
    }

    real err = 0;
    for (int i = 0; i < NCells; i++)
        err += (x[i] - x_known[i]).squaredNorm();
    return std::sqrt(err);
}

TEST_CASE("Direct 2D periodic: ILU(1) converges faster than ILU(0) as preconditioner")
{
    auto adj = build2DPeriodicAdj();

    // Build b = A * x_known
    auto symLUorig = std::make_shared<Direct::SerialSymLUStructure>(g_mpi, NCells);
    symLUorig->ObtainSymmetricSymbolicFactorization(adj, {0, NCells}, 0);
    TestBlockLU luOrig(symLUorig, NCells);
    fill2DLaplacian(luOrig);

    tVec x_known(NCells), b(NCells);
    for (int i = 0; i < NCells; i++)
    {
        x_known[i] = BVec::Zero();
        x_known[i](0) = std::sin(0.7 * i + 0.1);
        x_known[i](1) = std::cos(0.4 * i + 0.2);
        b[i] = BVec::Zero();
    }
    luOrig.MatMul(x_known, b);

    int nIter = 10;
    real errILU0 = iluPrecIterSolve(0, b, x_known, nIter);
    real errILU1 = iluPrecIterSolve(1, b, x_known, nIter);
    real errFull = iluPrecIterSolve(-1, b, x_known, nIter);

    std::cout << "[2D-ILU-prec-iter] ILU(0)=" << std::scientific << errILU0
              << " ILU(1)=" << errILU1
              << " Full=" << errFull
              << " (after " << nIter << " iters)" << std::endl;

    CHECK(errFull < 1e-10);   // complete LU converges in 1 iteration
    CHECK(errILU1 < errILU0); // ILU(1) converges faster
    CHECK(errILU0 < 1e-3);    // ILU(0) also converges
}

// ===================================================================
// LDLT: concrete derived class for symmetric 2x2 blocks
// ===================================================================
struct TestBlockLDLT;
using tLDLTBase = Direct::LocalLDLTBase<TestBlockLDLT, Block, tVec>;

struct TestBlockLDLT : public tLDLTBase
{
    std::vector<Block> diag;
    std::vector<std::vector<Block>> lower;

    TestBlockLDLT(ssp<Direct::SerialSymLUStructure> s, int N)
        : tLDLTBase(s), diag(N), lower(N)
    {
        for (DNDS::index i = 0; i < N; i++)
        {
            diag[i].setZero();
            lower[i].resize(s->lowerTriStructure[i].size());
            for (auto &b : lower[i])
                b.setZero();
        }
    }

    Block &GetDiag(DNDS::index i) { return diag[i]; }
    Block &GetLower(DNDS::index i, int ij) { return lower[i][ij]; }
    Block InvertDiag(const Block &v) { return v.inverse(); }
};

/// Fill 2D periodic Laplacian with SYMMETRIC diagonal: Diag = (4+0.1*i)*I.
/// Lower = -I at original adjacency positions; fill-in positions stay 0.
static void fill2DLaplacianSym(TestBlockLDLT &lu, double delta = 0.1)
{
    auto adj = build2DPeriodicAdj();
    Block I2 = Block::Identity();
    auto &symLU = *lu.symLU;

    for (int i = 0; i < NCells; i++)
    {
        lu.diag[i] = (4.0 + delta * i) * I2;
        auto &c2cPos = symLU.cell2cellFaceVLocal2FullRowPos[i];
        int nLower = symLU.lowerTriStructure[i].size();
        for (int ic2c = 0; ic2c < (int)adj[i].size(); ic2c++)
        {
            int pos = c2cPos[ic2c];
            if (pos == 0)
                continue;
            // LDLT only stores lower; pos 1..nLower is lower
            if (pos <= nLower)
                lu.lower[i][pos - 1] = -I2;
            // Upper positions: LDLT uses lower^T, no explicit upper storage
        }
    }
}

// Symmetric version for LU (MatMul reference) — same fill logic
static void fill2DLaplacianSymLU(TestBlockLU &lu, double delta = 0.1)
{
    auto adj = build2DPeriodicAdj();
    Block I2 = Block::Identity();
    auto &symLU = *lu.symLU;

    for (int i = 0; i < NCells; i++)
    {
        lu.diag[i] = (4.0 + delta * i) * I2;
        auto &c2cPos = symLU.cell2cellFaceVLocal2FullRowPos[i];
        for (int ic2c = 0; ic2c < (int)adj[i].size(); ic2c++)
        {
            int pos = c2cPos[ic2c];
            if (pos == 0)
                continue;
            int nLower = symLU.lowerTriStructure[i].size();
            if (pos <= nLower)
                lu.lower[i][pos - 1] = -I2;
            else
                lu.upper[i][pos - nLower - 1] = -I2;
        }
    }
}

// ===================================================================
// LDLT tests
// ===================================================================

TEST_CASE("Direct 2D periodic LDLT: MatMul matches LU MatMul for symmetric system")
{
    auto adj = build2DPeriodicAdj();

    auto symLU = std::make_shared<Direct::SerialSymLUStructure>(g_mpi, NCells);
    symLU->ObtainSymmetricSymbolicFactorization(adj, {0, NCells}, -1);

    // LU reference
    TestBlockLU luRef(symLU, NCells);
    fill2DLaplacianSymLU(luRef);

    // LDLT
    TestBlockLDLT ldlt(symLU, NCells);
    fill2DLaplacianSym(ldlt);

    tVec x(NCells), Ax_lu(NCells), Ax_ldlt(NCells);
    for (int i = 0; i < NCells; i++)
    {
        x[i] = BVec::Zero();
        x[i](0) = std::sin(0.3 * i);
        x[i](1) = std::cos(0.7 * i);
        Ax_lu[i] = BVec::Zero();
        Ax_ldlt[i] = BVec::Zero();
    }

    luRef.MatMul(x, Ax_lu);
    ldlt.MatMul(x, Ax_ldlt);

    real maxDiff = 0;
    for (int i = 0; i < NCells; i++)
        maxDiff = std::max(maxDiff, (Ax_lu[i] - Ax_ldlt[i]).norm());

    std::cout << "[2D-LDLT-MatMul] maxDiff = " << std::scientific << maxDiff << std::endl;
    CHECK(maxDiff < 1e-14);
}

TEST_CASE("Direct 2D periodic LDLT: complete decompose + solve is exact")
{
    auto adj = build2DPeriodicAdj();

    // Build reference b = A*x_known using LU MatMul
    auto symLU = std::make_shared<Direct::SerialSymLUStructure>(g_mpi, NCells);
    symLU->ObtainSymmetricSymbolicFactorization(adj, {0, NCells}, -1);
    TestBlockLU luRef(symLU, NCells);
    fill2DLaplacianSymLU(luRef);

    tVec x_known(NCells), b(NCells), x_solved(NCells);
    for (int i = 0; i < NCells; i++)
    {
        x_known[i] = BVec::Zero();
        x_known[i](0) = std::sin(0.7 * i + 0.1);
        x_known[i](1) = std::cos(0.4 * i + 0.2);
        b[i] = BVec::Zero();
        x_solved[i] = BVec::Zero();
    }
    luRef.MatMul(x_known, b);

    // Solve with LDLT
    auto symLU2 = std::make_shared<Direct::SerialSymLUStructure>(g_mpi, NCells);
    symLU2->ObtainSymmetricSymbolicFactorization(adj, {0, NCells}, -1);
    TestBlockLDLT ldlt(symLU2, NCells);
    fill2DLaplacianSym(ldlt);
    ldlt.InPlaceDecompose();
    ldlt.Solve(b, x_solved);

    real maxErr = 0;
    for (int i = 0; i < NCells; i++)
        maxErr = std::max(maxErr, (x_solved[i] - x_known[i]).norm());

    std::cout << "[2D-LDLT-Solve] maxErr = " << std::scientific << maxErr << std::endl;
    CHECK(maxErr < 1e-10);
}

// ===================================================================
// Native storage and staged symbolic/numeric work
// ===================================================================
TEST_CASE("Direct core ordering is stable within local parts and precedes symbolic fill")
{
    auto symbolic = std::make_shared<Direct::SerialSymLUStructure>(g_mpi, 6);
    symbolic->SetCoreFirstOrdering({2, 0, 5}, {0, 3, 6});
    CHECK(symbolic->localFillOrderingNew2Old == std::vector<DNDS::index>{0, 2, 1, 5, 3, 4});
    CHECK(symbolic->localPartCoreEnds == std::vector<DNDS::index>{2, 4});
    std::vector<std::vector<DNDS::index>> adjacency{{1}, {0, 2}, {1}, {4}, {3, 5}, {4}};
    symbolic->ObtainSymmetricSymbolicFactorization(adjacency, {0, 3, 6}, 0);
    CHECK(symbolic->ReadySolveRows({0, 2, 5}, true) == std::vector<DNDS::index>{0, 2, 5});
    CHECK_THROWS(symbolic->SetCoreFirstOrdering({0}));
    auto invalid = std::make_shared<Direct::SerialSymLUStructure>(g_mpi, 3);
    CHECK_THROWS(invalid->SetCoreFirstOrdering({1, 1}));
    CHECK_THROWS(invalid->SetCoreFirstOrdering({3}));
    CHECK(invalid->localFillOrderingNew2Old.empty());
    CHECK_THROWS(invalid->ReadySolveRows({0}, true));
}

TEST_CASE("ILU zero-fill support does not bound upper triangular readiness")
{
    auto symbolic = std::make_shared<Direct::SerialSymLUStructure>(g_mpi, 4);
    std::vector<std::vector<DNDS::index>> adjacency{{1}, {0, 2}, {1, 3}, {2}};
    symbolic->ObtainSymmetricSymbolicFactorization(adjacency, {0, 4}, 0);
    CHECK(symbolic->ReadySolveRows({0, 1, 2}, true) == std::vector<DNDS::index>{0, 1, 2});
    CHECK(symbolic->ReadySolveRows({0, 1, 2}, false).empty());
    CHECK(symbolic->ReadySolveRows({0, 1, 2, 3}, false) == std::vector<DNDS::index>{3, 2, 1, 0});
}

TEST_CASE("Generic ranged factors and separate triangular outputs preserve the full solve")
{
    for (int fill : {0, 1, -1})
    {
        auto symbolic = std::make_shared<Direct::SerialSymLUStructure>(g_mpi, NCells);
        symbolic->SetCoreFirstOrdering({0, 3, 7, 12});
        symbolic->ObtainSymmetricSymbolicFactorization(build2DPeriodicAdj(), {0, NCells}, fill);
        TestBlockLU full(symbolic, NCells), staged(symbolic, NCells);
        fill2DLaplacian(full);
        fill2DLaplacian(staged);
        full.InPlaceDecompose();
        staged.BeginNumericFactorization();
        CHECK_THROWS(staged.DecomposeRange(1, 4));
        staged.DecomposeRange(0, 4);
        CHECK_FALSE(staged.isDecomposed);
        staged.DecomposeRange(4, NCells);
        CHECK(staged.isDecomposed);
        CHECK_THROWS(staged.DecomposeRange(0, NCells));
        tVec rhs(NCells), reference(NCells), forward(NCells), output(NCells);
        for (int i = 0; i < NCells; ++i)
            rhs[i] = BVec{std::sin(.37 * i), std::cos(.51 * i)};
        full.Solve(rhs, reference);
        staged.SolveLowerRange(rhs, forward, 0, 4);
        staged.SolveLowerRange(rhs, forward, 4, NCells);
        staged.SolveUpperRange(forward, output, 4, NCells);
        staged.SolveUpperRange(forward, output, 0, 4);
        for (int i = 0; i < NCells; ++i)
            CHECK((output[i] - reference[i]).norm() < 1e-14);
        CHECK_THROWS(staged.SolveLowerRange(rhs, forward, -1, 4));
    }
}

TEST_CASE("Rebuilding symbolic factors replaces previous row incidences")
{
    const std::vector<std::vector<DNDS::index>> adjacency{{1}, {0, 2}, {1, 3}, {2}};
    for (int fill : {0, 1, -1})
    {
        auto symbolic = std::make_shared<Direct::SerialSymLUStructure>(g_mpi, 4);
        symbolic->ObtainSymmetricSymbolicFactorization(adjacency, {0, 4}, fill);
        const auto lower = symbolic->lowerTriStructure, upper = symbolic->upperTriStructure;
        const auto lowerNew = symbolic->lowerTriStructureNew, upperNew = symbolic->upperTriStructureNew;
        const auto generation = symbolic->generation;
        symbolic->ObtainSymmetricSymbolicFactorization(adjacency, {0, 4}, fill);
        CHECK(symbolic->generation == generation + 1);
        REQUIRE(symbolic->lowerTriStructure == lower);
        REQUIRE(symbolic->upperTriStructure == upper);
        REQUIRE(symbolic->lowerTriStructureNew == lowerNew);
        REQUIRE(symbolic->upperTriStructureNew == upperNew);
        Direct::NativeBlockLU matrix(symbolic, 2);
        {
            auto write = matrix.Matrix().Write();
            const auto blocks = write.View();
            for (DNDS::index i = 0; i < 4; ++i)
            {
                blocks(i, 0) = Eigen::Matrix2d::Identity() * 4;
                for (DNDS::index j = 1; j < blocks.BlockNumber(i); ++j)
                    blocks(i, j) = -Eigen::Matrix2d::Identity();
            }
        }
        matrix.Factorize();
        Runtime::Array<DNDS::real> rhs(Runtime::RowLayout::Dense(4, 2)), forward(rhs.Layout()), output(rhs.Layout());
        {
            auto write = rhs.Write();
            for (DNDS::index i = 0; i < 4; ++i)
                for (int j = 0; j < 2; ++j)
                    write.View()(i, j) = 4 - DNDS::index(adjacency[i].size());
        }
        matrix.Solve(rhs, forward, output);
        const auto read = output.Read();
        for (DNDS::index i = 0; i < 4; ++i)
            for (int j = 0; j < 2; ++j)
                CHECK(read.View()(i, j) == doctest::Approx(1).epsilon(1e-13));
    }
}
TEST_CASE("Unreported symbolic construction is rank local")
{
    if (g_mpi.rank == 0)
    {
        auto symbolic = std::make_shared<Direct::SerialSymLUStructure>(g_mpi, 2);
        symbolic->ObtainSymmetricSymbolicFactorization(std::vector<std::vector<DNDS::index>>{{1}, {0}}, {0, 2}, 0, false);
        CHECK(symbolic->hasSymbolicPattern);
        CHECK(symbolic->lowerTriStructure[1] == std::vector<DNDS::index>{0});
    }
    // Other ranks never enter construction. Its optional diagnostics must not
    // introduce a hidden collective before this explicit test synchronization.
    MPI_Barrier(g_mpi.comm);
}
TEST_CASE("Native block matrix shares immutable row metadata and retains access lifetime")
{
    using Storage = DNDS::Algebra::BlockMatrixStorage;
    std::optional<Storage::ReadAccess> retained;
    {
        Storage storage({2, 0, 1}, 2);
        CHECK(storage.RowNumber() == 3);
        CHECK(storage.Offsets() == std::vector<DNDS::index>{0, 2, 2, 3});
        auto alias = storage;
        {
            auto write = storage.Write();
            const auto values = write.View();
            values(0, 0) << 1, 2, 3, 4;
            values(0, 1).setConstant(7);
            values(2, 0).setIdentity();
            CHECK_THROWS(alias.Read());
        }
        retained.emplace(alias.Read());
        CHECK_THROWS(storage.Write());
    }
    const auto values = retained->View();
    CHECK(values.BlockNumber(1) == 0);
    CHECK(values(0, 0)(0, 1) == 2);
    CHECK(values(0, 1).sum() == 28);
    CHECK(values(2, 0).trace() == 2);
    CHECK_THROWS(Storage({1}, 0));
    CHECK_THROWS(Storage({-1}, 2));
    Storage empty({}, 5);
    CHECK(empty.RowNumber() == 0);
}

TEST_CASE("Native block LU solves strided DOFs and rejects stale matrices and aliases")
{
    namespace R = DNDS::Runtime;
    for (int fill : {0, 1, -1})
    {
        auto symbolic = std::make_shared<Direct::SerialSymLUStructure>(g_mpi, NCells);
        symbolic->SetCoreFirstOrdering({0, 3, 7, 12});
        symbolic->ObtainSymmetricSymbolicFactorization(build2DPeriodicAdj(), {0, NCells}, fill);
        Direct::NativeBlockLU native(symbolic, BS);
        Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(NCells * BS, NCells * BS);
        {
            auto access = native.Matrix().Write();
            const auto values = access.View();
            for (DNDS::index i = 0; i < NCells; ++i)
            {
                Block diagonal;
                diagonal << 6 + .03 * i, .04, -.02, 7 + .01 * i;
                values(i, 0) = diagonal;
                dense.block<BS, BS>(i * BS, i * BS) = diagonal;
                auto fillBlocks = [&](const auto &neighbors, DNDS::index first)
                {
                    for (DNDS::index slot = 0; slot < DNDS::index(neighbors.size()); ++slot)
                    {
                        const auto other = neighbors[slot];
                        const auto adjacent = build2DPeriodicAdj()[i];
                        const bool original = std::find(adjacent.begin(), adjacent.end(), other) != adjacent.end();
                        Block coupling = Block::Zero();
                        if (original)
                            coupling << -.7, .01, -.015, -.8;
                        values(i, first + slot) = coupling;
                        dense.block<BS, BS>(i * BS, other * BS) = coupling;
                    }
                };
                fillBlocks(symbolic->lowerTriStructure[i], 1);
                fillBlocks(symbolic->upperTriStructure[i], 1 + DNDS::index(symbolic->lowerTriStructure[i].size()));
            }
        }
        native.Factorize({4, NCells});
        CHECK(native.FactorsCurrent());
        CHECK(native.LastInverseFallbacks() == 0);
        for (const auto &shape : {R::RowLayout::Dense(NCells, BS, BS + 3), R::RowLayout::ColMajor(NCells, BS, {8, 1}),
                                  R::RowLayout::TiledColMajor(NCells, BS, 32)})
        {
            R::Array<DNDS::real> rhs(shape), forward(shape), output(shape);
            Eigen::VectorXd b(NCells * BS);
            {
                auto write = rhs.Write();
                const auto view = write.View<R::UniformTableRows>();
                for (DNDS::index i = 0; i < NCells; ++i)
                    for (DNDS::index j = 0; j < BS; ++j)
                        view(i, j) = b(i * BS + j) = std::sin(.31 * i + .2 * j) + .4;
            }
            const auto forwardVersion = forward.Version();
            {
                auto retained = output.Read();
                CHECK_THROWS(native.Solve(rhs, forward, output));
                CHECK(forward.Version() == forwardVersion);
            }
            native.Solve(rhs, forward, output);
            Eigen::VectorXd x(NCells * BS);
            {
                auto read = output.Read();
                const auto view = read.View<R::UniformTableRows>();
                for (DNDS::index i = 0; i < NCells; ++i)
                    for (DNDS::index j = 0; j < BS; ++j)
                        x(i * BS + j) = view(i, j);
            }
            CHECK((dense * x - b).norm() < .2 * b.norm());
            if (fill == -1)
                CHECK((x - dense.partialPivLu().solve(b)).norm() < 2e-13);
            R::Array<DNDS::real> stagedForward(shape), stagedOutput(shape);
            {
                auto solve = native.BeginSolve(rhs, stagedForward, stagedOutput);
                CHECK_THROWS(solve.Upper(0, NCells));
                CHECK_THROWS(solve.Finish());
                CHECK_THROWS(solve.Lower(4, NCells));
                solve.Lower(0, 4);
                CHECK_THROWS(stagedForward.Read());
                CHECK_THROWS(stagedOutput.Read());
                CHECK_THROWS(solve.Lower(0, 4));
                CHECK_THROWS(solve.Upper(0, 4));
                solve.Lower(4, NCells);
                CHECK_THROWS(solve.Upper(0, 4));
                solve.Upper(4, NCells);
                solve.Upper(0, 4);
                CHECK_THROWS(stagedOutput.Read());
                solve.Finish();
                CHECK_THROWS(solve.Finish());
            }
            {
                auto read = stagedOutput.Read();
                for (DNDS::index i = 0; i < NCells; ++i)
                    for (DNDS::index j = 0; j < BS; ++j)
                        CHECK(read.View<R::UniformTableRows>()(i, j) == x(i * BS + j));
            }
            CHECK_THROWS(native.Solve(rhs, rhs, output));
            CHECK_THROWS(native.Solve(rhs, forward, forward));
            {
                auto writer = native.Matrix().Write();
                CHECK_FALSE(native.FactorsCurrent());
                CHECK_THROWS(native.Solve(rhs, forward, output));
            }
            CHECK_FALSE(native.FactorsCurrent());
            native.Factorize();
        }
        R::Array<DNDS::real> oldRhs(R::RowLayout::Dense(NCells, BS)), oldForward(oldRhs.Layout()), oldOutput(oldRhs.Layout());
        {
            auto solve = native.BeginSolve(oldRhs, oldForward, oldOutput);
            solve.Lower(0, 4);
            symbolic->ObtainSymmetricSymbolicFactorization(build2DPeriodicAdj(), {0, NCells}, fill);
            CHECK_THROWS(solve.Lower(4, NCells));
            CHECK_THROWS(solve.Finish());
        }
        CHECK_FALSE(oldForward.IsValid());
        CHECK_FALSE(oldOutput.IsValid());
        CHECK_FALSE(native.FactorsCurrent());
        CHECK_THROWS(native.Factorize());
    }
}

TEST_CASE("Native block LU supports empty owners and counts guarded inverse fallback")
{
    namespace R = DNDS::Runtime;
    auto empty = std::make_shared<Direct::SerialSymLUStructure>(g_mpi, 0);
    std::vector<std::vector<DNDS::index>> noRows;
    empty->ObtainSymmetricSymbolicFactorization(noRows, {0, 0}, 0);
    Direct::NativeBlockLU zero(empty, 5);
    zero.Factorize();
    R::Array<DNDS::real> rhs0(R::RowLayout::Dense(0, 5)), middle0(R::RowLayout::Dense(0, 5)), output0(R::RowLayout::Dense(0, 5));
    CHECK_NOTHROW(zero.Solve(rhs0, middle0, output0));
    for (const std::vector<DNDS::index> &ends : {std::vector<DNDS::index>{0}, std::vector<DNDS::index>{0, 0}})
    {
        CHECK_NOTHROW(zero.Factorize(ends));
        CHECK(zero.FactorsCurrent());
        CHECK_NOTHROW(zero.Solve(rhs0, middle0, output0));
    }

    auto symbolic = std::make_shared<Direct::SerialSymLUStructure>(g_mpi, 2);
    std::vector<std::vector<DNDS::index>> twoRows(2);
    symbolic->ObtainSymmetricSymbolicFactorization(twoRows, {0, 1, 2}, 0);
    Direct::NativeBlockLU nearSingular(symbolic, 2);
    {
        auto write = nearSingular.Matrix().Write();
        for (DNDS::index i = 0; i < 2; ++i)
            write.View()(i, 0) << 1, 1 - 1e-9, 1 - 1e-9, 1;
    }
    nearSingular.Factorize();
    CHECK(nearSingular.LastInverseFallbacks() == 2);
    R::Array<DNDS::real> rhs(R::RowLayout::Dense(2, 2)), middle(R::RowLayout::Dense(2, 2)), output(R::RowLayout::Dense(2, 2));
    {
        auto write = rhs.Write();
        for (DNDS::index i = 0; i < 2; ++i)
            write.View()(i, 0) = write.View()(i, 1) = 2;
    }
    nearSingular.Solve(rhs, middle, output);
    auto read = output.Read();
    for (DNDS::index i = 0; i < 2; ++i)
    {
        CHECK(read.View()(i, 0) == doctest::Approx(2 / (2 - 1e-9)).epsilon(1e-6));
        CHECK(read.View()(i, 1) == doctest::Approx(2 / (2 - 1e-9)).epsilon(1e-6));
    }
}
TEST_CASE("Native triangular phases retain owners and advance independent local parts")
{
    namespace R = DNDS::Runtime;
    auto coupled = std::make_shared<Direct::SerialSymLUStructure>(g_mpi, 2);
    coupled->ObtainSymmetricSymbolicFactorization(std::vector<std::vector<DNDS::index>>{{1}, {0}}, {0, 1, 2}, 0);
    CHECK_THROWS(Direct::NativeBlockLU(coupled, 2));
    auto symbolic = std::make_shared<Direct::SerialSymLUStructure>(g_mpi, 2);
    std::vector<std::vector<DNDS::index>> rows(2);
    symbolic->ObtainSymmetricSymbolicFactorization(rows, {0, 1, 2}, 0);
    R::Array<DNDS::real> output(R::RowLayout::Dense(2, 2));
    std::optional<Direct::NativeBlockLU::TriangularSolve> retained;
    {
        auto owner = std::make_unique<Direct::NativeBlockLU>(symbolic, 2);
        {
            auto write = owner->Matrix().Write();
            for (DNDS::index i = 0; i < 2; ++i)
                write.View()(i, 0) = Eigen::Matrix2d::Identity() * 2;
        }
        owner->Factorize();
        R::Array<DNDS::real> rhs(output.Layout()), forward(output.Layout());
        {
            auto write = rhs.Write();
            for (DNDS::index i = 0; i < 2; ++i)
                for (DNDS::index j = 0; j < 2; ++j)
                    write.View()(i, j) = i + j + 1;
        }
        retained.emplace(owner->BeginSolve(rhs, forward, output));
    }
    // Each local part is independent, so its complete solve can precede the
    // other part's forward scan. No per-row readiness bitmap is needed.
    retained->Lower(1, 2);
    retained->Upper(1, 2);
    CHECK_THROWS(retained->Finish());
    retained->Lower(0, 1);
    retained->Upper(0, 1);
    retained->Finish();
    const auto read = output.Read();
    for (DNDS::index i = 0; i < 2; ++i)
        for (DNDS::index j = 0; j < 2; ++j)
            CHECK(read.View()(i, j) == .5 * (i + j + 1));
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    g_mpi.setWorld();
    doctest::Context ctx;
    ctx.applyCommandLine(argc, argv);
    int res = ctx.run();
    MPI_Finalize();
    return res;
}
