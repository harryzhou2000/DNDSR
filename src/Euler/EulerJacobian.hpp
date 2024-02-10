#pragma once

#include "Euler.hpp"
#include "DNDS/ArrayDerived/ArrayEigenUniMatrixBatch.hpp"
#include "Geom/Mesh.hpp"

namespace DNDS::Euler
{
    template <int nVarsFixed = 5>
    // static const int nVarsFixed = 5;
    struct JacobianLocalLU
    {
        using tLocalMat = ArrayEigenUniMatrixBatch<nVarsFixed, nVarsFixed>;
        using tComponent = Eigen::Matrix<real, nVarsFixed, nVarsFixed>;
        // using tIndices = Geom::UnstructuredMesh::tLocalMatStruct;
        /**
         * \brief
         * each Row:
         * 0: diag,
         * [1,1+mesh->lowerTriStructure[iCell].size()), lower
         * [1+mesh->lowerTriStructure[iCell].size(), end), upper
         */
        tLocalMat LDU; //
        ssp<Geom::UnstructuredMesh> mesh;

        bool isDecomposed = false;

        JacobianLocalLU(const ssp<Geom::UnstructuredMesh> &nMesh, int nVarsC) : mesh{nMesh}
        {
            DNDS_assert(mesh->lowerTriStructure.size() == mesh->NumCell());
            LDU.Resize(mesh->NumCell(), nVarsC, nVarsC);
            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            {
                LDU.ResizeRow(iCell,
                              1 + mesh->lowerTriStructure[iCell].size() +
                                  mesh->upperTriStructure[iCell].size());
                for (auto &v : LDU[iCell])
                    v.setZero();
            }
            LDU.Compress();
        }

        void setZero()
        {
            isDecomposed = false;
            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                for (auto &v : LDU[iCell])
                    v.setZero();
        }

        auto GetDiag(index i)
        {
            return LDU(i, 0);
        }
        auto GetLower(index i, int iInLow)
        {
            return LDU(i, 1 + iInLow);
        }
        auto GetUpper(index i, int iInUpp)
        {
            return LDU(i, 1 + mesh->lowerTriStructure[i].size() + iInUpp);
        }

        void InPlaceDecompose()
        {
            for (index iP = 0; iP < mesh->NumCell(); iP++)
            {
                index i = mesh->CellFillingReorderNew2Old(iP);

                auto &&lowerRow = mesh->lowerTriStructureNew[iP];
                /*********************/
                // lower part
                for (int ijP = 0; ijP < lowerRow.size(); ijP++)
                {
                    auto jP = lowerRow[ijP];
                    DNDS_assert(jP < iP);
                    auto j = mesh->CellFillingReorderNew2Old(jP);
                    // handle last j's job
                    if (ijP > 0)
                    {
                        auto &&lowerRowJP = mesh->lowerTriStructureNew[jP];
                        iterateIdentical(
                            lowerRow.begin(), lowerRow.end(), lowerRowJP.begin(), lowerRowJP.end(),
                            [&](index kP, auto pos1, auto pos2)
                            {
                                if (kP > jP - 1)
                                    return true;
                                DNDS_assert(kP < mesh->NumCell());
                                auto k = mesh->CellFillingReorderNew2Old(kP);
                                int jPInUpperPos = mesh->lowerTriStructureNewInUpper[jP][pos2];
                                this->GetLower(i, ijP) -= this->GetLower(i, pos1) * this->GetUpper(k, jPInUpperPos);
                                // if (iP < 3)
                                // log() << fmt::format("Lower Add at {},{},{} === {}", iP, jP - 1, kP, (this->GetLower(i, pos1) * this->GetUpper(k, jPInUpperPos))(0)) << std::endl;
                                return false;
                            });
                    }

                    // auto luDiag = this->GetDiag(j).fullPivLu();
                    // tComponent Aij = luDiag.solve(this->GetLower(i, ijP));
                    this->GetLower(i, ijP) *= this->GetDiag(j);
                }
                /*********************/
                // diag part
                for (int ikP = 0; ikP < lowerRow.size(); ikP++)
                {
                    auto kP = lowerRow[ikP];
                    auto k = mesh->CellFillingReorderNew2Old(kP);
                    int iPInUpperPos = mesh->lowerTriStructureNewInUpper[iP][ikP];
                    this->GetDiag(i) -= this->GetLower(i, ikP) * this->GetUpper(k, iPInUpperPos);
                }
                tComponent AI;
                // HardEigen::EigenLeastSquareInverse(this->GetDiag(i), , AI);
                auto luDiag = this->GetDiag(i).fullPivLu();
                DNDS_assert(luDiag.isInvertible());
                AI = luDiag.inverse();
                this->GetDiag(i) = AI;
                /*********************/
                // upper part
                auto &&upperRow = mesh->upperTriStructureNew[iP];
                for (int ijP = 0; ijP < upperRow.size(); ijP++)
                {
                    auto jP = upperRow[ijP];
                    DNDS_assert(jP > iP);
                    auto j = mesh->CellFillingReorderNew2Old(jP);
                    auto &&lowerRowJP = mesh->lowerTriStructureNew[jP];

                    iterateIdentical(
                        lowerRow.begin(), lowerRow.end(), lowerRowJP.begin(), lowerRowJP.end(),
                        [&](index kP, auto pos1, auto pos2)
                        {
                            if (kP >= iP)
                                return true;
                            DNDS_assert(kP < mesh->NumCell());
                            auto k = mesh->CellFillingReorderNew2Old(kP);
                            int jPInUpperPos = mesh->lowerTriStructureNewInUpper[jP][pos2];
                            this->GetUpper(i, ijP) -= this->GetLower(i, pos1) * this->GetUpper(k, jPInUpperPos);
                            // if (iP < 3)
                            // log() << fmt::format("Upper Add at {},{},{} === {}", iP, jP, kP,
                            //                      (this->GetLower(i, pos1) * this->GetUpper(k, jPInUpperPos))(0))
                            // << std::endl;
                            return false;
                        });
                }
            }
            isDecomposed = true;
        }

        void PrintLog()
        {
            log() << "nz Entries with Diag part inverse-ed" << std::endl;
            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            {
                log() << "=== Row " << iCell << std::endl
                      << std::setprecision(10);
                for (auto &v : LDU[iCell])
                    log() << v << std::endl
                          << std::endl;
            }
        }

        void MatMul(ArrayDOFV<nVarsFixed> &x, ArrayDOFV<nVarsFixed> &result)
        {
            DNDS_assert(!isDecomposed);
            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            {
                result[iCell] = this->GetDiag(iCell) * x[iCell];
                for (int ij = 0; ij < mesh->lowerTriStructure[iCell].size(); ij++)
                    result[iCell] += this->GetLower(iCell, ij) * x[mesh->lowerTriStructure[iCell][ij]];
                for (int ij = 0; ij < mesh->upperTriStructure[iCell].size(); ij++)
                    result[iCell] += this->GetUpper(iCell, ij) * x[mesh->upperTriStructure[iCell][ij]];
            }
        }

        void Solve(ArrayDOFV<nVarsFixed> &b, ArrayDOFV<nVarsFixed> &result)
        {
            DNDS_assert(isDecomposed);
            for (index iP = 0; iP < mesh->NumCell(); iP++)
            {
                index i = mesh->CellFillingReorderNew2Old(iP);
                result[i] = b[i];
                auto &&lowerRowOld = mesh->lowerTriStructure[i];
                for (int ij = 0; ij < lowerRowOld.size(); ij++)
                {
                    index j = lowerRowOld[ij];
                    result[i] -= this->GetLower(i, ij) * result[j];
                }
            }
            for (index iP = mesh->NumCell() - 1; iP >= 0; iP--)
            {
                index i = mesh->CellFillingReorderNew2Old(iP);
                auto &&upperRowOld = mesh->upperTriStructure[i];
                for (int ij = 0; ij < upperRowOld.size(); ij++)
                {
                    index j = upperRowOld[ij];
                    result[i] -= this->GetUpper(i, ij) * result[j];
                }
                // auto luDiag = this->GetDiag(i).fullPivLu();
                // result[i] = luDiag.solve(result[i]);
                result[i] = this->GetDiag(i) * result[i];
            }
        }
    };

}