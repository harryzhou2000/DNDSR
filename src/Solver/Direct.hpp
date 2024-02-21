#pragma once

#ifndef __DNDS_REALLY_COMPILING__
#define __DNDS_REALLY_COMPILING__
#define __DNDS_REALLY_COMPILING__HEADER_ON__
#endif
#include "DNDS/JsonUtil.hpp"
#include "DNDS/Defines.hpp"
#ifdef __DNDS_REALLY_COMPILING__HEADER_ON__
#undef __DNDS_REALLY_COMPILING__
#endif

#include <unordered_set>

#define JSON_ASSERT DNDS_assert
#include <json.hpp>

namespace DNDS::Direct
{
    struct DirectPrecControl
    {
        bool useDirectPrec = false;
        int32_t iluCode = 0;              // 0 for no fill, -1 for complete
        int32_t orderingCode = INT32_MIN; // INT32_MIN for auto 0 for natural, 1 for metis, 2 for MMD
        DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON(
            DirectPrecControl,
            useDirectPrec,
            iluCode,
            orderingCode)
        int getOrderingCode() const
        {
            return orderingCode == INT32_MIN
                       ? (iluCode == -1 ? 2 : 0) // auto decide
                       : orderingCode;
        }
        int getILUCode() const { return iluCode; }
    };
}

namespace DNDS::Direct
{
    struct SerialSymLUStructure
    {
        MPIInfo mpi;
        index N;
        using tLocalMatStruct = std::vector<std::vector<index>>;
        tLocalMatStruct lowerTriStructure;
        tLocalMatStruct upperTriStructure;
        tLocalMatStruct lowerTriStructureNew;
        tLocalMatStruct upperTriStructureNew;
        tLocalMatStruct lowerTriStructureNewInUpper;
        tLocalMatStruct cell2cellFaceVLocal2FullRowPos; // diag-lower-upper
        std::vector<index> localFillOrderingOld2New;
        std::vector<index> localFillOrderingNew2Old;


        SerialSymLUStructure(const MPIInfo &nMpi, index nN) : mpi(nMpi), N(nN){};

        index Num() const { return N; }

        index FillingReorderOld2New(index v)
        {
            return localFillOrderingOld2New.size() ? localFillOrderingOld2New[v] : v;
        }
        index FillingReorderNew2Old(index v)
        {
            return localFillOrderingNew2Old.size() ? localFillOrderingNew2Old[v] : v;
        }

        template <class TAdj>
        void ObtainSymmetricSymbolicFactorization(
            const TAdj &cell2cellFaceV,
            int iluCode)
        {
            std::vector<std::unordered_set<index>> cell2cellFaceVEnlarged;
            if (iluCode > 0) // do expansion of stencil
            {
                DNDS_assert(iluCode <= 5);
                cell2cellFaceVEnlarged.resize(this->Num());
                for (index iCell = 0; iCell < this->Num(); iCell++)
                    for (auto iCO : cell2cellFaceV[iCell])
                        cell2cellFaceVEnlarged[iCell].insert(iCO);
                for (int iFill = 0; iFill < iluCode; iFill++)
                {
                    for (index iCell = 0; iCell < this->Num(); iCell++)
                    {
                        std::unordered_set<index> newNeighbor;
                        for (auto iCO : cell2cellFaceVEnlarged[iCell])
                            for (auto iCOO : cell2cellFaceV[iCO])
                                newNeighbor.insert(iCOO);
                        for (auto iCN : newNeighbor)
                            cell2cellFaceVEnlarged[iCell].insert(iCN);
                    }
                }
                for (index iCell = 0; iCell < this->Num(); iCell++)
                    cell2cellFaceVEnlarged[iCell].erase(iCell); // this is redundant
            }

            std::vector<std::unordered_set<index>> triLowRows;
            std::vector<std::unordered_set<index>> triUppRows;
            std::vector<std::unordered_set<index>> midSymMatCols;
            triLowRows.resize(this->Num());
            triUppRows.resize(this->Num());
            midSymMatCols.resize(this->Num());
            index nnzOrig{this->Num()};

            for (index iCellP = 0; iCellP < this->Num(); iCellP++) // iterate over the columns
            {
                index iCell = this->FillingReorderNew2Old(iCellP);
                midSymMatCols[iCellP].insert(iCellP);
                for (auto iCellOther : cell2cellFaceV[iCell])
                {
                    index iCellOtherP = this->FillingReorderOld2New(iCellOther);
                    midSymMatCols[iCellP].insert(iCellOtherP); // assuming cell2cellFaceV is symmetric
                }
                nnzOrig += cell2cellFaceV[iCell].size();
            }

            for (index iCellP = 0; iCellP < this->Num(); iCellP++) // iterate over the columns
            {
                for (auto iCellOtherP : midSymMatCols[iCellP]) // emulate the symmetric factorization A L1L2L3...LN D LNT...L1T
                    if (iCellOtherP > iCellP)
                    {
                        index iCellOther = this->FillingReorderNew2Old(iCellOtherP);
                        for (auto iCellOtherPP : midSymMatCols[iCellP])
                            if (iCellOtherPP > iCellOtherP)
                            {
                                index iCellOtherOther = this->FillingReorderNew2Old(iCellOtherPP);
                                // to be always symmetric
                                // control over here to get incomplete LU structure
                                if (iluCode < 0 || (iluCode > 0 && cell2cellFaceVEnlarged[iCellOther].count(iCellOtherOther)))
                                {
                                    midSymMatCols[iCellOtherPP].insert(iCellOtherP); // upper part
                                    midSymMatCols[iCellOtherP].insert(iCellOtherPP); // lower part
                                }
                            }
                        triLowRows[iCellOtherP].insert(iCellP); // iCellP is iCol, iCellOtherP is iRow
                        triUppRows[iCellP].insert(iCellOtherP);
                    }
            }

            lowerTriStructure.resize(this->Num());
            upperTriStructure.resize(this->Num());
            lowerTriStructureNew.resize(this->Num());
            upperTriStructureNew.resize(this->Num());
            index nnzLower{0}, nnzUpper{0};
            for (index iCellP = 0; iCellP < this->Num(); iCellP++)
            {
                index iCell = this->FillingReorderNew2Old(iCellP);
                lowerTriStructure[iCell].reserve(triLowRows[iCellP].size());
                upperTriStructure[iCell].reserve(triUppRows[iCellP].size());
                lowerTriStructureNew[iCellP].reserve(triLowRows[iCellP].size());
                upperTriStructureNew[iCellP].reserve(triUppRows[iCellP].size());
                for (auto iCellOtherP : triLowRows[iCellP])
                    lowerTriStructureNew[iCellP].push_back(iCellOtherP);
                for (auto iCellOtherP : triUppRows[iCellP])
                    upperTriStructureNew[iCellP].push_back(iCellOtherP);
                std::sort(lowerTriStructureNew[iCellP].begin(), lowerTriStructureNew[iCellP].end());
                std::sort(upperTriStructureNew[iCellP].begin(), upperTriStructureNew[iCellP].end());
                for (auto iCellOtherP : lowerTriStructureNew[iCellP])
                    lowerTriStructure[iCell].push_back(this->FillingReorderNew2Old(iCellOtherP));
                for (auto iCellOtherP : upperTriStructureNew[iCellP])
                    upperTriStructure[iCell].push_back(this->FillingReorderNew2Old(iCellOtherP));

                nnzLower += lowerTriStructure[iCell].size();
                nnzUpper += upperTriStructure[iCell].size();
                // the lowerTriStructure and upperTriStructure's col indices are corresponding to
                // those of in *New Rows
                // col indices in *New Rows are sorted
            }
            if (mpi.rank == 0)
                log() << "Direct::ObtainSymmetricSymbolicFactorization(): Factorizing Done" << std::endl;
            MPISerialDo(mpi, [&]()
                        { log() << fmt::format("  ({}, {}/{})", mpi.rank,
                                               nnzLower + nnzUpper + this->Num(),
                                               nnzOrig)
                                << std::flush; });
            MPI::Barrier(mpi.comm);
            if (mpi.rank == 0)
                log() << std::endl;
            DNDS_assert(nnzLower == nnzUpper);

            // pre-search
            lowerTriStructureNewInUpper.resize(lowerTriStructureNew.size());
            for (index iP = 0; iP < this->Num(); iP++)
            {
                auto &&lowerRow = lowerTriStructureNew[iP];
                lowerTriStructureNewInUpper[iP].resize(lowerRow.size());
                for (int ijP = 0; ijP < lowerRow.size(); ijP++)
                {
                    index jP = lowerRow[ijP];
                    auto &&upperRow = upperTriStructureNew[jP];
                    auto ret = std::lower_bound(upperRow.begin(), upperRow.end(), iP);
                    DNDS_assert(ret != upperRow.end()); // has to be found
                    lowerTriStructureNewInUpper[iP][ijP] = ret - upperRow.begin();
                }
            }

            cell2cellFaceVLocal2FullRowPos.resize(this->Num());
            for (index i = 0; i < this->Num(); i++)
            {
                index iP = this->FillingReorderOld2New(i);
                cell2cellFaceVLocal2FullRowPos[i].resize(cell2cellFaceV[i].size(), -1);
                for (int ic2c = 0; ic2c < cell2cellFaceV[i].size(); ic2c++)
                {
                    index j = cell2cellFaceV[i][ic2c];
                    index jP = this->FillingReorderOld2New(j);
                    if (jP < iP)
                    {
                        auto &&row = lowerTriStructureNew[iP];
                        auto ret = std::lower_bound(row.begin(), row.end(), jP);
                        DNDS_assert(ret != row.end());
                        cell2cellFaceVLocal2FullRowPos[i][ic2c] =
                            (ret - row.begin()) + 1;
                    }
                    else if (jP > iP)
                    {
                        auto &&row = upperTriStructureNew[iP];
                        auto ret = std::lower_bound(row.begin(), row.end(), jP);
                        DNDS_assert(ret != row.end());
                        cell2cellFaceVLocal2FullRowPos[i][ic2c] =
                            (ret - row.begin()) + lowerTriStructureNew[iP].size() + 1;
                    }
                }
            }
        }
    };
}