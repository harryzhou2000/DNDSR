#include <iostream>
#include <cstdint>
#include <vector>
#include <fmt/core.h>
#include <cassert>

// g++ testMetis.cpp -o testMetis -I../external/Linux-x86_64/include -I../external/fmt-10.1.1/include -L../external/Linux-x86_64/lib -lmetis

namespace DNDS
{
    using index = int64_t;
}

namespace _METIS
{
#include "metis.h"

    static idx_t indexToIdx(DNDS::index v)
    {
        if constexpr (sizeof(DNDS::index) <= sizeof(idx_t))
            return v;
        else
            return v;
    }
}

int main()
{
    _METIS::idx_t nCell = _METIS::indexToIdx(100);
    _METIS::idx_t nCon{1}, options[METIS_NOPTIONS];
    _METIS::METIS_SetDefaultOptions(options);
    {
        options[_METIS::METIS_OPTION_CTYPE] = _METIS::METIS_CTYPE_RM;
        options[_METIS::METIS_OPTION_RTYPE] = _METIS::METIS_RTYPE_FM;
        options[_METIS::METIS_OPTION_IPTYPE] = _METIS::METIS_IPTYPE_EDGE;
        options[_METIS::METIS_OPTION_RTYPE] = _METIS::METIS_RTYPE_SEP1SIDED;
        options[_METIS::METIS_OPTION_NSEPS] = 1;
        options[_METIS::METIS_OPTION_NITER] = 10;
        options[_METIS::METIS_OPTION_UFACTOR] = 30;
        options[_METIS::METIS_OPTION_COMPRESS] = 0; // do not compress
        // options[_METIS::METIS_OPTION_CCORDER] = 0; //use default?
        options[_METIS::METIS_OPTION_SEED] = 0;    // ! seeding 0 for determined result
        options[_METIS::METIS_OPTION_PFACTOR] = 0; // not removing large vertices
        options[_METIS::METIS_OPTION_NUMBERING] = 0;
        options[_METIS::METIS_OPTION_DBGLVL] = _METIS::METIS_DBG_TIME | _METIS::METIS_DBG_IPART;
    }
    // std::vector<std::vector<index>> cell2cellFaceV = this->GetCell2CellFaceVLocal();
    std::vector<std::vector<DNDS::index>> cell2cellFaceV;
    {
        cell2cellFaceV.resize(nCell);
        for (DNDS::index iCell = 0; iCell < nCell; iCell++)
        {
            cell2cellFaceV[iCell].resize(2);
            cell2cellFaceV[iCell][0] = (nCell + iCell - 1) % nCell;
            cell2cellFaceV[iCell][1] = (iCell + 1) % nCell;
        }
    }
    _METIS::idx_t nSum;
    std::vector<_METIS::idx_t> adjncy, xadj, perm, iPerm;
    xadj.resize(nCell + 1);
    xadj[0] = 0;
    for (_METIS::idx_t iC = 0; iC < nCell; iC++)
        xadj[iC + 1] = xadj[iC] + cell2cellFaceV[iC].size(); //! check overflow!
    adjncy.resize(xadj.back());
    for (_METIS::idx_t iC = 0; iC < nCell; iC++)
        std::copy(cell2cellFaceV[iC].begin(), cell2cellFaceV[iC].end(), adjncy.begin() + xadj[iC]);
    perm.resize(nCell);
    iPerm.resize(nCell);
    std::cout << "nCell " << nCell << std::endl;
    for (auto v : xadj)
        std::cout << v << ", ";
    std::cout << std::endl;
    for (auto v : adjncy)
        std::cout << v << ", ";
    std::cout << std::endl;

    int ret = _METIS::METIS_NodeND(&nCell, xadj.data(), adjncy.data(), NULL, NULL, perm.data(), iPerm.data());
    // DNDS_assert_info(ret == _METIS::METIS_OK, fmt::format("Metis return not ok, [{}]", ret));
    std::cout << "here" << std::endl;
}