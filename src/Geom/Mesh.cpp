#include "Mesh.hpp"

#include "cgnslib.h"
#include <cstdlib>
#include <string>
#include <map>
#include <set>
namespace _METIS
{
#include "metis.h"
#include "parmetis.h"
}

namespace Geom
{

    constexpr Elem::ElemType __getElemTypeFromCGNSType(ElementType_t cgns_et)
    {
        switch (cgns_et)
        {
        case BAR_2:
            return Elem::Line2;
        case BAR_3:
            return Elem::Line3;
        case TRI_3:
            return Elem::Tri3;
        case TRI_6:
            return Elem::Tri6;
        case QUAD_4:
            return Elem::Quad4;
        case QUAD_9:
            return Elem::Quad9;
        case TETRA_4:
            return Elem::Tet4;
        case TETRA_10:
            return Elem::Tet10;
        case HEXA_8:
            return Elem::Hex8;
        case HEXA_27:
            return Elem::Hex27;
        case PENTA_6:
            return Elem::Prism6;
        case PENTA_18:
            return Elem::Prism18;
        case PYRA_5:
            return Elem::Pyramid5;
        case PYRA_14:
            return Elem::Pyramid14;
        default:
            return Elem::UnknownElem;
        }
    }

    void UnstructuredMeshSerialRW::
        ReadFromCGNSSerial(const std::string &fName, const t_FBCName_2_ID &FBCName_2_ID)
    {
        int cgErr = CG_OK;

        DNDS_MAKE_SSP(cell2nodeSerial, mesh->mpi);
        DNDS_MAKE_SSP(bnd2nodeSerial, mesh->mpi);
        DNDS_MAKE_SSP(coordSerial, mesh->mpi);
        DNDS_MAKE_SSP(cellElemInfoSerial, ElemInfo::CommType(), ElemInfo::CommMult(), mesh->mpi);
        DNDS_MAKE_SSP(bndElemInfoSerial, ElemInfo::CommType(), ElemInfo::CommMult(), mesh->mpi);
        DNDS_MAKE_SSP(bnd2cellSerial, mesh->mpi);

        if (mRank != mesh->mpi.rank) //! parallel done!!! now serial!!!
            return;

        int cgns_file = -1;
        if (cg_open(fName.c_str(), CG_MODE_READ, &cgns_file))
            cg_error_exit();
        int n_bases = -1;
        if (cg_nbases(cgns_file, &n_bases))
            cg_error_exit();
        DNDS::log() << "CGNS === N bases: " << n_bases << std::endl;
        DNDS_assert(n_bases > 0);
        DNDS_assert(n_bases == 1);
        std::vector<std::pair<int, int>> Base_Zone;
        std::vector<std::string> BaseNames;
        std::vector<std::string> ZoneNames;
        std::vector<std::string> ZoneFamilyNames;
        std::vector<std::array<cgsize_t, 9>> ZoneSizes;
        std::vector<tCoord> ZoneCoords; //! purely serial
        std::vector<tAdj> ZoneElems;
        std::vector<tElemInfoArray> ZoneElemInfos;
        std::vector<std::vector<std::vector<cgsize_t>>> ZoneConnect;
        std::vector<std::vector<std::vector<cgsize_t>>> ZoneConnectDonor;
        std::vector<std::vector<int>> ZoneConnectTargetIZone;
        /***************************************************************************/
        // TRAVERSE 1
        for (int iBase = 1; iBase <= n_bases; iBase++)
        {

            int cgns_base = -1;
            char basename[48]{0};
            int celldim = -1;
            int physdim = -1;
            int nzones = -1;

            if (cg_base_read(cgns_file, iBase, basename, &celldim, &physdim))
                cg_error_exit();
            DNDS_assert(celldim == mesh->dim);
            if (cg_nzones(cgns_file, iBase, &nzones))
                cg_error_exit();
            for (int iZone = 1; iZone <= nzones; iZone++)
            {
                char zonename[48]{0};
                std::array<cgsize_t, 9> size{0, 0, 0, 0, 0, 0, 0, 0, 0};
                if (cg_zone_read(cgns_file, iBase, iZone, zonename, size.data()))
                    cg_error_exit();
                ZoneType_t zoneType;
                if (cg_zone_type(cgns_file, iBase, iZone, &zoneType))
                    cg_error_exit();
                DNDS_assert(zoneType == Unstructured); //! only supports unstructured
                Base_Zone.push_back(std::make_pair(iBase, iZone));
                BaseNames.push_back(basename);
                ZoneNames.push_back(zonename);

                if (cg_goto(cgns_file, iBase, "Zone_t", iZone, ""))
                    cg_error_exit();
                char famname[48]{0};
                if (cg_famname_read(famname))
                    cg_error_exit();
                ZoneFamilyNames.push_back(famname); //* family name used for Volume condition

                DNDS::index nNodes = size[0];
                DNDS::index nVols = size[1];
                DNDS::index nBVertex = size[2];

                //*** read coords
                {
                    int ncoords;
                    cgErr |= cg_ncoords(cgns_file, iBase, iZone, &ncoords);
                    // DNDS_assert(ncoords == mesh->dim); //!Even if has Z coord, we only use X,Y in 2d

                    ZoneCoords.emplace_back(std::make_shared<decltype(ZoneCoords)::value_type::element_type>());
                    ZoneCoords.back()->Resize(nNodes);
                    std::vector<double> coordsRead(nNodes);
                    cgsize_t RMin[3]{1, 0, 0};
                    cgsize_t RMax[3]{nNodes, 0, 0};
                    //* READ X
                    cgErr |= cg_coord_read(cgns_file, iBase, iZone, "CoordinateX", RealDouble, RMin, RMax, coordsRead.data());
                    for (DNDS::index i = 0; i < nNodes; i++)
                        ZoneCoords.back()->operator[](i)[0] = coordsRead.at(i);
                    //* READ Y
                    cgErr |= cg_coord_read(cgns_file, iBase, iZone, "CoordinateY", RealDouble, RMin, RMax, coordsRead.data());
                    for (DNDS::index i = 0; i < nNodes; i++)
                        ZoneCoords.back()->operator[](i)[1] = coordsRead.at(i);
                    //* READ Z
                    if (mesh->dim == 3)
                        cgErr |= cg_coord_read(cgns_file, iBase, iZone, "CoordinateX", RealDouble, RMin, RMax, coordsRead.data());
                    else
                        for (auto &i : coordsRead)
                            i = 0;
                    for (DNDS::index i = 0; i < nNodes; i++)
                        ZoneCoords.back()->operator[](i)[2] = coordsRead.at(i);

                    if (cgErr)
                        cg_error_exit();
                    DNDS::log() << "CGNS === Zone " << iZone << " Coords Reading Done" << std::endl;
                }
                // for (DNDS::index i = 0; i < ZoneCoords.back()->Size(); i++)
                //     std::cout << (*ZoneCoords.back())[i].transpose() << std::endl;

                //*** read Elems
                {
                    int nSections;
                    cgErr |= cg_nsections(cgns_file, iBase, iZone, &nSections);
                    DNDS_assert(nSections >= 1);
                    cgsize_t cstart = 0;
                    cgsize_t cend = 0;
                    cgsize_t maxend = 0;
                    //*Total size
                    for (int iSection = 1; iSection <= nSections; iSection++)
                    {
                        char sectionName[48];
                        ElementType_t etype;
                        cgsize_t start;
                        cgsize_t end;
                        int nBnd{0}, parentFlag{0};
                        cgErr |= cg_section_read(cgns_file, iBase, iZone, iSection, sectionName, &etype, &start, &end, &nBnd, &parentFlag);
                        // cgsize_t elemDataSize{0};
                        // cgErr |= cg_ElementDataSize(cgns_file, iBase, iZone, iSection, &elemDataSize);
                        DNDS_assert(cend == start - 1); //? testing
                        cstart = start;
                        cend = end;
                        maxend = std::max(end, maxend);
                    }
                    ZoneElems.emplace_back(std::make_shared<decltype(ZoneElems)::value_type::element_type>());
                    ZoneElems.back()->Resize(maxend);
                    ZoneElemInfos.emplace_back(std::make_shared<decltype(ZoneElemInfos)::value_type::element_type>());
                    ZoneElemInfos.back()->Resize(maxend);
                    //*Resize row
                    for (int iSection = 1; iSection <= nSections; iSection++)
                    {
                        char sectionName[48];
                        ElementType_t etype;
                        cgsize_t start;
                        cgsize_t end;
                        int nBnd{0}, parentFlag{0};
                        cgErr |= cg_section_read(cgns_file, iBase, iZone, iSection, sectionName, &etype, &start, &end, &nBnd, &parentFlag);
                        cgsize_t elemDataSize{0};
                        cgErr |= cg_ElementDataSize(cgns_file, iBase, iZone, iSection, &elemDataSize);
                        std::vector<cgsize_t> elemsRead(elemDataSize);

                        int nElemSec = end - start + 1;
                        if (etype == MIXED)
                        {
                            std::vector<cgsize_t> elemStarts(nElemSec + 1); //* note size
                            cgErr |= cg_poly_elements_read(cgns_file, iBase, iZone, iSection, elemsRead.data(), elemStarts.data(), NULL);
                            for (cgsize_t i = 0; i < nElemSec; i++)
                            {
                                ElementType_t c_etype = static_cast<ElementType_t>(elemsRead.at(elemStarts[i]));
                                if (__getElemTypeFromCGNSType(etype) != Elem::UnknownElem)
                                {
                                    DNDS::log() << "Error ETYPE " << std::to_string(etype) << std::endl;
                                    DNDS_assert_info(false, "Unsupported Element! ");
                                }
                                Elem::ElemType ct = __getElemTypeFromCGNSType(etype);
                                DNDS_assert_info(Elem::Element{ct}.GetNumNodes() + 1 == elemStarts[i + 1] - elemStarts[i],
                                                 "Element Node Number Mismatch!");
                                ZoneElems.back()->ResizeRow(start - 1 + i, Elem::Element{ct}.GetNumNodes());
                                for (t_index iNode = 0; iNode < Elem::Element{ct}.GetNumNodes(); iNode++)
                                    ZoneElems.back()->operator()(start - 1 + i, iNode) =
                                        elemsRead.at(elemStarts[i] + 1 + iNode) - 1; //! convert to 0 based; pointing to zonal index
                                ZoneElemInfos.back()->operator()(start - 1 + i, 0).setElemType(ct);
                                ZoneElemInfos.back()->operator()(start - 1 + i, 0).zone = -1 - iZone; //! bnd database here
                            }
                            /// @todo //TODO: TEST with actual data (MIXED TYPE) !!!!!!
                        }
                        else if (__getElemTypeFromCGNSType(etype) != Elem::UnknownElem)
                        {
                            Elem::ElemType ct = __getElemTypeFromCGNSType(etype);
                            cgErr |= cg_elements_read(cgns_file, iBase, iZone, iSection, elemsRead.data(), NULL);
                            DNDS_assert(elemDataSize / Elem::Element{ct}.GetNumNodes() == nElemSec);
                            for (cgsize_t i = 0; i < nElemSec; i++)
                            {

                                ZoneElems.back()->ResizeRow(start - 1 + i, Elem::Element{ct}.GetNumNodes());
                                for (t_index iNode = 0; iNode < Elem::Element{ct}.GetNumNodes(); iNode++)
                                    ZoneElems.back()->operator()(start - 1 + i, iNode) =
                                        elemsRead.at(Elem::Element{ct}.GetNumNodes() * i + iNode) - 1; //! convert to 0 based; pointing to zonal index
                                ZoneElemInfos.back()->operator()(start - 1 + i, 0).setElemType(ct);
                                ZoneElemInfos.back()->operator()(start - 1 + i, 0).zone = -1 - iZone; //! bnd database here
                            }
                        }
                        else
                        {
                            DNDS::log() << "Error ETYPE " << std::to_string(etype) << std::endl;
                            DNDS_assert_info(false, "Unsupported Element! ");
                        }
                    }

                    if (cgErr)
                        cg_error_exit();
                    DNDS::log() << "CGNS === Zone " << iZone << " Elems Reading Done" << std::endl;
                }

                //* read BCs
                {
                    int nBC;
                    cgErr |= cg_nbocos(cgns_file, iBase, iZone, &nBC);
                    for (int iBC = 1; iBC <= nBC; iBC++)
                    {
                        char boconame[48];
                        PointSetType_t pType;
                        cgsize_t nPts, normalListSize;
                        BCType_t bcType;
                        int NormalIndex[3];
                        DataType_t normalDataType;
                        int nDataset;
                        GridLocation_t gloc;
                        cgErr |= cg_boco_info(cgns_file, iBase, iZone, iBC, boconame, &bcType, &pType, &nPts, NormalIndex, &normalListSize, &normalDataType, &nDataset);
                        cgErr |= cg_boco_gridlocation_read(cgns_file, iBase, iZone, iBC, &gloc);
                        DNDS_assert_info(pType == PointRange, "Only Point Range supported in BC!");
                        if (mesh->dim == 2)
                            DNDS_assert(gloc == EdgeCenter);
                        if (mesh->dim == 3)
                            DNDS_assert(gloc == FaceCenter);
                        std::vector<cgsize_t> pts(nPts);
                        std::vector<double> normalBuf(normalListSize); // should have checked normalDataType, but it is not used here so not checked
                        cgErr |= cg_boco_read(cgns_file, iBase, iZone, iBC, pts.data(), normalBuf.data());
                        DNDS_assert(pts[0] >= 1 && pts[1] <= ZoneElems.back()->Size());

                        t_index BCCode = FBCName_2_ID(std::string(boconame));
                        if (BCCode == BC_TYPE_NULL)
                        {
                            DNDS::log() << "CGNS: [  " << boconame << "  ]     " << std::endl;
                            DNDS_assert_info(false, "BC NAME NOT FOUND IN DATABASE");
                        }
                        for (DNDS::index i = pts[0] - 1; i < pts[1]; i++)
                        {
                            ZoneElemInfos.back()->operator()(i, 0).zone = BCCode; //! setting BC code
                        }
                    }

                    if (cgErr)
                        cg_error_exit();
                    DNDS::log() << "CGNS === Zone " << iZone << " BCs Reading Done" << std::endl;
                }
            }
        }
        /***************************************************************************/

        /***************************************************************************/
        // TRAVERSE 2
        for (int iBase = 1; iBase <= n_bases; iBase++)
        {

            int cgns_base = -1;
            char basename[48]{0};
            int celldim = -1;
            int physdim = -1;
            int nzones = -1;

            if (cg_base_read(cgns_file, iBase, basename, &celldim, &physdim))
                cg_error_exit();
            DNDS_assert(celldim == mesh->dim);
            if (cg_nzones(cgns_file, iBase, &nzones))
                cg_error_exit();
            for (int iZone = 1; iZone <= nzones; iZone++)
            {
                char zonename[48]{0};
                std::array<cgsize_t, 9> size{0, 0, 0, 0, 0, 0, 0, 0, 0};
                if (cg_zone_read(cgns_file, iBase, iZone, zonename, size.data()))
                    cg_error_exit();
                ZoneType_t zoneType;
                if (cg_zone_type(cgns_file, iBase, iZone, &zoneType))
                    cg_error_exit();
                DNDS_assert(zoneType == Unstructured); //! only supports unstructured

                DNDS::index nNodes = size[0];
                DNDS::index nVols = size[1];
                DNDS::index nBVertex = size[2];

                //*** read Connectivity
                {
                    ZoneConnect.emplace_back();
                    ZoneConnectDonor.emplace_back();
                    ZoneConnectTargetIZone.emplace_back();
                    int nConns;
                    cgErr |= cg_nconns(cgns_file, iBase, iZone, &nConns);
                    for (int iConn = 1; iConn <= nConns; iConn++)
                    {
                        char connName[48], donorName[48];
                        GridLocation_t gLoc;
                        GridConnectivityType_t connType;
                        PointSetType_t ptType, ptType_donor;
                        cgsize_t npts, npts_donor;
                        ZoneType_t donorZT;
                        DataType_t donorDT;

                        cgErr |= cg_conn_info(cgns_file, iBase, iZone, iConn, connName, &gLoc, &connType, &ptType, &npts,
                                              donorName, &donorZT, &ptType_donor, &donorDT, &npts_donor);

                        DNDS_assert_info(connType == Abutting1to1, "Only support Abutting1to1 in connection!");
                        DNDS_assert_info(ptType == PointList, "Only Supports PointList in connection!");
                        DNDS_assert_info(ptType_donor == PointListDonor, "Only Supports PointListDonor in connection!");
                        DNDS_assert_info(donorZT == Unstructured, "Only Supports Unstructured in connection!");
                        DNDS_assert_info(gLoc == Vertex, "Only Supports Vertex in connection!");
                        DNDS_assert_info(npts_donor == npts, "Only Supports npts_donor == npts in connection!");
                        // std::vector<cgsize_t> ptSet(npts);
                        // std::vector<cgsize_t> ptSet_donor(npts);
                        ZoneConnectDonor.back().emplace_back(npts);
                        ZoneConnect.back().emplace_back(npts);
                        cgErr |= cg_conn_read(cgns_file, iBase, iZone, iConn, ZoneConnect.back().back().data(), donorDT, ZoneConnectDonor.back().back().data());
                        int iGZFound = -1;
                        for (int iGZ = 0; iGZ < ZoneNames.size(); iGZ++) // find the donor
                        {
                            if (Base_Zone.at(iGZ).first == iBase)
                            {
                                if (ZoneNames.at(iGZ) == donorName)
                                {
                                    iGZFound = iGZ;
                                }
                            }
                        }
                        DNDS_assert(iGZFound >= 0);
                        ZoneConnectTargetIZone.back().push_back(iGZFound);
                        DNDS::log() << "CGNS === Connection at Zone-Zone: " << iZone << " - " << iGZFound + 1 << std::endl;
                    }

                    if (cgErr)
                        cg_error_exit();
                }
            }
        }

        /***************************************************************************/

        /***************************************************************************/
        // ASSEMBLE
        int nZones = Base_Zone.size();
        std::vector<DNDS::index> ZoneNodeSizes(ZoneCoords.size());
        std::vector<DNDS::index> ZoneNodeStarts(ZoneCoords.size() + 1);
        ZoneNodeStarts[0] = 0;
        for (int i = 0; i < ZoneNodeSizes.size(); i++)
            ZoneNodeStarts[i + 1] = ZoneNodeStarts[i] + (ZoneNodeSizes[i] = ZoneCoords[i]->Size());

        // std::vector<tPoint> PointsFull(ZoneNodeStarts.back());

        std::vector<DNDS::index> NodeOld2New(ZoneNodeStarts.back(), -1);
        DNDS::index cTop = 0;
        for (int iGZ = 0; iGZ < ZoneNodeSizes.size(); iGZ++)
        {
            for (int iOther = 0; iOther < ZoneConnect.at(iGZ).size(); iOther++)
            {
                int iGZOther = ZoneConnectTargetIZone.at(iGZ).at(iOther);
                if (iGZOther < iGZ) // that has been set
                {
                    for (DNDS::index iNode = 0; iNode < ZoneConnect.at(iGZ).at(iOther).size(); iNode++)
                    {
                        NodeOld2New.at(ZoneNodeStarts.at(iGZ) + ZoneConnect.at(iGZ).at(iOther).at(iNode) - 1) //! note ZoneConnect is 1-based
                            = NodeOld2New.at(ZoneNodeStarts.at(iGZOther) + ZoneConnectDonor.at(iGZ).at(iOther).at(iNode) - 1);
                        DNDS_assert(NodeOld2New.at(ZoneNodeStarts.at(iGZ) + ZoneConnect.at(iGZ).at(iOther).at(iNode) - 1) >= 0);
                    }
                }
            }
            for (DNDS::index iNode = ZoneNodeStarts.at(iGZ); iNode < ZoneNodeStarts.at(iGZ + 1); iNode++)
            {
                if (NodeOld2New.at(iNode) < 0)
                {
                    NodeOld2New.at(iNode) = cTop;
                    cTop++;
                }
            }
        }
        DNDS::log() << "CGNS === Assembled Zones have NNodes " << cTop << std::endl;

        coordSerial->Resize(cTop);
        for (DNDS::index i = 0; i < coordSerial->Size(); i++)
            coordSerial->operator[](i).setConstant(DNDS::UnInitReal);

        for (int iGZ = 0; iGZ < ZoneNodeSizes.size(); iGZ++)
        {
            for (DNDS::index iNode = ZoneNodeStarts.at(iGZ); iNode < ZoneNodeStarts.at(iGZ + 1); iNode++)
            {
                auto coordC = (*ZoneCoords[iGZ])[iNode - ZoneNodeStarts.at(iGZ)];
                if (!DNDS::IsUnInitReal((*coordSerial)[NodeOld2New.at(iNode)][0]))
                    DNDS_assert_info(((*coordSerial)[NodeOld2New.at(iNode)] - coordC).norm() < 1e-15, "Not same points on the connection");
                (*coordSerial)[NodeOld2New.at(iNode)] = coordC;
            }
        }

        DNDS::index nVolElem = 0;
        DNDS::index nBndElem = 0;
        for (int iGZ = 0; iGZ < ZoneNodeSizes.size(); iGZ++)
        {
            for (DNDS::index iElem = 0; iElem < ZoneElems[iGZ]->Size(); iElem++)
            {
                for (DNDS::rowsize j = 0; j < ZoneElems[iGZ]->RowSize(iElem); j++)
                    ZoneElems[iGZ]->operator()(iElem, j) = NodeOld2New[ZoneNodeStarts[iGZ] + ZoneElems[iGZ]->operator()(iElem, j)];
                //* Convert to assembled index
                auto Elem = Elem::Element{ZoneElemInfos[iGZ]->operator()(iElem, 0).getElemType()};
                if (Elem.GetDim() == mesh->dim)
                    nVolElem++;
                if (Elem.GetDim() == mesh->dim - 1 && ZoneElemInfos[iGZ]->operator()(iElem, 0).zone > 0)
                    nBndElem++;
            }
        }

        cell2nodeSerial->Resize(nVolElem);
        bnd2nodeSerial->Resize(nBndElem);
        cellElemInfoSerial->Resize(nVolElem);
        bndElemInfoSerial->Resize(nBndElem);
        nVolElem = 0;
        nBndElem = 0;
        for (int iGZ = 0; iGZ < ZoneNodeSizes.size(); iGZ++)
        {
            for (DNDS::index iElem = 0; iElem < ZoneElems[iGZ]->Size(); iElem++)
            {
                auto Elem = Elem::Element{ZoneElemInfos[iGZ]->operator()(iElem, 0).getElemType()};
                if (Elem.GetDim() == mesh->dim)
                {
                    cell2nodeSerial->ResizeRow(nVolElem, ZoneElems[iGZ]->RowSize(iElem));
                    for (DNDS::rowsize j = 0; j < ZoneElems[iGZ]->RowSize(iElem); j++)
                        cell2nodeSerial->operator()(nVolElem, j) = ZoneElems[iGZ]->operator()(iElem, j);
                    cellElemInfoSerial->operator()(nVolElem, 0) = ZoneElemInfos[iGZ]->operator()(iElem, 0);
                    nVolElem++;
                }
                if (Elem.GetDim() == mesh->dim - 1 && ZoneElemInfos[iGZ]->operator()(iElem, 0).zone > 0)
                {
                    bnd2nodeSerial->ResizeRow(nBndElem, ZoneElems[iGZ]->RowSize(iElem));
                    for (DNDS::rowsize j = 0; j < ZoneElems[iGZ]->RowSize(iElem); j++)
                        bnd2nodeSerial->operator()(nBndElem, j) = ZoneElems[iGZ]->operator()(iElem, j);
                    bndElemInfoSerial->operator()(nBndElem, 0) = ZoneElemInfos[iGZ]->operator()(iElem, 0);
                    nBndElem++;
                }
            }
        }
        std::cout << "CGNS === Vol Elem [  " << nVolElem << "  ]"
                  << ", "
                  << " Bnd Elem [  " << nBndElem << "  ]" << std::endl;

        bnd2nodeSerial->Compress();
        cell2nodeSerial->Compress();
        /***************************************************************************/

        /***************************************************************************/
        // Get partial inverse info: bnd2cell

        bnd2cellSerial->Resize(bnd2nodeSerial->Size());
        for (DNDS::index iB = 0; iB < bnd2cellSerial->Size(); iB++)
            (*bnd2cellSerial)[iB][0] = (*bnd2cellSerial)[iB][1] = DNDS::UnInitIndex;
        // get node 2 bnd (for linear complexity)
        std::vector<std::vector<DNDS::index>> node2bnd(coordSerial->Size());
        std::vector<DNDS::rowsize> node2bndSiz(coordSerial->Size(), 0);
        for (DNDS::index iBFace = 0; iBFace < bnd2nodeSerial->Size(); iBFace++)
            for (DNDS::rowsize iN = 0; iN < Elem::Element{(*bndElemInfoSerial)(iBFace, 0).getElemType()}.GetNumVertices(); iN++)
                node2bndSiz[(*bnd2nodeSerial)(iBFace, iN)]++;
        for (DNDS::index iNode = 0; iNode < node2bnd.size(); iNode++)
            node2bnd[iNode].reserve(node2bndSiz[iNode]);
        for (DNDS::index iBFace = 0; iBFace < bnd2nodeSerial->Size(); iBFace++)
            for (DNDS::rowsize iN = 0; iN < Elem::Element{(*bndElemInfoSerial)(iBFace, 0).getElemType()}.GetNumVertices(); iN++)
                node2bnd[(*bnd2nodeSerial)(iBFace, iN)].push_back(iBFace); // Note that only primary vertices are included for computational cost

        // search cell 2 bnd
        for (DNDS::index iCell = 0; iCell < cell2nodeSerial->Size(); iCell++)
        {
            auto CellElem = Elem::Element{(*cellElemInfoSerial)[iCell]->getElemType()};
            std::vector<DNDS::index> cell2nodeRow{(*cell2nodeSerial)[iCell].begin(), (*cell2nodeSerial)[iCell].begin() + CellElem.GetNumVertices()};
            std::sort(cell2nodeRow.begin(), cell2nodeRow.end());
            // Note that only primary vertices are included for computational cost
            for (auto iNode : cell2nodeRow)
            {
                for (auto iB : node2bnd[iNode])
                {
                    auto BndElem = Elem::Element{(*bndElemInfoSerial)(iB, 0).getElemType()};
                    std::vector<DNDS::index> bnd2nodeRow{(*bnd2nodeSerial)[iB].begin(), (*bnd2nodeSerial)[iB].begin() + BndElem.GetNumVertices()};
                    std::sort(bnd2nodeRow.begin(), bnd2nodeRow.end());
                    if (std::includes(cell2nodeRow.begin(), cell2nodeRow.end(), bnd2nodeRow.begin(), bnd2nodeRow.end()))
                    {
                        // found as bnd
                        DNDS_assert_info(
                            (*bnd2cellSerial)[iB][0] == DNDS::UnInitIndex || (*bnd2cellSerial)[iB][0] == iCell, "bnd2cell not untouched!");
                        // DNDS_assert((*bnd2cellSerial)[iB][0] == DNDS::UnInitIndex);
                        (*bnd2cellSerial)[iB][0] = iCell;
                    }
                }
            }
        }
        for (DNDS::index iB = 0; iB < bnd2cellSerial->Size(); iB++)
            DNDS_assert((*bnd2cellSerial)[iB][0] != DNDS::UnInitIndex); // check all bnds have a cell

        cg_close(cgns_file);

        std::cout << "CGNS === Serial Read Done" << std::endl;
        // Memory with DM240-120 here: 18G ; after deconstruction done: 7.5G
    }

    // void UnstructuredMeshSerialRW::InterpolateTopology() //!could be useful for parallel?
    // {
    //     // count node 2 face
    //     DNDS_MAKE_SSP(cell2faceSerial, mesh->mpi);
    //     DNDS_MAKE_SSP(face2cellSerial, mesh->mpi);
    //     DNDS_MAKE_SSP(face2nodeSerial, mesh->mpi);
    //     DNDS_MAKE_SSP(faceElemInfoSerial, ElemInfo::CommType(), ElemInfo::CommMult(), mesh->mpi);

    //     if (mRank != mesh->mpi.rank)
    //         return;

    //     for (DNDS::index iCell = 0; iCell <cell2nodeSerial->Size(); iCell++)
    //     {

    //         // TODO
    //     }
    // }

    void UnstructuredMeshSerialRW::
        BuildCell2Cell()
    {
        DNDS_MAKE_SSP(cell2cellSerial, mesh->mpi);
        // if (mRank != mesh->mpi.rank)
        //     return;

        /// TODO: abstract these: invert cone (like node 2 cell -> cell 2 node) (also support operating on pair)
        /**********************************************************************************************************************/
        // getting node2cell // only primary vertices
        std::vector<std::vector<DNDS::index>> node2cell(coordSerial->Size());
        std::vector<DNDS::rowsize> node2cellSz(coordSerial->Size(), 0);
        for (DNDS::index iCell = 0; iCell < cell2nodeSerial->Size(); iCell++)
            for (DNDS::rowsize iN = 0; iN < Elem::Element{(*cellElemInfoSerial)(iCell, 0).getElemType()}.GetNumVertices(); iN++)
                node2cellSz[(*cell2nodeSerial)(iCell, iN)]++;
        for (DNDS::index iNode = 0; iNode < node2cell.size(); iNode++)
            node2cell[iNode].reserve(node2cellSz[iNode]);
        for (DNDS::index iCell = 0; iCell < cell2nodeSerial->Size(); iCell++)
            for (DNDS::rowsize iN = 0; iN < Elem::Element{(*cellElemInfoSerial)(iCell, 0).getElemType()}.GetNumVertices(); iN++)
                node2cell[(*cell2nodeSerial)(iCell, iN)].push_back(iCell);
        node2cellSz.clear();
        /**********************************************************************************************************************/

        cell2cellSerial->Resize(cell2nodeSerial->Size());
        for (DNDS::index iCell = 0; iCell < cell2nodeSerial->Size(); iCell++)
        {
            auto CellElem = Elem::Element{(*cellElemInfoSerial)[iCell]->getElemType()};
            std::vector<DNDS::index> cell2nodeRow{(*cell2nodeSerial)[iCell].begin(), (*cell2nodeSerial)[iCell].begin() + CellElem.GetNumVertices()};
            // only primary vertices
            std::set<DNDS::index> c_neighbors; // could optimize?
            for (auto iNode : cell2nodeRow)
                for (auto iCellOther : node2cell[iNode])
                    c_neighbors.insert(iCellOther);
            if (!c_neighbors.erase(iCell))
                DNDS_assert_info(false, "neighbors should include myself now");
            cell2cellSerial->ResizeRow(iCell, c_neighbors.size());
            DNDS::rowsize ic2c = 0;
            for (auto iCellOther : c_neighbors)
                (*cell2cellSerial)(iCell, ic2c++) = iCellOther;
        }
    }

    void UnstructuredMeshSerialRW::
        MeshPartitionCell2Cell()
    {
        //! preset hyper config, should be optional in the future
        bool isSerial = true;
        _METIS::idx_t nPart = mesh->mpi.size;
        cnPart = nPart;

        //! assuming all adj point to local numbers now
        // * Tend to local-global issues putting into
        cell2cellSerial->Compress();
        cell2cellSerial->AssertConsistent();
        cell2cellSerial->createGlobalMapping();

        std::vector<_METIS::idx_t> vtxdist(mesh->mpi.size + 1);
        for (DNDS::MPI_int r = 0; r <= mesh->mpi.size; r++)
            vtxdist[r] = cell2cellSerial->pLGlobalMapping->ROffsets().at(r); //! warning: no check overflow
        std::vector<_METIS::idx_t> xadj(cell2cellSerial->Size() + 1);
        for (DNDS::index iCell = 0; iCell < xadj.size(); iCell++)
            xadj[iCell] = (cell2cellSerial->rowPtr(iCell) - cell2cellSerial->rowPtr(0)); //! warning: no check overflow
        std::vector<_METIS::idx_t> adjncy(xadj.back());
        DNDS_assert(cell2cellSerial->DataSize() == xadj.back());
        for (DNDS::index iAdj = 0; iAdj < xadj.back(); iAdj++)
            adjncy[iAdj] = cell2cellSerial->data()[iAdj]; //! warning: no check overflow
        if (adjncy.size() == 0)
            adjncy.resize(1, -1); //*coping with zero sized data

        _METIS::idx_t nCell = cell2cellSerial->Size(); //! warning: no check overflow
        _METIS::idx_t nCon{1}, options[METIS_NOPTIONS];

        {
            options[_METIS::METIS_OPTION_OBJTYPE] = _METIS::METIS_OBJTYPE_CUT;
            options[_METIS::METIS_OPTION_CTYPE] = _METIS::METIS_CTYPE_RM;
            options[_METIS::METIS_OPTION_IPTYPE] = _METIS::METIS_IPTYPE_GROW;
            options[_METIS::METIS_OPTION_RTYPE] = _METIS::METIS_RTYPE_FM;
            // options[METIS_OPTION_NO2HOP] = 0; // only available in metis 5.1.0
            options[_METIS::METIS_OPTION_NCUTS] = 1;
            options[_METIS::METIS_OPTION_NITER] = 10;
            options[_METIS::METIS_OPTION_UFACTOR] = 30;
            options[_METIS::METIS_OPTION_MINCONN] = 0;
            options[_METIS::METIS_OPTION_CONTIG] = 1; // ! forcing contigious partition now ? necessary?
            options[_METIS::METIS_OPTION_SEED] = 0;   // ! seeding 0 for determined result
            options[_METIS::METIS_OPTION_NUMBERING] = 0;
            options[_METIS::METIS_OPTION_DBGLVL] = _METIS::METIS_DBG_TIME | _METIS::METIS_DBG_IPART;
        }
        std::vector<_METIS::idx_t> partOut(nCell);
        if (nCell == 0)
            partOut.resize(1, -1); //*coping with zero sized data
        if (nPart > 1)
        {
            if (mesh->mpi.size == 1 || (isSerial && mesh->mpi.rank == mRank))
            {
                _METIS::idx_t objval;
                int ret = _METIS::METIS_PartGraphKway(
                    &nCell, &nCon, xadj.data(), adjncy.data(), NULL, NULL, NULL,
                    &nPart, NULL, NULL, options, &objval, partOut.data());
                if (ret != _METIS::METIS_OK)
                {
                    DNDS::log() << "METIS returned not OK: [" << ret << "]" << std::endl;
                    DNDS_assert(false);
                }
            }
            else if (mesh->mpi.size != 1 && (!isSerial))
            {
                ///@todo //TODO: parmetis needs testing!
                for (int i = 0; i < vtxdist.size() - 1; i++)
                    DNDS_assert_info(vtxdist[i + 1] - vtxdist[i] > 0, "need more than zero cells on each proc!");
                std::vector<_METIS::real_t> tpWeights(nPart * nCon, 1.0 / nPart); //! assuming homogenous
                _METIS::real_t ubVec[1]{1.05};
                DNDS_assert(nCon == 1);
                _METIS::idx_t optsC[3];
                _METIS::idx_t wgtflag{0}, numflag{0};
                optsC[0] = 1;
                optsC[1] = 1;
                optsC[2] = 0;
                _METIS::idx_t objval;
                int ret = _METIS::ParMETIS_V3_PartKway(
                    vtxdist.data(), xadj.data(), adjncy.data(), NULL, NULL, &wgtflag, &numflag,
                    &nCon, &nPart, tpWeights.data(), ubVec, optsC, &objval, partOut.data(),
                    &mesh->mpi.comm);
                if (ret != _METIS::METIS_OK)
                {
                    DNDS::log() << "METIS returned not OK: [" << ret << "]" << std::endl;
                    DNDS_assert(false);
                }
            }
        }
        else
        {
            partOut.assign(partOut.size(), 0);
        }
        cellPartition.resize(cell2cellSerial->Size());
        for (DNDS::index i = 0; i < cellPartition.size(); i++)
            cellPartition[i] = partOut[i];
    }

    /*******************************************************************************************************************/
    /*******************************************************************************************************************/
    /*******************************************************************************************************************/

    /**
     * Used for generating pushing data structure
     *  @todo: //TODO test on parallel re-distributing
     */
    // for one-shot usage, partition data corresponds to mpi

    template <class TPartitionIdx>
    void Partition2LocalIdx(
        const std::vector<TPartitionIdx> &partition,
        std::vector<DNDS::index> &localPush,
        std::vector<DNDS::index> &localPushStart, const DNDS::MPIInfo &mpi)
    {
        // localPushStart.resize(mpi.size);
        std::vector<DNDS::index> localPushSizes(mpi.size, 0);
        for (auto r : partition)
        {
            DNDS_assert(r < mpi.size);
            localPushSizes[r]++;
        }
        DNDS::AccumulateRowSize(localPushSizes, localPushStart);
        localPush.resize(localPushStart[mpi.size]);
        localPushSizes.assign(mpi.size, 0);
        DNDS_assert(partition.size() == localPush.size());
        for (DNDS::index i = 0; i < partition.size(); i++)
            localPush[localPushStart[partition[i]] + (localPushSizes[partition[i]]++)] = i;
    }

    /**
     * Serial2Global is used for converting adj data to point to reordered global
     *  @todo: //TODO test on parallel re-distributing
     */
    template <class TPartitionIdx>
    void Partition2Serial2Global(
        const std::vector<TPartitionIdx> &partition,
        std::vector<DNDS::index> &serial2Global, const DNDS::MPIInfo &mpi, DNDS::MPI_int nPart)
    {
        serial2Global.resize(partition.size());
        /****************************************/
        std::vector<DNDS::index> numberAtLocal(nPart, 0);
        for (auto r : partition)
            numberAtLocal[r]++;
        std::vector<DNDS::index> numberTotal(nPart), numberPrev(nPart);
        MPI_Allreduce(numberAtLocal.data(), numberTotal.data(), nPart, DNDS::DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
        MPI_Scan(numberAtLocal.data(), numberPrev.data(), nPart, DNDS::DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
        std::vector<DNDS::index> numberTotalPlc(nPart + 1);
        numberTotalPlc[0] = 0;
        for (DNDS::MPI_int r = 0; r < nPart; r++)
            numberTotalPlc[r + 1] = numberTotalPlc[r] + numberTotal[r], numberPrev[r] -= numberAtLocal[r];
        // 2 things here: accumulate total and subtract local from prev
        /****************************************/
        numberAtLocal.assign(numberAtLocal.size(), 0);
        DNDS::index iFill = 0;
        for (auto r : partition)
            serial2Global[iFill++] = (numberAtLocal[r]++) + numberTotalPlc[r] + numberPrev[r];
    }

    /**
     * In this section, Serial means un-reordered index, i-e
     * pointing to the *serial data, cell2node then pointing to the
     * cell2node global indices:
     * 0 1; 2 3
     *
     * Global means the re-ordered data's global indices
     * if partition ==
     * 0 1; 0 1
     * then Serial 2 Global is:
     * 0 2; 1 3
     *
     * if in proc 0, a cell refers to node 2, then must be seen in JSG ghost, gets global output == 1
     *
     * comm complexity: same as data comm
     * @todo: //TODO test on parallel re-distributing
     */

    template <class TAdj = tAdj1>
    void ConvertAdjSerial2Global(TAdj &arraySerialAdj,
                                 const std::vector<DNDS::index> &partitionJSerial2Global,
                                 const DNDS::MPIInfo &mpi)
    {
        // IndexArray JSG(IndexArray::tContext(partitionJSerial2Global.size()), mpi);
        // forEachInArray(
        //     JSG, [&](IndexArray::tComponent &e, index i)
        //     { e[0] = partitionJSerial2Global[i]; });
        // JSGGhost.createGlobalMapping();
        tAdj1 JSG, JSGGhost;
        DNDS_MAKE_SSP(JSG, mpi);
        DNDS_MAKE_SSP(JSGGhost, mpi);
        JSG->Resize(partitionJSerial2Global.size());
        for (DNDS::index i = 0; i < JSG->Size(); i++)
            (*JSG)(i, 0) = partitionJSerial2Global[i];
        JSG->createGlobalMapping();
        std::vector<DNDS::index> ghostJSerialQuery;

        // get ghost
        DNDS::index nGhost = 0;
        for (DNDS::index i = 0; i < arraySerialAdj->Size(); i++)
        {
            for (DNDS::rowsize j = 0; j < arraySerialAdj->RowSize(i); j++)
            {
                DNDS::index v = (*arraySerialAdj)(i, j);
                if (v == DNDS::UnInitIndex)
                    break;
                DNDS::MPI_int rank = -1;
                DNDS::index val = -1;
                if (!JSG->pLGlobalMapping->search(v, rank, val))
                    DNDS_assert_info(false, "search failed");
                if (rank != mpi.rank) //! excluding self
                    nGhost++;
            }
        }
        ghostJSerialQuery.reserve(nGhost);
        for (DNDS::index i = 0; i < arraySerialAdj->Size(); i++)
        {
            for (DNDS::rowsize j = 0; j < arraySerialAdj->RowSize(i); j++)
            {
                DNDS::index v = (*arraySerialAdj)(i, j);
                if (v == DNDS::UnInitIndex)
                    break;
                DNDS::MPI_int rank = -1;
                DNDS::index val = -1;
                JSG->pLGlobalMapping->search(v, rank, val);
                if (rank != mpi.rank) //! excluding self
                    ghostJSerialQuery.push_back(v);
            }
        }
        // PrintVec(ghostJSerialQuery, std::cout);
        typename DNDS::ArrayTransformerType<tAdj1::element_type>::Type JSGTrans;
        JSGTrans.setFatherSon(JSG, JSGGhost);
        JSGTrans.createGhostMapping(ghostJSerialQuery);
        JSGTrans.createMPITypes();
        JSGTrans.pullOnce();

        for (DNDS::index i = 0; i < arraySerialAdj->Size(); i++)
        {
            for (DNDS::rowsize j = 0; j < arraySerialAdj->RowSize(i); j++)
            {
                DNDS::index &v = (*arraySerialAdj)(i, j);
                if (v == DNDS::UnInitIndex)
                    break;
                DNDS::MPI_int rank = -1;
                DNDS::index val = -1;
                if (!JSGTrans.pLGhostMapping->search(v, rank, val))
                    DNDS_assert_info(false, "search failed");
                if (rank == -1)
                    v = (*JSG)(val, 0);
                else
                    v = (*JSGGhost)(val, 0);
            }
        }
    }

    template <class TArr = tAdj1>
    void TransferDataSerial2Global(TArr &arraySerial,
                                   TArr &arrayDist,
                                   const std::vector<DNDS::index> &pushIndex,
                                   const std::vector<DNDS::index> &pushIndexStart,
                                   const DNDS::MPIInfo &mpi)
    {
        typename DNDS::ArrayTransformerType<typename TArr::element_type>::Type trans;
        trans.setFatherSon(arraySerial, arrayDist);
        trans.createFatherGlobalMapping();
        trans.createGhostMapping(pushIndex, pushIndexStart);
        trans.createMPITypes();
        trans.pullOnce();
    }

    //! inefficient, use Partition2Serial2Global ! only used for convenient comparison
    void PushInfo2Serial2Global(std::vector<DNDS::index> &serial2Global,
                                DNDS::index localSize,
                                const std::vector<DNDS::index> &pushIndex,
                                const std::vector<DNDS::index> &pushIndexStart,
                                const DNDS::MPIInfo &mpi)
    {
        tIndPair Serial2Global;
        DNDS_MAKE_SSP(Serial2Global.father, mpi);
        DNDS_MAKE_SSP(Serial2Global.son, mpi);
        Serial2Global.father->Resize(localSize);
        Serial2Global.TransAttach();
        Serial2Global.trans.createFatherGlobalMapping();
        Serial2Global.trans.createGhostMapping(pushIndex, pushIndexStart);
        Serial2Global.trans.createMPITypes();
        Serial2Global.son->createGlobalMapping();
        // Set son to son's global
        for (DNDS::index iSon = 0; iSon < Serial2Global.son->Size(); iSon++)
            (*Serial2Global.son)[iSon] = Serial2Global.son->pLGlobalMapping->operator()(mpi.rank, iSon);
        Serial2Global.trans.pushOnce();
        serial2Global.resize(localSize);
        for (DNDS::index iFat = 0; iFat < Serial2Global.father->Size(); iFat++)
            serial2Global[iFat] = Serial2Global.father->operator[](iFat);
    }

    // template <class TAdj = tAdj1>
    // void ConvertAdjSerial2Global(TAdj &arraySerialAdj,
    //                              const std::vector<DNDS::index> &partitionJSerial2Global,
    //                              const DNDS::MPIInfo &mpi)
    // {
    // }
    /*******************************************************************************************************************/
    /*******************************************************************************************************************/
    /*******************************************************************************************************************/

    void UnstructuredMeshSerialRW::
        PartitionReorderToMeshCell2Cell()
    {
        DNDS_assert(cnPart == mesh->mpi.size);
        // * 1: get the nodal partition
        nodePartition.resize(coordSerial->Size(), static_cast<DNDS::MPI_int>(INT32_MAX));
        for (DNDS::index iCell = 0; iCell < cell2nodeSerial->Size(); iCell++)
            for (DNDS::rowsize ic2n = 0; ic2n < (*cell2nodeSerial).RowSize(iCell); ic2n++)
                nodePartition[(*cell2nodeSerial)(iCell, ic2n)] = std::min(nodePartition[(*cell2nodeSerial)(iCell, ic2n)], cellPartition.at(iCell));
        // * 1: get the bnd partition
        bndPartition.resize(bnd2cellSerial->Size());
        for (DNDS::index iBnd = 0; iBnd < bnd2cellSerial->Size(); iBnd++)
            bndPartition[iBnd] = cellPartition[(*bnd2cellSerial)(iBnd, 0)];

        std::vector<DNDS::index> cell_push, cell_pushStart, node_push, node_pushStart, bnd_push, bnd_pushStart;
        Partition2LocalIdx(cellPartition, cell_push, cell_pushStart, mesh->mpi);
        Partition2LocalIdx(nodePartition, node_push, node_pushStart, mesh->mpi);
        Partition2LocalIdx(bndPartition, bnd_push, bnd_pushStart, mesh->mpi);
        std::vector<DNDS::index> cell_Serial2Global, node_Serial2Global, bnd_Serial2Global;
        Partition2Serial2Global(cellPartition, cell_Serial2Global, mesh->mpi, mesh->mpi.size);
        Partition2Serial2Global(nodePartition, node_Serial2Global, mesh->mpi, mesh->mpi.size);
        // Partition2Serial2Global(bndPartition, bnd_Serial2Global, mesh->mpi, mesh->mpi.size);//seems not needed for now
        // PushInfo2Serial2Global(cell_Serial2Global, cellPartition.size(), cell_push, cell_pushStart, mesh->mpi);//*safe validation version
        // PushInfo2Serial2Global(node_Serial2Global, nodePartition.size(), node_push, node_pushStart, mesh->mpi);//*safe validation version
        // PushInfo2Serial2Global(bnd_Serial2Global, bndPartition.size(), bnd_push, bnd_pushStart, mesh->mpi);    //*safe validation version

        ConvertAdjSerial2Global(cell2nodeSerial, node_Serial2Global, mesh->mpi);
        ConvertAdjSerial2Global(cell2cellSerial, cell_Serial2Global, mesh->mpi);
        ConvertAdjSerial2Global(bnd2nodeSerial, node_Serial2Global, mesh->mpi);
        ConvertAdjSerial2Global(bnd2cellSerial, cell_Serial2Global, mesh->mpi);

        DNDS_MAKE_SSP(mesh->coords.father, mesh->mpi);
        DNDS_MAKE_SSP(mesh->coords.son, mesh->mpi);
        DNDS_MAKE_SSP(mesh->cellElemInfo.father, ElemInfo::CommType(), ElemInfo::CommMult(), mesh->mpi);
        DNDS_MAKE_SSP(mesh->cellElemInfo.son, ElemInfo::CommType(), ElemInfo::CommMult(), mesh->mpi);
        DNDS_MAKE_SSP(mesh->bndElemInfo.father, ElemInfo::CommType(), ElemInfo::CommMult(), mesh->mpi);
        DNDS_MAKE_SSP(mesh->bndElemInfo.son, ElemInfo::CommType(), ElemInfo::CommMult(), mesh->mpi);
        DNDS_MAKE_SSP(mesh->cell2node.father, mesh->mpi);
        DNDS_MAKE_SSP(mesh->cell2node.son, mesh->mpi);
        DNDS_MAKE_SSP(mesh->cell2cell.father, mesh->mpi);
        DNDS_MAKE_SSP(mesh->cell2cell.son, mesh->mpi);
        DNDS_MAKE_SSP(mesh->bnd2node.father, mesh->mpi);
        DNDS_MAKE_SSP(mesh->bnd2node.son, mesh->mpi);
        DNDS_MAKE_SSP(mesh->bnd2cell.father, mesh->mpi);
        DNDS_MAKE_SSP(mesh->bnd2cell.son, mesh->mpi);

        // coord transferring
        TransferDataSerial2Global(coordSerial, mesh->coords.father, node_push, node_pushStart, mesh->mpi);

        // cells transferring
        TransferDataSerial2Global(cell2cellSerial, mesh->cell2cell.father, cell_push, cell_pushStart, mesh->mpi);
        TransferDataSerial2Global(cell2nodeSerial, mesh->cell2node.father, cell_push, cell_pushStart, mesh->mpi);
        TransferDataSerial2Global(cellElemInfoSerial, mesh->cellElemInfo.father, cell_push, cell_pushStart, mesh->mpi);

        // bnds transferring
        TransferDataSerial2Global(bnd2cellSerial, mesh->bnd2cell.father, bnd_push, bnd_pushStart, mesh->mpi);
        TransferDataSerial2Global(bnd2nodeSerial, mesh->bnd2node.father, bnd_push, bnd_pushStart, mesh->mpi);
        TransferDataSerial2Global(bndElemInfoSerial, mesh->bndElemInfo.father, bnd_push, bnd_pushStart, mesh->mpi);

        DNDS::MPISerialDo(mesh->mpi, [&]()
                          { std::cout << "Rank " << mesh->mpi.rank << " : nCell " << mesh->cell2cell.father->Size() << std::endl; });
        DNDS::MPISerialDo(mesh->mpi, [&]()
                          { std::cout << "Rank " << mesh->mpi.rank << " : nNode " << mesh->coords.father->Size() << std::endl; });
        DNDS::MPISerialDo(mesh->mpi, [&]()
                          { std::cout << "Rank " << mesh->mpi.rank << " : nBnd " << mesh->bnd2node.father->Size() << std::endl; });
    }
}