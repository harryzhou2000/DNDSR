#include "Mesh.hpp"

#include <cgnslib.h>

namespace DNDS::Geom
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
        mode = SerialReadAndDistribute;
        this->dataIsSerialIn = true;
        
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
            DNDS_assert_info(celldim == mesh->dim, "CGNS file need to be with correct dim");
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
                        cgErr |= cg_coord_read(cgns_file, iBase, iZone, "CoordinateZ", RealDouble, RMin, RMax, coordsRead.data());
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
                        // DNDS_assert(cend == start - 1); //? testing//!not valid!
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
                                ZoneElemInfos.back()->operator()(start - 1 + i, 0).zone = Geom::BC_ID_INTERNAL; //! initialized as inner,need new way of doing vol condition
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
                                ZoneElemInfos.back()->operator()(start - 1 + i, 0).zone = Geom::BC_ID_INTERNAL; //! initialized as inner,need new way of doing vol condition
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
                        if (BCCode == BC_ID_NULL)
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
                if (Elem.GetDim() == mesh->dim - 1 && !FaceIDIsTrueInternal(ZoneElemInfos[iGZ]->operator()(iElem, 0).zone))
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
                if (Elem.GetDim() == mesh->dim - 1 && !FaceIDIsTrueInternal(ZoneElemInfos[iGZ]->operator()(iElem, 0).zone)) //! periodic ones are also recorded
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
}