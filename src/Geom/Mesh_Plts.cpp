#include "Mesh.hpp"

#include <filesystem>
#include "base64_rfc4648.hpp"

namespace DNDS::Geom
{
    void UnstructuredMeshSerialRW::
        PrintSerialPartPltBinaryDataArray(std::string fname,
                                          int arraySiz, int arraySizPoint,
                                          const std::function<std::string(int)> &names,
                                          const std::function<DNDS::real(int, DNDS::index)> &data,
                                          const std::function<std::string(int)> &namesPoint,
                                          const std::function<DNDS::real(int, DNDS::index)> &dataPoint,
                                          double t, int flag)
    {
        auto mpi = mesh->mpi;
        std::string fnameIn = fname;

        if (mpi.rank != mRank && flag == 0) //* now only operating on mRank if serial
            return;

        if (flag == 0)
        {
            fname += ".plt";
            DNDS_assert(mode == SerialOutput && dataIsSerialOut);
        }
        if (flag == 1)
        {

            std::filesystem::path outPath{fname + ".dir"};
            std::filesystem::create_directories(outPath);
            char BUF[512];
            std::sprintf(BUF, "%06d", mpi.rank);
            fname = outPath / (std::string(BUF) + ".plt");
        }

        std::ofstream fout(fname, std::ios::binary);
        if (!fout)
        {
            DNDS::log() << "Error: WriteMeshDebugTecASCII open \"" << fname << "\" failure" << std::endl;
            DNDS_assert(false);
        }
        const char magic_word[] = "#!TDV112";
        const int b_magic_word = sizeof(magic_word) - 1;
        int32_t intBuf;
        double_t doubleBuf;
        float_t floatBuf;

        auto writeInt = [&](int d) -> void
        {
            intBuf = d;
            fout.write((char *)(&intBuf), sizeof(intBuf));
        };
        auto writeFloat = [&](float_t d) -> void
        {
            floatBuf = d;
            fout.write((char *)(&floatBuf), sizeof(floatBuf));
        };
        auto writeDouble = [&](double_t d) -> void
        {
            doubleBuf = d;
            fout.write((char *)(&doubleBuf), sizeof(doubleBuf));
        };
        auto writeString = [&](const std::string &title) -> void
        {
            for (auto i : title)
            {
                intBuf = i;
                fout.write((char *)(&intBuf), sizeof(intBuf));
            }
            intBuf = 0;
            fout.write((char *)(&intBuf), sizeof(intBuf));
        };
        fout.write(magic_word, b_magic_word);
        writeInt(1);
        writeInt(0); //! full: write both grid and data
        writeString("Title Here");
        writeInt(arraySiz + 3 + 1 + arraySizPoint); // nvars
        writeString("X");
        writeString("Y");
        writeString("Z");
        for (int idata = 0; idata < arraySiz; idata++)
            writeString(names(idata));
        writeString("iPart");
        for (int idata = 0; idata < arraySizPoint; idata++)
            writeString(namesPoint(idata));

        /********************************/
        // cellZone in header
        /********************************/
        writeFloat(299.0f); // 299.0 indicates a v112 zone header, available in v191
        writeString("zone_0");
        writeInt(-1);   // ParentZone: No longer used.
        writeInt(-1);   // StrandID: static strand ID
        writeDouble(t); // solution time
        writeInt(-1);   // default zone color
        if (mesh->dim == 2)
            writeInt(3); // 2d: quad zone
        else if (mesh->dim == 3)
            writeInt(5); // 3d: brick zone
        else if (mesh->dim == 1)
            writeInt(1); // 1d: line zone
        writeInt(1);     // specifyVarLocation

        for (int idim = 0; idim < 3; idim++)
            writeInt(0); // xyz at node
        for (int idata = 0; idata < arraySiz; idata++)
            writeInt(1); // data at center
        writeInt(1);     // iPart
        for (int idata = 0; idata < arraySizPoint; idata++)
            writeInt(0); // data at point

        writeInt(0); // Are raw local 1-to-1 face neighbors supplied?
        writeInt(0); // Number of miscellaneous user-defined face

        /*************************************************/
        // get the output arrays: serial or parallel
        tCoord coordSerialDummy;
        tAdj cell2nodeSerialDummy;
        tElemInfoArray cellElemInfoSerialDummy;
        DNDS_MAKE_SSP(coordSerialDummy, mesh->mpi);
        DNDS_MAKE_SSP(cell2nodeSerialDummy, mesh->mpi);
        DNDS_MAKE_SSP(cellElemInfoSerialDummy, ElemInfo::CommType(), ElemInfo::CommMult(), mesh->mpi);
        tCoordPair coordOut;
        tAdjPair cell2nodeOut;
        tElemInfoArrayPair cellElemInfoOut;
        DNDS::index nCell = 0;
        DNDS::index nNode = 0;
        if (flag == 0)
        {
            coordOut.father = coordSerial;
            coordOut.son = coordSerialDummy;
            cell2nodeOut.father = cell2nodeSerial;
            cell2nodeOut.son = cell2nodeSerialDummy;
            cellElemInfoOut.father = cellElemInfoSerial;
            cellElemInfoOut.son = cellElemInfoSerialDummy;
            nCell = cell2nodeOut.Size();
            nNode = coordOut.Size();
        }
        else if (flag == 1)
        {
            coordOut.father = mesh->coords.father;
            coordOut.son = mesh->coords.son;
            cell2nodeOut.father = mesh->cell2node.father;
            cell2nodeOut.son = mesh->cell2node.son;
            cellElemInfoOut.father = mesh->cellElemInfo.father;
            cellElemInfoOut.son = mesh->cellElemInfo.son;
            nCell = cell2nodeOut.father->Size(); //! only non-ghost cells are output
            nNode = coordOut.Size();             //! need all the nodes with ghost
        }
        /*************************************************/

        writeInt(nNode); // node number
        writeInt(nCell); // cell number

        writeInt(0); // I dim
        writeInt(0); // J dim
        writeInt(0); // K dim
        writeInt(0); // No more Auxiliary name/value pairs.

        writeFloat(357.0f); // end of header, EOH marker

        writeFloat(299.0f); // 299.0 indicates a v112 zone header, available in v191

        /********************************/
        // cellZone data
        /********************************/

        for (int idim = 0; idim < 3; idim++)
            writeInt(2); // double for node
        for (int idata = 0; idata < arraySiz; idata++)
            writeInt(2); // double for data
        writeInt(2);     // double for iPart
        for (int idata = 0; idata < arraySizPoint; idata++)
            writeInt(2); // double for data pint

        writeInt(0);  // no passive
        writeInt(0);  // no sharing
        writeInt(-1); // no sharing

        std::vector<double_t> minVal(3 + arraySiz, DNDS::veryLargeReal);
        std::vector<double_t> maxVal(3 + arraySiz, -DNDS::veryLargeReal); // for all non-shared non-passive
        std::vector<double_t> minValPoint(arraySizPoint, DNDS::veryLargeReal);
        std::vector<double_t> maxValPoint(arraySizPoint, DNDS::veryLargeReal);
        for (int idim = 0; idim < 3; idim++)
            for (DNDS::index i = 0; i < nNode; i++)
            {
                minVal[idim] = std::min(coordOut[i](idim), minVal[idim]);
                maxVal[idim] = std::max(coordOut[i](idim), maxVal[idim]);
            };

        for (int idata = 0; idata < arraySiz; idata++)
            for (DNDS::index iv = 0; iv < nCell; iv++)
            {
                minVal[3 + idata] = std::min(data(idata, iv), minVal[3 + idata]);
                maxVal[3 + idata] = std::max(data(idata, iv), maxVal[3 + idata]);
            }

        for (int idata = 0; idata < arraySizPoint; idata++)
            for (DNDS::index iv = 0; iv < nNode; iv++)
            {
                minValPoint[idata] = std::min(dataPoint(idata, iv), minValPoint[idata]);
                maxValPoint[idata] = std::max(dataPoint(idata, iv), maxValPoint[idata]);
            }

        for (int idim = 0; idim < 3; idim++)
        {
            writeDouble(minVal[idim]);
            writeDouble(maxVal[idim]);
        }
        for (int idata = 0; idata < arraySiz; idata++)
        {
            writeDouble(minVal[3 + idata]);
            writeDouble(maxVal[3 + idata]);
        }
        writeDouble(0);
        writeDouble(mpi.size);
        for (int idata = 0; idata < arraySizPoint; idata++)
        {
            writeDouble(minValPoint[3 + idata]);
            writeDouble(maxValPoint[3 + idata]);
        }

        for (int idim = 0; idim < 3; idim++)
            for (DNDS::index i = 0; i < nNode; i++)
            {
                writeDouble(coordOut[i](idim));
                // std::cout << (*coordSerial)[i](idim) << std::endl;
            };

        for (int idata = 0; idata < arraySiz; idata++)
            for (DNDS::index iv = 0; iv < nCell; iv++)
            {
                writeDouble(data(idata, iv));
            }

        for (DNDS::index iv = 0; iv < nCell; iv++)
        {
            DNDS::MPI_int r = -1;
            DNDS::index v = -1;
            if (flag == 0)
                cell2nodeSerialOutTrans.pLGlobalMapping->search(iv, r, v);
            else if (flag == 1)
                r = mesh->mpi.rank;
            writeDouble(r);
        }

        for (int idata = 0; idata < arraySizPoint; idata++)
            for (DNDS::index in = 0; in < nNode; in++)
            {
                writeDouble(data(idata, in));
            }

        for (DNDS::index iv = 0; iv < nCell; iv++)
        {
            auto elem = Elem::Element{cellElemInfoOut[iv]->getElemType()};
            auto c2n = cell2nodeOut[iv];
            switch (elem.GetParamSpace())
            {
            case Elem::ParamSpace::LineSpace:
                writeInt(c2n[0] + 0);
                writeInt(c2n[1] + 0);
                break;
            case Elem::ParamSpace::TriSpace:
                writeInt(c2n[0] + 0);
                writeInt(c2n[1] + 0);
                writeInt(c2n[2] + 0);
                writeInt(c2n[2] + 0);
                break;
            case Elem::ParamSpace::QuadSpace:
                writeInt(c2n[0] + 0);
                writeInt(c2n[1] + 0);
                writeInt(c2n[2] + 0);
                writeInt(c2n[3] + 0); // ! note that tis is zero based
                break;
            case Elem::ParamSpace::TetSpace:
                writeInt(c2n[0] + 0);
                writeInt(c2n[1] + 0);
                writeInt(c2n[2] + 0);
                writeInt(c2n[2] + 0);
                writeInt(c2n[3] + 0);
                writeInt(c2n[3] + 0);
                writeInt(c2n[3] + 0);
                writeInt(c2n[3] + 0);
                break;
            case Elem::ParamSpace::HexSpace:
                writeInt(c2n[0] + 0);
                writeInt(c2n[1] + 0);
                writeInt(c2n[2] + 0);
                writeInt(c2n[3] + 0);
                writeInt(c2n[4] + 0);
                writeInt(c2n[5] + 0);
                writeInt(c2n[6] + 0);
                writeInt(c2n[7] + 0);
                break;
            case Elem::ParamSpace::PrismSpace:
                writeInt(c2n[0] + 0);
                writeInt(c2n[1] + 0);
                writeInt(c2n[2] + 0);
                writeInt(c2n[2] + 0);
                writeInt(c2n[3] + 0);
                writeInt(c2n[4] + 0);
                writeInt(c2n[5] + 0);
                writeInt(c2n[5] + 0);
                break;
            case Elem::ParamSpace::PyramidSpace:
                writeInt(c2n[0] + 0);
                writeInt(c2n[1] + 0);
                writeInt(c2n[2] + 0);
                writeInt(c2n[3] + 0);
                writeInt(c2n[4] + 0);
                writeInt(c2n[4] + 0);
                writeInt(c2n[4] + 0);
                writeInt(c2n[4] + 0);
                break;
            default:
                DNDS_assert(false); //! 2d or 3d elems
            }
        };
        fout.close();
    }

    void UnstructuredMeshSerialRW::PrintSerialPartVTKDataArray(
        std::string fname,
        int arraySiz, int vecArraySiz, int arraySizPoint, int vecArraySizPoint,
        const std::function<std::string(int)> &names,
        const std::function<DNDS::real(int, DNDS::index)> &data,
        const std::function<std::string(int)> &vectorNames,
        const std::function<DNDS::real(int, DNDS::index, DNDS::rowsize)> &vectorData,
        const std::function<std::string(int)> &namesPoint,
        const std::function<DNDS::real(int, DNDS::index)> &dataPoint,
        const std::function<std::string(int)> &vectorNamesPoint,
        const std::function<DNDS::real(int, DNDS::index, DNDS::rowsize)> &vectorDataPoint,
        double t, int flag)
    {
        auto mpi = mesh->mpi;
        std::string fnameIN = fname;

        if (mpi.rank != mRank && flag == 0) //* now only operating on mRank if serial
            return;

        if (flag == 0)
        {
            fname += ".vtu";
            DNDS_assert(mode == SerialOutput && dataIsSerialOut);
        }
        std::filesystem::path outPath; // only valid if parallel out
        if (flag == 1)
        {
            outPath = {fname + ".vtu.dir"};
            std::filesystem::create_directories(outPath);
            char BUF[512];
            std::sprintf(BUF, "%04d", mpi.rank);
            fname = outPath / (std::string(BUF) + ".vtu");
        }

        std::ofstream fout(fname);
        if (!fout)
        {
            DNDS::log() << "Error: PrintSerialPartVTKDataArray open \"" << fname << "\" failure" << std::endl;
            DNDS_assert(false);
        }

        /*************************************************/
        // get the output arrays: serial or parallel
        tCoord coordSerialDummy;
        tAdj cell2nodeSerialDummy;
        tElemInfoArray cellElemInfoSerialDummy;
        DNDS_MAKE_SSP(coordSerialDummy, mesh->mpi);
        DNDS_MAKE_SSP(cell2nodeSerialDummy, mesh->mpi);
        DNDS_MAKE_SSP(cellElemInfoSerialDummy, ElemInfo::CommType(), ElemInfo::CommMult(), mesh->mpi);
        tCoordPair coordOut;
        tAdjPair cell2nodeOut;
        tElemInfoArrayPair cellElemInfoOut;
        DNDS::index nCell = 0;
        DNDS::index nNode = 0;
        if (flag == 0)
        {
            coordOut.father = coordSerial;
            coordOut.son = coordSerialDummy;
            cell2nodeOut.father = cell2nodeSerial;
            cell2nodeOut.son = cell2nodeSerialDummy;
            cellElemInfoOut.father = cellElemInfoSerial;
            cellElemInfoOut.son = cellElemInfoSerialDummy;
            nCell = cell2nodeOut.Size();
            nNode = coordOut.Size();
        }
        else if (flag == 1)
        {
            coordOut.father = mesh->coords.father;
            coordOut.son = mesh->coords.son;
            cell2nodeOut.father = mesh->cell2node.father;
            cell2nodeOut.son = mesh->cell2node.son;
            cellElemInfoOut.father = mesh->cellElemInfo.father;
            cellElemInfoOut.son = mesh->cellElemInfo.son;
            nCell = cell2nodeOut.father->Size(); //! only non-ghost cells are output
            nNode = coordOut.Size();             //! need all the nodes with ghost
        }
        /*************************************************/

        std::string indentV = "  ";
        std::string newlineV = "\n";

        auto writeXMLEntity =
            [&](
                std::ostream &out, int level, const auto &name,
                const std::vector<std::pair<std::string, std::string>> &attr,
                auto &&writeContent) -> void
        {
            for (int i = 0; i < level; i++)
                out << indentV;
            out << "<" << name << " ";
            for (auto &a : attr)
                out << a.first << "=\"" << a.second << "\" ";
            out << ">" << newlineV;

            writeContent(out, level + 1);

            for (int i = 0; i < level; i++)
                out << indentV;
            out << "</" << name << ">" << newlineV;
        };

        auto writeCoords = [&](auto &out, int level)
        {
            writeXMLEntity(
                out, level, "Points",
                {},
                [&](auto &out, int level)
                {
                    writeXMLEntity(
                        out, level, "DataArray",
                        {{"type", "Float64"},
                         {"NumberOfComponents", "3"},
                         {"format", "ascii"}},
                        [&](auto &out, int level)
                        {
                            std::vector<double> coordsOutData(nNode * 3);
                            for (index i = 0; i < nNode; i++)
                            {
                                coordsOutData[i * 3 + 0] = coordOut[i](0);
                                coordsOutData[i * 3 + 1] = coordOut[i](1);
                                coordsOutData[i * 3 + 2] = coordOut[i](2);
                            }
                            // out << cppcodec::base64_rfc4648::encode(
                            //     (uint8_t *)coordsOutData.data(),
                            //     nNode * 3 * sizeof(double));
                            for (auto v : coordsOutData)
                                out << std::setprecision(16) << v << " ";
                            out << newlineV;
                        });
                });
        };

        std::vector<int64_t> cell2nodeOutData;
        cell2nodeOutData.reserve(cell2nodeOut.father->DataSize());
        std::vector<int64_t> cell2nodeOffsetData(nCell);
        std::vector<uint8_t> cellTypeData(nCell);
        index cellEnd = 0;
        for (index iCell = 0; iCell < nCell; iCell++)
        {
            auto elem = Elem::Element{cellElemInfoOut[iCell]->getElemType()};
            auto c2n = cell2nodeOut[iCell];
            auto vtkCell = Elem::ToVTKVertsAndData(elem, c2n);
            cellTypeData[iCell] = vtkCell.first;
            for (auto in : vtkCell.second)
                cell2nodeOutData.push_back(in);
            cellEnd += vtkCell.second.size();
            cell2nodeOffsetData[iCell] = cellEnd;
        }

        auto writeCells = [&](auto &out, int level)
        {
            writeXMLEntity(
                out, level, "Cells",
                {},
                [&](auto &out, int level)
                {
                    writeXMLEntity(
                        out, level, "DataArray",
                        {{"type", "Int64"},
                         {"Name", "connectivity"},
                         {"format", "ascii"}},
                        [&](auto &out, int level)
                        {
                            // out << cppcodec::base64_rfc4648::encode(
                            //     (uint8_t *)cell2nodeOutData.data(),
                            //     cell2nodeOutData.size() * sizeof(int64_t));
                            for (auto v : cell2nodeOutData)
                                out << v << " ";
                            out << newlineV;
                        });
                    writeXMLEntity(
                        out, level, "DataArray",
                        {{"type", "Int64"},
                         {"Name", "offsets"},
                         {"format", "ascii"}},
                        [&](auto &out, int level)
                        {
                            // out << cppcodec::base64_rfc4648::encode(
                            //     (uint8_t *)cell2nodeOffsetData.data(),
                            //     cell2nodeOffsetData.size() * sizeof(int64_t));
                            for (auto v : cell2nodeOffsetData)
                                out << v << " ";
                            out << newlineV;
                        });
                    writeXMLEntity(
                        out, level, "DataArray",
                        {{"type", "UInt8"},
                         {"Name", "types"},
                         {"format", "ascii"}},
                        [&](auto &out, int level)
                        {
                            // out << cppcodec::base64_rfc4648::encode(
                            //     (uint8_t *)cellTypeData.data(),
                            //     cellTypeData.size() * sizeof(uint8_t));
                            for (int v : cellTypeData) //! cout << uint8_t(v) is ill-posed
                                out << v << " ";
                            out << newlineV;
                        });
                });
        };

        auto writeCellData = [&](auto &out, int level)
        {
            std::string namesAll;
            for (int i = 0; i < arraySiz; i++)
                namesAll.append(names(i) + " ");
            std::string vectorsNameAll;
            for (int i = 0; i < vecArraySiz; i++)
                vectorsNameAll.append(vectorNames(i) + " ");

            writeXMLEntity(
                out, level, "CellData",
                {{"Scalars", namesAll},
                 {"Vectors", vectorsNameAll}},
                [&](auto &out, int level)
                {
                    for (int i = 0; i < arraySiz; i++)
                    {
                        writeXMLEntity(
                            out, level, "DataArray",
                            {{"type", "Float64"},
                             {"Name", names(i)},
                             {"format", "ascii"}},
                            [&](auto &out, int level)
                            {
                                std::vector<double> dataOutC(nCell);
                                for (index iCell = 0; iCell < nCell; iCell++)
                                    dataOutC[iCell] = data(i, iCell);
                                // out << cppcodec::base64_rfc4648::encode(
                                //     (uint8_t *)dataOutC.data(),
                                //     dataOutC.size() * sizeof(double));
                                for (auto v : dataOutC)
                                    out << std::setprecision(16) << v << " ";
                                out << newlineV;
                            });
                    }
                    for (int i = 0; i < vecArraySiz; i++)
                    {
                        writeXMLEntity(
                            out, level, "DataArray",
                            {{"type", "Float64"},
                             {"Name", vectorNames(i)},
                             {"NumberOfComponents", "3"},
                             {"format", "ascii"}},
                            [&](auto &out, int level)
                            {
                                std::vector<double> dataOutC(nCell * 3);
                                for (index iCell = 0; iCell < nCell; iCell++)
                                {
                                    dataOutC[iCell * 3 + 0] = vectorData(i, iCell, 0);
                                    dataOutC[iCell * 3 + 1] = vectorData(i, iCell, 1);
                                    dataOutC[iCell * 3 + 2] = vectorData(i, iCell, 2);
                                }
                                // out << cppcodec::base64_rfc4648::encode(
                                //     (uint8_t *)dataOutC.data(),
                                //     dataOutC.size() * sizeof(double));
                                for (auto v : dataOutC)
                                    out << std::setprecision(16) << v << " ";
                                out << newlineV;
                            });
                    }
                });
        };

        auto writePointData = [&](auto &out, int level)
        {
            std::string namesAll;
            for (int i = 0; i < arraySizPoint; i++)
                namesAll.append(namesPoint(i) + " ");
            std::string vectorsNameAll;
            for (int i = 0; i < vecArraySizPoint; i++)
                vectorsNameAll.append(vectorNamesPoint(i) + " ");

            writeXMLEntity(
                out, level, "PointData",
                {{"Scalars", namesAll},
                 {"Vectors", vectorsNameAll}},
                [&](auto &out, int level)
                {
                    for (int i = 0; i < arraySizPoint; i++)
                    {
                        writeXMLEntity(
                            out, level, "DataArray",
                            {{"type", "Float64"},
                             {"Name", namesPoint(i)},
                             {"format", "ascii"}},
                            [&](auto &out, int level)
                            {
                                for (index ii = 0; ii < nNode; ii++)
                                    out << std::setprecision(16) << data(i, ii) << " ";
                                out << newlineV;
                            });
                    }
                    for (int i = 0; i < vecArraySiz; i++)
                    {
                        writeXMLEntity(
                            out, level, "DataArray",
                            {{"type", "Float64"},
                             {"Name", vectorNamesPoint(i)},
                             {"NumberOfComponents", "3"},
                             {"format", "ascii"}},
                            [&](auto &out, int level)
                            {
                                for (index ii = 0; ii < nNode; ii++)
                                {
                                    out << std::setprecision(16) << vectorData(i, ii, 0) << " ";
                                    out << std::setprecision(16) << vectorData(i, ii, 1) << " ";
                                    out << std::setprecision(16) << vectorData(i, ii, 2) << " ";
                                }
                                out << newlineV;
                            });
                    }
                });
        };

        std::string endianName{"BigEndian"};

        uint32_t beTest = 1; // need c++20 for STL support
        uint8_t beTestS = *(uint8_t *)(&beTest);
        if (beTestS == 1)
            endianName = "LittleEndian";

        writeXMLEntity(
            fout, 0, "VTKFile",
            {{"type", "UnstructuredGrid"},
             {"byte_order", endianName}},
            [&](auto &out, int level)
            {
                writeXMLEntity(
                    out, level, "UnstructuredGrid",
                    {},
                    [&](auto &out, int level)
                    {
                        writeXMLEntity(
                            out, level, "Piece",
                            {{"NumberOfPoints", std::to_string(nNode)},
                             {"NumberOfCells", std::to_string(nCell)}},
                            [&](auto &out, int level)
                            {
                                /************************/
                                // coord data
                                writeCoords(out, level);
                                writeCells(out, level);
                                writeCellData(out, level);
                                writePointData(out, level);
                            });
                    });
            });

        if (mpi.rank == mRank && flag == 1)
        {
            std::ofstream foutP{fnameIN + ".pvtu"};
            DNDS_assert(foutP);

            writeXMLEntity(
                foutP, 0, "VTKFile",
                {{"type", "PUnstructuredGrid"},
                 {"byte_order", endianName}},
                [&](auto &out, int level)
                {
                    writeXMLEntity(
                        out, level, "PUnstructuredGrid",
                        {{"GhostLevel", "0"}},
                        [&](auto &out, int level)
                        {
                            {
                                std::string namesAll;
                                for (int i = 0; i < arraySiz; i++)
                                    namesAll.append(names(i) + " ");
                                std::string vectorsNameAll;
                                for (int i = 0; i < vecArraySiz; i++)
                                    vectorsNameAll.append(vectorNames(i) + " ");
                                writeXMLEntity(
                                    out, level, "PCellData",
                                    {{"Scalars", namesAll},
                                     {"Vectors", vectorsNameAll}},
                                    [&](auto &out, int level)
                                    {
                                        for (int i = 0; i < arraySiz; i++)
                                        {
                                            writeXMLEntity(
                                                out, level, "PDataArray",
                                                {{"type", "Float64"},
                                                 {"Name", names(i)},
                                                 {"format", "ascii"}},
                                                [&](auto &out, int level) {
                                                });
                                        }
                                        for (int i = 0; i < vecArraySiz; i++)
                                        {
                                            writeXMLEntity(
                                                out, level, "PDataArray",
                                                {{"type", "Float64"},
                                                 {"Name", vectorNames(i)},
                                                 {"NumberOfComponents", "3"},
                                                 {"format", "ascii"}},
                                                [&](auto &out, int level) {
                                                });
                                        }
                                    });
                            }

                            {
                                std::string namesAll;
                                for (int i = 0; i < arraySizPoint; i++)
                                    namesAll.append(namesPoint(i) + " ");
                                std::string vectorsNameAll;
                                for (int i = 0; i < vecArraySizPoint; i++)
                                    vectorsNameAll.append(vectorNamesPoint(i) + " ");
                                writeXMLEntity(
                                    out, level, "PPointData",
                                    {{"Scalars", namesAll},
                                     {"Vectors", vectorsNameAll}},
                                    [&](auto &out, int level)
                                    {
                                        for (int i = 0; i < arraySizPoint; i++)
                                        {
                                            writeXMLEntity(
                                                out, level, "PDataArray",
                                                {{"type", "Float64"},
                                                 {"Name", namesPoint(i)},
                                                 {"format", "ascii"}},
                                                [&](auto &out, int level) {
                                                });
                                        }
                                        for (int i = 0; i < vecArraySizPoint; i++)
                                        {
                                            writeXMLEntity(
                                                out, level, "PDataArray",
                                                {{"type", "Float64"},
                                                 {"Name", vectorNamesPoint(i)},
                                                 {"NumberOfComponents", "3"},
                                                 {"format", "ascii"}},
                                                [&](auto &out, int level) {
                                                });
                                        }
                                    });
                            }
                            writeXMLEntity(
                                out, level, "PPoints",
                                {},
                                [&](auto &out, int level)
                                {
                                    writeXMLEntity(
                                        out, level, "PDataArray",
                                        {{"type", "Float64"},
                                         {"NumberOfComponents", "3"}},
                                        [&](auto &out, int level) {
                                        });
                                });
                            for (MPI_int iRank = 0; iRank < mpi.size; iRank++)
                            {
                                char BUF[512];
                                std::sprintf(BUF, "%04d", iRank);
                                std::string cFileName = outPath / (std::string(BUF) + ".vtu");
                                std::string cFileNameRelPVTU = outPath.lexically_relative(outPath.parent_path()) / (std::string(BUF) + ".vtu");
                                writeXMLEntity(
                                    out, level, "Piece",
                                    {{"Source", cFileNameRelPVTU}},
                                    [&](auto &out, int level) {

                                    });
                            }
                        });
                });
        }
    }
}