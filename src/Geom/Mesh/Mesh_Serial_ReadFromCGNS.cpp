#include "Mesh.hpp"
#include "NativeMeshCGNS.hpp"
#include "FieldTransfer.hpp"
#include "Geom/OpenFOAMMesh.hpp"

namespace DNDS::Geom
{
    void UnstructuredMeshSerialRW::ReadFromCGNSSerial(const std::string &fName, const t_FBCName_2_ID &FBCName_2_ID)
    {
        mode = SerialReadAndDistribute;
        dataIsSerialIn = true;
        const auto mpi = mesh->getMPI();
        DNDS_check_throw_info(mRank >= 0 && mRank < mpi.size, "invalid CGNS serial reader rank");
        NativeSerialMeshData data;
        // Preserve this serial reader's historical local-call contract. The
        // separate ReadNativeCGNS entry point coordinates collective failures.
        if (mpi.rank == mRank)
            data = ReadNativeCGNSSerial(fName, mesh->dim, FBCName_2_ID);
        // Transitional legacy setup adapter. The parser and native entry point
        // have no dependency on this copy; remaining legacy setup callers do.
        auto copy = [&](auto &destination, const auto &source, MPI_Datatype datatype, int count)
        {
            using Array = typename std::decay_t<decltype(destination)>::element_type;
            destination = std::make_shared<Array>(ObjName{"CGNS compatibility"}, datatype, count, mesh->getMPI());
            auto read = source.Read();
            destination->Resize(index(source.Rows()));
            for (size_t row = 0; row < source.Rows(); ++row)
            {
                if constexpr (Array::IsCSR())
                    destination->ResizeRow(index(row), rowsize(read.View().RowSize(row)));
                for (size_t col = 0; col < read.View().RowSize(row); ++col)
                    (*destination)(index(row), rowsize(col)) = read.View()(row, col);
            }
            destination->Compress();
        };
        copy(coordSerial, data.coordinates, MPI_DOUBLE, 1);
        copy(cell2nodeSerial, data.cellNodes, MPI_INT64_T, 1);
        copy(bnd2nodeSerial, data.boundaryNodes, MPI_INT64_T, 1);
        copy(bnd2cellSerial, data.boundaryCells, MPI_INT64_T, 1);
        copy(cellElemInfoSerial, data.cellInfo, ElemInfo::CommType(), ElemInfo::CommMult());
        copy(bndElemInfoSerial, data.boundaryInfo, ElemInfo::CommType(), ElemInfo::CommMult());
        copy(cell2cellOrigSerial, data.cellIds, MPI_INT64_T, 1);
        copy(node2nodeOrigSerial, data.nodeIds, MPI_INT64_T, 1);
        copy(bnd2bndOrigSerial, data.boundaryIds, MPI_INT64_T, 1);
    }

    void UnstructuredMeshSerialRW::
        ReadFromOpenFOAMAndConvertSerial(const std::string &fName, const std::map<std::string, std::string> &nameMapping, const t_FBCName_2_ID &FBCName_2_ID)
    {
        mode = SerialReadAndDistribute;
        this->dataIsSerialIn = true;

        cell2nodeSerial = make_ssp<decltype(cell2nodeSerial)::element_type>(ObjName{"ReadFromOpenFOAMAndConvertSerial::cell2nodeSerial"}, mesh->getMPI());
        bnd2nodeSerial = make_ssp<decltype(bnd2nodeSerial)::element_type>(ObjName{"ReadFromOpenFOAMAndConvertSerial::bnd2nodeSerial"}, mesh->getMPI());
        coordSerial = make_ssp<decltype(coordSerial)::element_type>(ObjName{"ReadFromOpenFOAMAndConvertSerial::coordSerial"}, mesh->getMPI());
        cellElemInfoSerial = make_ssp<decltype(cellElemInfoSerial)::element_type>(ObjName{"ReadFromOpenFOAMAndConvertSerial::cellElemInfoSerial"}, ElemInfo::CommType(), ElemInfo::CommMult(), mesh->getMPI());
        bndElemInfoSerial = make_ssp<decltype(bndElemInfoSerial)::element_type>(ObjName{"ReadFromOpenFOAMAndConvertSerial::bndElemInfoSerial"}, ElemInfo::CommType(), ElemInfo::CommMult(), mesh->getMPI());
        bnd2cellSerial = make_ssp<decltype(bnd2cellSerial)::element_type>(ObjName{"ReadFromOpenFOAMAndConvertSerial::bnd2cellSerial"}, mesh->getMPI());

        if (mRank != mesh->getMPI().rank) //! parallel done!!! now serial!!!
            return;

        std::filesystem::path ofFilePath(fName);
        std::ifstream pointsIFS(ofFilePath / "points");
        std::ifstream facesIFS(ofFilePath / "faces");
        std::ifstream ownerIFS(ofFilePath / "owner");
        std::ifstream neighbourIFS(ofFilePath / "neighbour");
        std::ifstream boundaryIFS(ofFilePath / "boundary");

        DNDS_assert_info(pointsIFS.is_open(), "points file not found!");
        DNDS_assert_info(facesIFS.is_open(), "faces file not found!");
        DNDS_assert_info(ownerIFS.is_open(), "owner file not found!");
        DNDS_assert_info(neighbourIFS.is_open(), "neighbour file not found!");
        DNDS_assert_info(boundaryIFS.is_open(), "boundary file not found!");

        OpenFOAM::OpenFOAMReader ofReader;
        ofReader.ReadPoints(pointsIFS);
        ofReader.ReadFaces(facesIFS);
        ofReader.ReadOwner(ownerIFS);
        ofReader.ReadNeighbour(neighbourIFS);
        ofReader.ReadBoundary(boundaryIFS);

        OpenFOAM::OpenFOAMConverter ofConverter;
        ofConverter.BuildFaceElemInfo(ofReader);
        ofConverter.BuildCell2Face(ofReader);
        ofConverter.BuildCell2Node(ofReader);

        coordSerial->Resize(ofReader.points.size());
        for (index iN = 0; iN < coordSerial->Size(); iN++)
            (*coordSerial)[iN] = ofReader.points.at(iN);
        log() << "OpenFOAM === got num node: " << coordSerial->Size() << std::endl;

        cell2nodeSerial->Resize(ofConverter.cell2node.size());
        cellElemInfoSerial->Resize(ofConverter.cell2node.size());
        for (index iC = 0; iC < cell2nodeSerial->Size(); iC++)
        {
            cell2nodeSerial->ResizeRow(iC, ofConverter.cell2node[iC].size());
            for (size_t iN = 0; iN < ofConverter.cell2node[iC].size(); iN++)
                (*cell2nodeSerial)[iC][iN] = ofConverter.cell2node[iC][iN];
            cellElemInfoSerial->operator()(iC, 0) = ofConverter.cellElemInfo.at(iC);
        }

        log() << "OpenFOAM === got num cell: " << cell2nodeSerial->Size() << std::endl;

        index nBnd = ofReader.owner.size() - ofReader.neighbour.size();
        bnd2nodeSerial->Resize(nBnd);
        bndElemInfoSerial->Resize(nBnd);
        bnd2cellSerial->Resize(nBnd);
        for (index iBnd = 0; iBnd < nBnd; iBnd++)
        {
            index iFaceOF = iBnd + ofReader.neighbour.size();
            bnd2nodeSerial->ResizeRow(iBnd, ofReader.faces[iFaceOF].size());
            for (size_t ib2c = 0; ib2c < ofReader.faces[iFaceOF].size(); ib2c++)
                (*bnd2nodeSerial)[iBnd][ib2c] = ofReader.faces[iFaceOF][ib2c];
            bnd2cellSerial->operator()(iBnd, 0) = ofReader.owner.at(iFaceOF);
            bnd2cellSerial->operator()(iBnd, 1) = UnInitIndex;
            bndElemInfoSerial->operator()(iBnd, 0) = ofConverter.faceElemInfo.at(iFaceOF);
        }

        log() << "nameMapping size: " << nameMapping.size() << std::endl;

        for (auto &bc : ofReader.boundaryConditions)
        {
            auto boconame = bc.first;
            if (nameMapping.count(boconame))
                boconame = nameMapping.at(boconame);
            t_index BCCode = FBCName_2_ID(std::string(boconame));
            if (BCCode == BC_ID_NULL)
            {
                DNDS_assert_info(false, fmt::format("BC NAME [{}] NOT FOUND IN DATABASE", boconame));
            }
            for (index iBndC = 0; iBndC < bc.second.nFaces; iBndC++)
                bndElemInfoSerial->operator()(iBndC + bc.second.startFace - ofReader.neighbour.size(), 0).zone = BCCode;
        }
        for (index iBnd = 0; iBnd < nBnd; iBnd++)
        {
            DNDS_assert(bndElemInfoSerial->operator()(iBnd, 0).zone != BC_ID_NULL);
        }

        log() << "OpenFOAM === got num bnd: " << nBnd << std::endl;

        log() << "OpenFOAM === Serial Read Done" << std::endl;
    }
}
