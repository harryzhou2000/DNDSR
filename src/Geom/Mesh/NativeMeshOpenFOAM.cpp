#include "NativeMeshOpenFOAM.hpp"
#include "Geom/OpenFOAMMesh.hpp"
#include "DNDS/Runtime/RowBuilder.hpp"
#include "FieldTransfer.hpp"
#include <fstream>
#include <filesystem>
#include <set>

namespace DNDS::Geom
{
    namespace
    {
        void Require(bool value, const char *message) { DNDS_check_throw_info(value, message); }
        template <class T>
        Runtime::Array<T> Rows(const std::vector<std::vector<T>> &rows)
        {
            Runtime::RowBuilder builder(rows.size(), sizeof(T));
            for (size_t row = 0; row < rows.size(); ++row)
            {
                builder.ResizeRow(row, rows[row].size());
                auto output = builder.Row<T>(row);
                std::copy(rows[row].begin(), rows[row].end(), output.data);
            }
            return Runtime::Array<T>::FromSharedStorage(builder.Commit());
        }
        Runtime::Array<index> Ids(size_t size)
        {
            Runtime::Array<index> values(Runtime::RowLayout::Dense(size, 1));
            auto write = values.Write();
            for (size_t row = 0; row < size; ++row)
                write.View()(row, 0) = index(row);
            return values;
        }
    }

    NativeSerialMeshData ReadNativeOpenFOAMSerial(const std::string &path,
                                                  const std::map<std::string, std::string> &nameMapping, const t_FBCName_2_ID &boundaryId)
    {
        OpenFOAM::OpenFOAMReader reader;
        std::array<std::ifstream, 5> streams;
        const char *files[]{"points", "faces", "owner", "neighbour", "boundary"};
        for (size_t i = 0; i < streams.size(); ++i)
        {
            streams[i].open(std::filesystem::path(path) / files[i]);
            Require(streams[i].is_open(), "native OpenFOAM input file missing");
            streams[i].exceptions(std::ios::badbit | std::ios::failbit);
        }
        reader.ReadPoints(streams[0]);
        reader.ReadFaces(streams[1]);
        reader.ReadOwner(streams[2]);
        reader.ReadNeighbour(streams[3]);
        reader.ReadBoundary(streams[4]);
        Require(!reader.points.empty() && reader.owner.size() == reader.faces.size() &&
                    reader.neighbour.size() <= reader.owner.size(),
                "OpenFOAM face/owner counts invalid");
        for (const auto &point : reader.points)
            Require(point.allFinite(), "OpenFOAM coordinate nonfinite");
        index cells = 0;
        for (size_t face = 0; face < reader.faces.size(); ++face)
        {
            const auto &nodes = reader.faces[face];
            Require(nodes.size() == 3 || nodes.size() == 4, "OpenFOAM supports triangular/quadrilateral faces");
            std::set<index> unique;
            for (auto node : nodes)
                Require(node >= 0 && uint64_t(node) < reader.points.size() && unique.insert(node).second, "OpenFOAM face node invalid/repeated");
            Require(reader.owner[face] >= 0 && uint64_t(reader.owner[face]) < reader.faces.size(), "OpenFOAM owner invalid");
            cells = std::max(cells, reader.owner[face] + 1);
            if (face < reader.neighbour.size())
            {
                Require(reader.neighbour[face] >= 0 && uint64_t(reader.neighbour[face]) < reader.faces.size() &&
                            reader.neighbour[face] != reader.owner[face],
                        "OpenFOAM neighbor invalid");
                cells = std::max(cells, reader.neighbour[face] + 1);
            }
        }
        OpenFOAM::OpenFOAMConverter converter;
        converter.BuildFaceElemInfo(reader);
        converter.BuildCell2Face(reader);
        Require(converter.cell2face.size() == size_t(cells), "OpenFOAM cell count invalid");
        for (const auto &faces : converter.cell2face)
        {
            size_t triangles = 0, quads = 0;
            std::set<index> vertices;
            std::map<std::pair<index, index>, int> edgeUses;
            for (auto face : faces)
            {
                const auto &nodes = reader.faces[size_t(face)];
                triangles += nodes.size() == 3;
                quads += nodes.size() == 4;
                vertices.insert(nodes.begin(), nodes.end());
                for (size_t slot = 0; slot < nodes.size(); ++slot)
                {
                    auto a = nodes[slot], b = nodes[(slot + 1) % nodes.size()];
                    if (a > b)
                        std::swap(a, b);
                    ++edgeUses[{a, b}];
                }
            }
            Require(triangles == 0 && quads == 6 && vertices.size() == 8,
                    "OpenFOAM converter currently supports canonical Hex8 cells only");
            for (const auto &[edge, count] : edgeUses)
                Require(count == 2, "OpenFOAM cell surface is not closed");
        }
        converter.BuildCell2Node(reader);
        const size_t boundaryStart = reader.neighbour.size(), boundaryCount = reader.faces.size() - boundaryStart;
        std::vector<std::vector<index>> boundaryRows, boundaryCells;
        std::vector<std::vector<ElemInfo>> boundaryInfo;
        for (size_t face = boundaryStart; face < reader.faces.size(); ++face)
        {
            boundaryRows.push_back(reader.faces[face]);
            boundaryCells.push_back({reader.owner[face], UnInitIndex});
            boundaryInfo.push_back({converter.faceElemInfo[face]});
        }
        std::vector<bool> assigned(boundaryCount);
        for (const auto &[name, patch] : reader.boundaryConditions)
        {
            Require(patch.startFace >= index(boundaryStart) && patch.nFaces >= 0 &&
                        uint64_t(patch.startFace) <= reader.faces.size() && uint64_t(patch.nFaces) <= reader.faces.size() - size_t(patch.startFace),
                    "OpenFOAM boundary range invalid");
            const auto mapped = nameMapping.count(name) ? nameMapping.at(name) : name;
            const auto id = boundaryId(mapped);
            Require(id != BC_ID_NULL, "OpenFOAM boundary ID unavailable");
            for (index offset = 0; offset < patch.nFaces; ++offset)
            {
                const auto row = size_t(patch.startFace + offset) - boundaryStart;
                Require(!assigned[row], "OpenFOAM boundary patches overlap");
                assigned[row] = true;
                boundaryInfo[row][0].zone = id;
            }
        }
        Require(std::all_of(assigned.begin(), assigned.end(), [](bool value)
                            { return value; }),
                "OpenFOAM boundaries not fully assigned");
        NativeSerialMeshData result;
        result.coordinates = Runtime::Array<real>(Runtime::RowLayout::Dense(reader.points.size(), 3));
        {
            auto write = result.coordinates.Write();
            for (size_t row = 0; row < reader.points.size(); ++row)
                for (size_t component = 0; component < 3; ++component)
                    write.View()(row, component) = reader.points[row][Eigen::Index(component)];
        }
        result.cellNodes = Rows(converter.cell2node);
        result.boundaryNodes = Rows(boundaryRows);
        result.boundaryCells = Rows(boundaryCells);
        std::vector<std::vector<ElemInfo>> cellInfo;
        for (auto info : converter.cellElemInfo)
            cellInfo.push_back({info});
        result.cellInfo = Rows(cellInfo);
        result.boundaryInfo = Rows(boundaryInfo);
        result.cellIds = Ids(converter.cell2node.size());
        result.nodeIds = Ids(reader.points.size());
        result.boundaryIds = Ids(boundaryCount);
        return result;
    }

    std::shared_ptr<const NativeMeshState> ReadNativeOpenFOAM(const MPIInfo &mpi, const std::string &path,
                                                              const std::map<std::string, std::string> &nameMapping, const t_FBCName_2_ID &boundaryId, int readerRank,
                                                              NativeMeshBuildOptions options, std::optional<NativePeriodicImportOptions> periodic, NativeSerialDistribution distribution)
    {
        Require(mpi.comm != MPI_COMM_NULL, "native OpenFOAM requires a communicator");
        std::string signature;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            int rank = -1, size = 0;
            MPI_Comm_rank(mpi.comm, &rank); MPI_Comm_size(mpi.comm, &size);
            Require(rank == mpi.rank && size == mpi.size && readerRank >= 0 && readerRank < size &&
                    options.cellLayers >= 1 && path.size() <= size_t(INT_MAX), "OpenFOAM reader options invalid");
            signature = nlohmann::json{{"path", path}, {"mapping", nameMapping}, {"reader", readerRank},
                {"layers", options.cellLayers}, {"star", options.completeNodeStar}, {"periodic", bool(periodic)}, {"distribution", int(distribution)}}.dump();
            if (periodic)
            {
                std::vector<real> frame{periodic->tolerance};
                for (size_t f = 0; f < 4; ++f)
                {
                    for (int i = 0; i < 9; ++i) frame.push_back(periodic->frames.rotation[f].map().data()[i]);
                    for (int i = 0; i < 3; ++i) frame.push_back(periodic->frames.translation[f].map()[i]);
                    for (int i = 0; i < 3; ++i) frame.push_back(periodic->frames.rotationCenter[f].map()[i]);
                }
                for (auto value : frame) Require(std::isfinite(value), "OpenFOAM periodic settings nonfinite");
                Require(periodic->tolerance > 0, "OpenFOAM periodic tolerance invalid");
                signature += nlohmann::json(frame).dump();
            }
            Require(signature.size() <= size_t(INT_MAX), "OpenFOAM settings too large"); });
        int count = mpi.rank == 0 ? int(signature.size()) : 0;
        MPI_Bcast(&count, 1, MPI_INT, 0, mpi.comm);
        std::string expected = mpi.rank == 0 ? signature : std::string(size_t(count), '\0');
        MPI_Bcast(expected.data(), count, MPI_CHAR, 0, mpi.comm);
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { Require(signature == expected, "OpenFOAM settings differ across ranks"); });
        NativeSerialMeshData serial;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { if (mpi.rank == readerRank) serial = ReadNativeOpenFOAMSerial(path, nameMapping, boundaryId); });
        return DistributeNativeSerialMesh(mpi, std::move(serial), 3, readerRank, options, std::move(periodic), distribution);
    }
}
