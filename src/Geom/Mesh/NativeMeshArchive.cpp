#include "NativeMeshArchive.hpp"
#include "Mesh.hpp"
#include "DNDS/Runtime/ArrayIO.hpp"
#include <nlohmann/json.hpp>
#include <cmath>
#include <limits>

namespace DNDS::Geom
{
    namespace
    {
        using Json = nlohmann::json;
        namespace R = Runtime;
        namespace S = Serializer;

        template <class F>
        void ArchiveStage(const MPIInfo &mpi, F &&action)
        {
            std::exception_ptr error;
            int valid = 1;
            try
            {
                action();
            }
            catch (...)
            {
                error = std::current_exception();
                valid = 0;
            }
            MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, mpi.comm);
            if (error)
                std::rethrow_exception(error);
            DNDS_check_throw_info(valid, "native mesh archive operation failed on another rank");
        }

        void Agree(const MPIInfo &mpi, const std::string &value)
        {
            ArchiveStage(mpi, [&]
                         { DNDS_check_throw_info(value.size() <= size_t(INT_MAX), "native archive metadata exceeds MPI range"); });
            int length = int(value.size());
            MPI_Bcast(&length, 1, MPI_INT, 0, mpi.comm);
            std::string expected;
            ArchiveStage(mpi, [&]
                         { expected = mpi.rank == 0 ? value : std::string(size_t(length), '\0'); });
            MPI_Bcast(expected.data(), length, MPI_CHAR, 0, mpi.comm);
            ArchiveStage(mpi, [&]
                         { DNDS_check_throw_info(value == expected, "native archive metadata or collective options differ between ranks"); });
        }

        int64_t Integer(const Json &value, int64_t minimum = 0, int64_t maximum = INT64_MAX)
        {
            DNDS_check_throw_info(value.is_number_integer(), "native archive integer metadata has a noninteger type");
            if (value.is_number_unsigned())
            {
                const auto number = value.get<uint64_t>();
                DNDS_check_throw_info(number <= uint64_t(maximum) && (minimum <= 0 || number >= uint64_t(minimum)), "native archive integer metadata exceeds its range");
                return int64_t(number);
            }
            const auto number = value.get<int64_t>();
            DNDS_check_throw_info(number >= minimum && number <= maximum, "native archive integer metadata exceeds its range");
            return number;
        }

        bool Boolean(const Json &value)
        {
            DNDS_check_throw_info(value.is_boolean(), "native archive boolean metadata has an invalid type");
            return value.get<bool>();
        }

        bool ValidNamespace(const std::string &value)
        {
            if (value.size() != 36 || value == "00000000-0000-0000-0000-000000000000")
                return false;
            for (size_t i = 0; i < value.size(); ++i)
            {
                const bool dash = i == 8 || i == 13 || i == 18 || i == 23;
                if (dash ? value[i] != '-' : !((value[i] >= '0' && value[i] <= '9') || (value[i] >= 'a' && value[i] <= 'f')))
                    return false;
            }
            return true;
        }

        EntityKind Kind(const Json &value)
        {
            const int kind = int(Integer(value, 0, int(EntityKind::NUM_KINDS) - 1));
            DNDS_check_throw_info(kind >= 0 && kind < int(EntityKind::NUM_KINDS), "native archive entity kind is invalid");
            return EntityKind(kind);
        }

        Json Describe(const AdjKind &key)
        {
            key.Validate();
            std::vector<int> path;
            for (auto kind : key.path)
                path.push_back(int(kind));
            return Json{{"from", int(key.from)}, {"to", int(key.to)}, {"via", int(key.via)}, {"path", path}, {"selection", int(key.selection)}, {"remove_self", key.removeSelf}, {"order", int(key.order)}, {"periodic", int(key.periodic)}, {"name", key.semanticName}, {"version", key.semanticVersion}, {"provenance", key.provenance}};
        }

        AdjKind Definition(const Json &value)
        {
            DNDS_check_throw_info(value.is_object() && value.at("path").is_array(), "native archive relation definition is malformed");
            AdjKind result(Kind(value.at("from")), Kind(value.at("to")), Kind(value.at("via")));
            for (const auto &kind : value.at("path"))
                result.path.push_back(Kind(kind));
            const int selection = int(Integer(value.at("selection"), 0, int(RelationSelection::Custom))),
                      order = int(Integer(value.at("order"), 0, int(RelationOrder::SortedUnique))),
                      periodic = int(Integer(value.at("periodic"), 0, int(RelationPeriodicPolicy::FrameMatched)));
            DNDS_check_throw_info(selection >= 0 && selection <= int(RelationSelection::Custom) &&
                                      order >= 0 && order <= int(RelationOrder::SortedUnique) &&
                                      periodic >= 0 && periodic <= int(RelationPeriodicPolicy::FrameMatched),
                                  "native archive relation policy is invalid");
            result.selection = RelationSelection(selection);
            result.order = RelationOrder(order);
            result.periodic = RelationPeriodicPolicy(periodic);
            result.removeSelf = Boolean(value.at("remove_self"));
            result.semanticName = value.at("name").get<std::string>();
            const auto version = Integer(value.at("version"), 0, UINT32_MAX);
            DNDS_check_throw_info(version >= 0 && uint64_t(version) <= UINT32_MAX, "native archive relation version is invalid");
            result.semanticVersion = uint32_t(version);
            result.provenance = value.at("provenance").get<std::string>();
            result.Validate();
            return result;
        }

        template <class F>
        void Companions(NativeOwnedMeshData &data, F &&visit)
        {
            visit("coordinates", EntityKind::Node, data.coords, "float64");
            visit("node_wall_distance", EntityKind::Node, data.nodeWallDist, "float64");
            visit("cell_info", EntityKind::Cell, data.cellInfo, "Geom.ElemInfo.v1");
            visit("boundary_info", EntityKind::Bnd, data.bndInfo, "Geom.ElemInfo.v1");
            visit("face_info", EntityKind::Face, data.faceInfo, "Geom.ElemInfo.v1");
            visit("edge_info", EntityKind::Edge, data.edgeInfo, "Geom.ElemInfo.v1");
            visit("cell_nodes_pbi", EntityKind::Cell, data.cellNodesPbi, "Geom.NodePeriodicBits.v1");
            visit("boundary_nodes_pbi", EntityKind::Bnd, data.bndNodesPbi, "Geom.NodePeriodicBits.v1");
            visit("cell_faces_pbi", EntityKind::Cell, data.cellFacesPbi, "Geom.NodePeriodicBits.v1");
            visit("face_nodes_pbi", EntityKind::Face, data.faceNodesPbi, "Geom.NodePeriodicBits.v1");
            visit("cell_edges_pbi", EntityKind::Cell, data.cellEdgesPbi, "Geom.NodePeriodicBits.v1");
            visit("edge_nodes_pbi", EntityKind::Edge, data.edgeNodesPbi, "Geom.NodePeriodicBits.v1");
        }

        index Width(const R::RowLayout &layout)
        {
            DNDS_check_throw_info(layout.IsVariable() || layout.Width() <= size_t(INT64_MAX), "native archive row width exceeds index range");
            return layout.IsVariable() ? -1 : index(layout.Width());
        }

        Json Metadata(NativeOwnedMeshData &data, const MPIInfo &mpi, bool perRank)
        {
            const auto &q = data.requirements;
            Json result{{"format", "DNDSR.NativeMesh"}, {"version", 1}, {"writer_ranks", mpi.size}, {"per_rank", perRank}, {"dimension", data.dimension}, {"periodic", data.periodic}, {"namespace", data.identity.meshNamespace}, {"requirements", {q.faces, q.inverse, q.edges, q.cellFaceNeighbors, q.wallDistance, q.allFaceCellsLocal}}};
            result["families"] = Json::array();
            for (const auto &[kind, space] : data.spaces)
                result["families"].push_back({{"kind", int(kind)}, {"global_rows", space.NumGlobal()}, {"next_id", data.identity.nextIds.at(kind)}});
            result["ancestry"] = nullptr;
            if (data.provenance)
            {
                (void)data.provenance->Signature();
                result["ancestry"] = {{"source_namespace", data.provenance->sourceNamespace}, {"families", Json::array()}};
                for (const auto &[kind, values] : data.provenance->sourceIds)
                    result["ancestry"]["families"].push_back({{"kind", int(kind)}, {"source_kind", int(data.provenance->sourceKinds.at(kind))}, {"width", -1}});
            }
            result["fixed"] = Json::array();
            for (const auto &rows : data.relations)
                result["fixed"].push_back(rows ? Json(Width(rows->Layout())) : Json());
            result["named"] = Json::array();
            for (const auto &[key, rows] : data.namedRelations)
                result["named"].push_back({{"definition", Describe(key)}, {"width", Width(rows.Layout())}});
            result["companions"] = Json::object();
            Companions(data, [&](const char *name, EntityKind kind, const auto &rows, const char *schema)
                       {
                if (rows) result["companions"][name] = {{"kind", int(kind)}, {"width", Width(rows->Layout())}, {"schema", schema}}; });
            result["transforms"] = Json::array();
            for (size_t image = 0; image < 4; ++image)
            {
                std::vector<real> values;
                for (size_t j = 0; j < 9; ++j)
                    values.push_back(data.periodicInfo.rotation[image].map().data()[j]);
                for (size_t j = 0; j < 3; ++j)
                    values.push_back(data.periodicInfo.translation[image].map()[j]);
                for (size_t j = 0; j < 3; ++j)
                    values.push_back(data.periodicInfo.rotationCenter[image].map()[j]);
                for (auto value : values)
                    DNDS_check_throw_info(std::isfinite(value), "native archive transform is not finite");
                result["transforms"].push_back(values);
            }
            return result;
        }

        template <class T>
        R::Array<T> ReadRows(S::SerializerBaseSSP serializer, const std::string &name, const MPIInfo &mpi,
                             S::ArrayGlobalOffset partition, index width, const std::string &schema)
        {
            DNDS_check_throw_info(width >= -1, "native archive uniform width is invalid");
            auto array = R::ReadArray<T>(serializer, name, mpi, partition, R::Device::Host(), schema);
            if (width >= 0)
            {
                // The enclosing read stage agrees after both I/O and shape
                // checks. No nested collective here: a per-rank codec error
                // must not skip a collective entered by successful peers.
                for (size_t row = 0; row < array.Rows(); ++row)
                    DNDS_check_throw_info(array.RowSize(row) == size_t(width), "native archive uniform row shape mismatch");
                DNDS_check_throw_info(array.Layout().Elements() == CheckedSize::Multiply(array.Rows(), size_t(width)), "native archive uniform storage extent mismatch");
                return R::Array<T>::FromStorage(R::RowLayout::Dense(array.Rows(), size_t(width)), array.Storage());
            }
            return array;
        }

        void Arguments(const MPIInfo &mpi, const S::SerializerBaseSSP &serializer, const std::string &name)
        {
            DNDS_check_throw_info(mpi.comm != MPI_COMM_NULL, "native archive requires a communicator");
            ArchiveStage(mpi, [&]
                         {
                int rank = -1, size = 0;
                MPI_Comm_rank(mpi.comm, &rank);
                MPI_Comm_size(mpi.comm, &size);
                DNDS_check_throw_info(serializer && !name.empty() && mpi.rank == rank && mpi.size == size,
                                      "native archive arguments or MPIInfo are invalid");
                if (!serializer->IsPerRank())
                {
                    const auto &serializerMPI = serializer->getMPI();
                    DNDS_check_throw_info(serializerMPI.comm != MPI_COMM_NULL && serializerMPI.rank == rank && serializerMPI.size == size,
                                          "native archive serializer MPIInfo differs from its caller");
                    int comparison = MPI_UNEQUAL;
                    MPI_Comm_compare(serializerMPI.comm, mpi.comm, &comparison);
                    DNDS_check_throw_info(comparison == MPI_IDENT || comparison == MPI_CONGRUENT,
                                          "native archive serializer communicator differs from its caller");
                } });
            Agree(mpi, name + ":" + std::to_string(serializer->IsPerRank()));
        }
    }

    void UnstructuredMesh::WriteNativeArchive(S::SerializerBaseSSP serializer, const std::string &name) const
    {
        DNDS_check_throw_info(mpi.comm != MPI_COMM_NULL, "native archive requires a communicator");
        ArchiveStage(mpi, [&]
                     { DNDS_check_throw_info(_nativeState, "publish native mesh state before native archive write"); });
        WriteNativeMeshArchive(*_nativeState, mpi, std::move(serializer), name);
    }

    std::shared_ptr<UnstructuredMesh> UnstructuredMesh::ReadNativeArchive(
        const MPIInfo &mpi, S::SerializerBaseSSP serializer, const std::string &name, S::ArrayGlobalOffset partition, int ghostLayers)
    {
        return FromNative(mpi, ReadNativeMeshArchive(mpi, std::move(serializer), name, partition, ghostLayers));
    }

    void WriteNativeMeshArchive(const NativeMeshState &source, const MPIInfo &mpi,
                                S::SerializerBaseSSP serializer, const std::string &name)
    {
        Arguments(mpi, serializer, name);
        auto owned = NativeMeshAssembler::CaptureOwned(source, mpi);
        Json metadata;
        std::string encoded;
        ArchiveStage(mpi, [&]
                     { metadata = Metadata(owned, mpi, serializer->IsPerRank()); encoded = metadata.dump(); });
        Agree(mpi, encoded);
        size_t familyNumber = 0;
        for (const auto &[kind, space] : owned.spaces)
        {
            std::vector<index> counts;
            ArchiveStage(mpi, [&]
                         { counts.resize(size_t(mpi.size)); });
            const index local = space.NumOwned();
            MPI_Allgather(&local, 1, DNDS_MPI_INDEX, counts.data(), 1, DNDS_MPI_INDEX, mpi.comm);
            ArchiveStage(mpi, [&]
                         {
                index sum = 0;
                for (auto count : counts)
                {
                    DNDS_check_throw_info(count >= 0 && count <= INT64_MAX - sum, "native archive writer row counts overflow");
                    sum += count;
                }
                DNDS_check_throw_info(sum == space.NumGlobal(), "native archive writer row counts disagree with its entity space");
                metadata["families"][familyNumber]["writer_rows"] = counts; });
            ++familyNumber;
        }
        ArchiveStage(mpi, [&]
                     { encoded = metadata.dump(); });
        Agree(mpi, encoded);
        R::Detail::SerializerPath guard(serializer);
        ArchiveStage(mpi, [&]
                     {
            serializer->CreatePath(name);
            serializer->GoToPath(name);
            serializer->WriteString("metadata_json", encoded);
            if (serializer->IsPerRank()) serializer->WriteInt("writer_rank", mpi.rank); });
        for (const auto &[kind, ids] : owned.ids)
            ArchiveStage(mpi, [&]
                         { R::WriteArray(ids, serializer, "ids_" + std::to_string(int(kind)), mpi); });
        if (owned.provenance)
            for (const auto &[kind, values] : owned.provenance->sourceIds)
                ArchiveStage(mpi, [&]
                             { R::WriteArray(values, serializer, "ancestry_" + std::to_string(int(kind)), mpi); });
        for (size_t slot = 0; slot < owned.relations.size(); ++slot)
            if (owned.relations[slot])
                ArchiveStage(mpi, [&]
                             { R::WriteArray(*owned.relations[slot], serializer, "fixed_" + std::to_string(slot), mpi); });
        size_t number = 0;
        for (const auto &[key, rows] : owned.namedRelations)
        {
            ArchiveStage(mpi, [&]
                         { R::WriteArray(rows, serializer, "named_" + std::to_string(number), mpi); });
            ++number;
        }
        Companions(owned, [&](const char *field, EntityKind, const auto &rows, const char *schema)
                   {
            if (rows) ArchiveStage(mpi, [&] { R::WriteArray(*rows, serializer, field, mpi, schema); }); });
    }

    std::shared_ptr<const NativeMeshState> ReadNativeMeshArchive(const MPIInfo &mpi, S::SerializerBaseSSP serializer,
                                                                 const std::string &name, S::ArrayGlobalOffset partition, int ghostLayers)
    {
        Arguments(mpi, serializer, name);
        ArchiveStage(mpi, [&]
                     { DNDS_check_throw_info(ghostLayers >= 1 && (partition == S::ArrayGlobalOffset_Unknown || partition == S::ArrayGlobalOffset_EvenSplit) &&
                                                 (!serializer->IsPerRank() || partition == S::ArrayGlobalOffset_Unknown),
                                             "native archive requires matching or direct even ownership partitions"); });
        Agree(mpi, std::to_string(partition.offset()) + ":" + std::to_string(ghostLayers));
        R::Detail::SerializerPath guard(serializer);
        std::string encoded;
        ArchiveStage(mpi, [&]
                     {
            serializer->GoToPath(name);
            serializer->ReadString("metadata_json", encoded);
            if (serializer->IsPerRank())
            {
                index rank = -1;
                serializer->ReadIndex("writer_rank", rank);
                DNDS_check_throw_info(rank == mpi.rank, "per-rank native archive belongs to another rank");
            } });
        Agree(mpi, encoded);
        Json metadata;
        NativeOwnedMeshData owned;
        std::map<EntityKind, index> totals;
        std::map<EntityKind, std::vector<index>> writerRows;
        std::vector<AdjKind> named;
        std::map<EntityKind, index> provenanceWidths;
        ArchiveStage(mpi, [&]
                     {
            metadata = Json::parse(encoded);
            DNDS_check_throw_info(metadata.is_object() && metadata.at("format") == "DNDSR.NativeMesh" &&
                                      Integer(metadata.at("version"), 1, 1) == 1 &&
                                      Boolean(metadata.at("per_rank")) == serializer->IsPerRank(),
                                  "native archive format or serializer mode mismatch");
            const auto writerRanks = Integer(metadata.at("writer_ranks"), 1, INT_MAX);
            DNDS_check_throw_info(!(partition == S::ArrayGlobalOffset_Unknown) || writerRanks == mpi.size,
                                  "native archive writer partition mismatch");
            owned.dimension = int(Integer(metadata.at("dimension"), 1, 3));
            owned.periodic = Boolean(metadata.at("periodic"));
            owned.identity.meshNamespace = metadata.at("namespace").get<std::string>();
            DNDS_check_throw_info(ValidNamespace(owned.identity.meshNamespace), "native archive mesh namespace is invalid");
            const auto &flags = metadata.at("requirements");
            DNDS_check_throw_info(flags.is_array() && flags.size() == 6, "native archive requirements are malformed");
            owned.requirements = {Boolean(flags[0]), Boolean(flags[1]), Boolean(flags[2]), Boolean(flags[3]), Boolean(flags[4]), Boolean(flags[5])};
            DNDS_check_throw_info(metadata.at("families").is_array(), "native archive family inventory is malformed");
            for (const auto &family : metadata.at("families"))
            {
                DNDS_check_throw_info(family.is_object() && family.at("writer_rows").is_array() &&
                                          family.at("writer_rows").size() == size_t(writerRanks), "native archive writer row inventory is malformed");
                const auto kind = Kind(family.at("kind"));
                const index total = Integer(family.at("global_rows")), next = Integer(family.at("next_id"));
                DNDS_check_throw_info(totals.emplace(kind, total).second, "native archive entity inventory is duplicated");
                std::vector<index> counts;
                index sum = 0;
                for (const auto &value : family.at("writer_rows"))
                {
                    const index count = Integer(value);
                    DNDS_check_throw_info(count <= INT64_MAX - sum, "native archive writer row counts overflow");
                    sum += count;
                    counts.push_back(count);
                }
                DNDS_check_throw_info(sum == total, "native archive writer row sum differs from global family rows");
                writerRows.emplace(kind, std::move(counts));
                owned.identity.nextIds.emplace(kind, next);
            }
            DNDS_check_throw_info(totals.count(EntityKind::Node) && totals.count(EntityKind::Cell) && totals.count(EntityKind::Bnd), "native archive primary families are missing");
            DNDS_check_throw_info(metadata.at("fixed").is_array() && metadata.at("fixed").size() == owned.relations.size(), "native archive fixed relation inventory is malformed");
            for (const auto &width : metadata.at("fixed")) if (!width.is_null()) (void)Integer(width, -1);
            DNDS_check_throw_info(metadata.at("named").is_array() && metadata.at("companions").is_object(), "native archive relation/companion inventory is malformed");
            std::set<AdjKind> distinct;
            for (const auto &relation : metadata.at("named"))
            {
                DNDS_check_throw_info(relation.is_object(), "native archive named relation is malformed");
                (void)Integer(relation.at("width"), -1);
                auto key = Definition(relation.at("definition"));
                DNDS_check_throw_info(totals.count(key.from) && totals.count(key.to) && distinct.insert(key).second, "native archive named definition is unbound or duplicated");
                named.push_back(std::move(key));
            }
            if (metadata.contains("ancestry") && !metadata.at("ancestry").is_null())
            {
                const auto &ancestry = metadata.at("ancestry");
                DNDS_check_throw_info(ancestry.is_object() && ancestry.at("families").is_array() && !ancestry.at("families").empty(),
                                      "native archive ancestry inventory is malformed");
                owned.provenance.emplace();
                owned.provenance->sourceNamespace = ancestry.at("source_namespace").get<std::string>();
                DNDS_check_throw_info(ValidNamespace(owned.provenance->sourceNamespace), "native archive ancestry namespace is invalid");
                for (const auto &family : ancestry.at("families"))
                {
                    const auto kind = Kind(family.at("kind")), sourceKind = Kind(family.at("source_kind"));
                    const auto width = Integer(family.at("width"), -1);
                    DNDS_check_throw_info(totals.count(kind) && provenanceWidths.emplace(kind, width).second,
                                          "native archive ancestry family is unbound or duplicated");
                    owned.provenance->sourceKinds.emplace(kind, sourceKind);
                }
            }
            size_t recognized = 0;
            Companions(owned, [&](const char *field, EntityKind kind, auto &, const char *schema)
            {
                if (!metadata.at("companions").contains(field)) return;
                ++recognized;
                const auto &description = metadata.at("companions").at(field);
                DNDS_check_throw_info(description.is_object(), "native archive companion description is malformed");
                (void)Integer(description.at("width"), -1);
                DNDS_check_throw_info(Kind(description.at("kind")) == kind && description.at("schema") == schema && totals.count(kind), "native archive companion type/family mismatch");
            });
            DNDS_check_throw_info(recognized == metadata.at("companions").size(), "native archive contains an unknown companion");
            const auto &transforms = metadata.at("transforms");
            DNDS_check_throw_info(transforms.is_array() && transforms.size() == 4, "native archive transforms are malformed");
            for (size_t image = 0; image < 4; ++image)
            {
                DNDS_check_throw_info(transforms[image].is_array(), "native archive transform row is malformed");
                for (const auto &value : transforms[image]) DNDS_check_throw_info(value.is_number(), "native archive transform entry is not numeric");
                const auto values = transforms[image].get<std::vector<real>>();
                DNDS_check_throw_info(values.size() == 15 && std::all_of(values.begin(), values.end(), [](real x) { return std::isfinite(x); }), "native archive transform shape/value mismatch");
                for (size_t j = 0; j < 9; ++j) owned.periodicInfo.rotation[image].map().data()[j] = values[j];
                for (size_t j = 0; j < 3; ++j) owned.periodicInfo.translation[image].map()[j] = values[9 + j];
                for (size_t j = 0; j < 3; ++j) owned.periodicInfo.rotationCenter[image].map()[j] = values[12 + j];
            } });
        for (const auto &[kind, total] : totals)
            ArchiveStage(mpi, [&]
                         { owned.ids.emplace(kind, ReadRows<index>(serializer, "ids_" + std::to_string(int(kind)), mpi, partition, 1, R::ArraySchema<index>())); });
        for (const auto &[kind, width] : provenanceWidths)
            ArchiveStage(mpi, [&]
                         {
                auto values = ReadRows<index>(serializer, "ancestry_" + std::to_string(int(kind)), mpi, partition, width, R::ArraySchema<index>());
                DNDS_check_throw_info(values.Rows() == owned.ids.at(kind).Rows(), "native archive ancestry partition differs from its entity family");
                owned.provenance->sourceIds.emplace(kind, std::move(values)); });
        std::map<EntityKind, index> localMaxima;
        ArchiveStage(mpi, [&]
                     {
            for (const auto &[kind, ids] : owned.ids)
            {
                const index total = totals.at(kind), remainder = total % mpi.size;
                // Avoid rank*total overflow when rejecting malicious counts.
                const index evenRows = total / mpi.size + index(mpi.rank + 1) * remainder / mpi.size - index(mpi.rank) * remainder / mpi.size;
                const index expectedRows = partition == S::ArrayGlobalOffset_Unknown ? writerRows.at(kind).at(size_t(mpi.rank)) : evenRows;
                DNDS_check_throw_info(ids.Rows() == size_t(expectedRows), "native archive restored family partition differs from saved ownership");
                auto values = ids.Read();
                index maximum = -1;
                for (size_t row = 0; row < ids.Rows(); ++row)
                {
                    const auto value = values.View()(row, 0);
                    DNDS_check_throw_info(value >= 0 && value < INT64_MAX, "native archive persistent ID is invalid or exhausted");
                    maximum = std::max(maximum, value);
                }
                localMaxima.emplace(kind, maximum);
            } });
        owned.identity = PrepareMeshIdentity(mpi, std::move(owned.identity), localMaxima);
        const auto &nodeIds = owned.ids.at(EntityKind::Node);
        auto nodes = EntitySpace::CreateOwned(EntityKind::Node, mpi, index(nodeIds.Rows()));
        owned.spaces.emplace(EntityKind::Node, nodes);
        for (const auto &[kind, ids] : owned.ids)
            if (kind != EntityKind::Node)
                owned.spaces.emplace(kind, nodes.CreateSibling(kind, index(ids.Rows())));
        ArchiveStage(mpi, [&]
                     {
            for (const auto &[kind, space] : owned.spaces)
                DNDS_check_throw_info(space.NumGlobal() == totals.at(kind), "native archive family global row count mismatch"); });
        for (size_t slot = 0; slot < owned.relations.size(); ++slot)
            if (!metadata["fixed"][slot].is_null())
                ArchiveStage(mpi, [&]
                             { owned.relations[slot] = ReadRows<index>(serializer, "fixed_" + std::to_string(slot), mpi, partition, Integer(metadata["fixed"][slot], -1), R::ArraySchema<index>()); });
        for (size_t number = 0; number < named.size(); ++number)
            ArchiveStage(mpi, [&]
                         { owned.namedRelations.emplace(named[number], ReadRows<index>(serializer, "named_" + std::to_string(number), mpi, partition, Integer(metadata["named"][number].at("width"), -1), R::ArraySchema<index>())); });
        Companions(owned, [&](const char *field, EntityKind, auto &rows, const char *schema)
                   {
            if (!metadata["companions"].contains(field)) return;
            using Array = typename std::decay_t<decltype(rows)>::value_type;
            using T = std::remove_const_t<std::remove_reference_t<decltype(std::declval<Array>().Read().View()(0, 0))>>;
            ArchiveStage(mpi, [&] { rows = ReadRows<T>(serializer, field, mpi, partition, Integer(metadata["companions"][field].at("width"), -1), schema); }); });
        return NativeMeshAssembler::Build(mpi, std::move(owned), ghostLayers);
    }
}
