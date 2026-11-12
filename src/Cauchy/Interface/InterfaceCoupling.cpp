#include "Cauchy/Interface/InterfaceCoupling.hpp"
#include "Cauchy/Interface/CPU/InterfacePacketKernels.hpp"
#include "Cauchy/Interface/CPU/InterfaceSamplingKernels.hpp"
#include "CFV/Plans/DistributedInterfaceSamplingPlan.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include <limits>
#include <mutex>
#include <nlohmann/json.hpp>

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    namespace
    {
        using SchemaJson = nlohmann::ordered_json;

        std::string DescribeSchema(const EquationSchema &schema)
        {
            SchemaJson blocks = SchemaJson::array();
            for (const auto &block : schema.blocks)
            {
                SchemaJson components = SchemaJson::array();
                for (const auto &component : block.components)
                    components.push_back(
                        {{"name", component.name},
                         {"units", component.units},
                         {"conservation_group", component.conservationGroup}});
                blocks.push_back({{"name", block.name},
                                  {"components", std::move(components)},
                                  {"vector_starts", block.vectorStarts}});
            }
            return SchemaJson{{"family", schema.family},
                              {"energy_convention", schema.energyConvention},
                              {"physical_accumulation", schema.physicalAccumulation},
                              {"blocks", std::move(blocks)}}
                .dump();
        }

        EquationSchema ParseSchema(const std::string &description)
        {
            const auto json = SchemaJson::parse(description);
            EquationSchema schema;
            schema.family = json.at("family").get<std::string>();
            schema.energyConvention =
                json.at("energy_convention").get<std::string>();
            schema.physicalAccumulation =
                json.at("physical_accumulation").get<std::string>();
            for (const auto &encodedBlock : json.at("blocks"))
            {
                EquationBlockSchema block;
                block.name = encodedBlock.at("name").get<std::string>();
                block.vectorStarts =
                    encodedBlock.at("vector_starts").get<std::vector<index>>();
                for (const auto &encodedComponent :
                     encodedBlock.at("components"))
                    block.components.push_back(
                        {encodedComponent.at("name").get<std::string>(),
                         encodedComponent.at("units").get<std::string>(),
                         encodedComponent.at("conservation_group")
                             .get<std::string>()});
                schema.blocks.push_back(std::move(block));
            }
            return schema;
        }

        void BroadcastString(std::string &value, int root, MPI_Comm communicator)
        {
            index size = index(value.size());
            MPI::Bcast(&size, 1, MPI_INT64_T, root, communicator);
            DNDS_check_throw_info(
                size >= 0 && size <= std::numeric_limits<int>::max(),
                "interface schema descriptor is too large");
            value.resize(size_t(size));
            if (size)
                MPI::Bcast(value.data(), int(size), MPI_CHAR, root,
                           communicator);
        }
    }
    struct InterfaceCoupling::Data
    {
        struct Connection
        {
            struct DonorField
            {
                R::MetadataBuffer<index> logicalComponents;
                std::optional<CFV::DistributedSamplingWorkspace> workspace;
                R::Array<real> values;
                std::optional<R::TransferBinding> result;
            };
            index donor, receiver;
            InterfaceSettings settings;
            std::shared_ptr<R::CommunicationContext> context;
            std::optional<Geom::BoundaryPatch> patch;
            // A single distributed match/sample batch, with explicit output
            // destinations in the quadrature and face-centered packet tables.
            R::Array<CPU::InterfacePacketRow> receiverRows{R::RowLayout::Dense(0, 1)};
            R::Array<real> receiverReferencePoints{R::RowLayout::Dense(0, 3)};
            std::optional<std::pair<Geom::RigidPose, Geom::RigidPose>> poses;
            std::optional<CFV::DistributedInterfaceSamplingPlan> sampling;
            std::vector<DonorField> fields;
            /// Collective, immutable donor physical grouping: mode, logical
            /// component count, then one logical-component map per field.
            std::vector<index> donorDescriptor;
            /// Exact logical schema and donor-ordinal to receiver-ordinal map.
            /// The latter is broadcast from a real receiver rank, so all
            /// coupling ranks share one setup-time compatibility decision.
            std::string donorSchema;
            R::MetadataBuffer<index> receiverComponents;
            R::MetadataBuffer<index> vectorStarts;
            // Initial history is empty but retains the same fixed width as
            // every later publication, including nonmember/zero-query ranks.
            R::Array<index> lastDonors{R::RowLayout::Dense(0, 1)};
            uint64_t mapGeneration = 0;
        };
        mutable std::mutex operationMutex;
        // One controller operation at a time; do not wait behind an MPI path.
        // Commit/discard remain allocation-free, nonthrowing publication.
        std::unique_lock<std::mutex> Enter(bool publication = false) const
        {
            std::unique_lock<std::mutex> lock(operationMutex, std::try_to_lock);
            if (publication)
                DNDS_assert_info(lock.owns_lock(), "concurrent interface publication");
            else
                DNDS_check_throw_info(lock.owns_lock(), "interface coupling already has an active operation");
            return lock;
        }
        std::shared_ptr<R::CommunicationContext> control;
        MPIInfo mpi;
        std::vector<std::optional<FiniteVolume>> discrete;
        std::vector<std::optional<Physics::Chemistry::ChemicalWorkerPool>> workers;
        std::vector<RigidMotion> motion;
        std::vector<std::optional<BoundaryInputs>> incoming;
        std::vector<std::optional<BoundaryInputs>> accepted;
        bool useAccepted = false;
        std::vector<Connection> connections;
        index changedDonors = 0;
        uint64_t mapBuilds = 0;
        real maximumGap = 0;
        explicit Data(std::shared_ptr<R::CommunicationContext> owner) : control(std::move(owner))
        {
            DNDS_check_throw_info(control, "interface coupling requires an owning communication context");
            mpi = MPIInfo(control->Communicator());
        }
    };
    InterfaceCoupling::InterfaceCoupling(
        const SimulationConfig &config,
        std::shared_ptr<R::CommunicationContext> context,
        const std::vector<std::optional<FiniteVolume>> &localDomains,
        const std::vector<std::optional<Physics::Chemistry::ChemicalWorkerPool>> &workers)
        : _data(std::make_shared<Data>(std::move(context)))
    {
        auto &data = *_data;
        const auto coupling = data.mpi.comm;
        MPI::LocalStage(data.mpi, [&]
                        {
            DNDS_check_throw_info(localDomains.size()==config.Domains().size(),"interface domain descriptor count differs");
            data.discrete.resize(localDomains.size());
            data.workers = workers;
            if (data.workers.empty())
                data.workers.resize(localDomains.size());
            DNDS_check_throw_info(data.workers.size() == localDomains.size(),
                                  "interface chemistry-worker descriptor count differs");
            data.incoming.resize(localDomains.size());
            data.accepted.resize(localDomains.size());
            for(const auto &domain:config.Domains()) data.motion.emplace_back(domain.settings.motion,domain.settings.mesh.dimension);
            for(index iDomain=0;iDomain<index(localDomains.size());++iDomain)
                if(localDomains[iDomain])
                {
                    data.discrete[iDomain]=*localDomains[iDomain];
                    if(localDomains[iDomain]->HasInterfaces())
                        data.incoming[iDomain]=localDomains[iDomain]->BuildBoundaryInputs(data.workers[iDomain]);
                } });
        for (const auto &settings : config.Interfaces())
        {
            Data::Connection connection;
            connection.settings = settings;
            const auto domainIndex = [&](const std::string &name)
            {
                const auto found = std::find_if(config.Domains().begin(), config.Domains().end(), [&](const auto &domain)
                                                { return domain.placement.name == name; });
                return index(found - config.Domains().begin());
            };
            connection.donor = domainIndex(settings.donor);
            connection.receiver = domainIndex(settings.receiver);
            DNDS_check_throw_info(
                config.Root().at("domains").at(size_t(connection.donor)).at("settings").at("equation") ==
                    config.Root().at("domains").at(size_t(connection.receiver)).at("settings").at("equation"),
                "interface donor and receiver need identical equation, material and scaling settings");
            // Independent connection protocols intentionally isolate their
            // query/result namespaces; this is not a lifetime-only duplication.
            connection.context = R::CommunicationContext::Duplicate(coupling);
            // Patch construction is domain-collective. A local exception must
            // propagate to the fatal boundary, not to a coupling reduction
            // while another donor rank is still constructing the patch.
            {
                if (auto &fv = data.discrete[connection.donor])
                    connection.patch = fv->BuildBoundaryPatch(settings.donorZone);
                if (auto &fv = data.discrete[connection.receiver])
                {
                    std::vector<CPU::InterfacePacketRow> selectedRows;
                    const auto pointRead = fv->ReadBoundaryPoints();
                    const auto points = pointRead.View<R::FixedRows<1>>();
                    for (index iPoint = 0; iPoint < points.RowCount(); ++iPoint)
                        if (points(iPoint, 0).interface && points(iPoint, 0).zone == settings.receiverZone)
                            selectedRows.push_back({iPoint, false});
                    const auto faceRead = fv->ReadBoundaryFaces();
                    const auto faces = faceRead.View<R::FixedRows<1>>();
                    for (index iFace = 0; iFace < faces.RowCount(); ++iFace)
                        if (faces(iFace, 0).interface && faces(iFace, 0).zone == settings.receiverZone)
                            selectedRows.push_back({iFace, true});
                    connection.receiverRows = R::Array<CPU::InterfacePacketRow>(
                        R::RowLayout::Dense(selectedRows.size(), 1));
                    connection.receiverReferencePoints = R::Array<real>(
                        R::RowLayout::Dense(selectedRows.size(), 3));
                    auto write = connection.receiverRows.Write();
                    auto pointWrite = connection.receiverReferencePoints.Write();
                    const auto rows = write.View<R::FixedRows<1>>();
                    const auto locations = pointWrite.View<R::FixedRows<3>>();
                    for (index iRow = 0; iRow < rows.RowCount(); ++iRow)
                    {
                        rows(iRow, 0) = selectedRows[size_t(iRow)];
                        const auto row = selectedRows[size_t(iRow)];
                        const auto point = row.faceCenter
                                               ? faces(row.row, 0).point
                                               : points(row.row, 0).point;
                        for (int d = 0; d < 3; ++d)
                            locations(iRow, d) = point[size_t(d)];
                    }
                    std::vector<index> vectors;
                    index first = 0;
                    for (const auto &block : fv->Equation().Schema().blocks)
                    {
                        for (const auto start : block.vectorStarts)
                            vectors.push_back(first + start);
                        first += index(block.components.size());
                    }
                    connection.vectorStarts = R::MetadataBuffer<index>(vectors);
                }
            }
            index counts[2]{connection.patch ? connection.patch->Size() : 0, connection.receiverRows.RowCount()};
            MPI::AllreduceInPlace<MPI::Reduction::Sum>(
                MPI::ContiguousView<index>{counts, 2}, coupling);
            DNDS_check_throw_info(counts[0] > 0 && counts[1] > 0, "interface donor patch or receiver boundary quadrature is globally empty; check zone IDs");
            data.connections.push_back(std::move(connection));
        }
    }
    std::optional<BoundaryInputs> InterfaceCoupling::Inputs(index domain) const
    {
        auto session = _data->Enter();
        DNDS_check_throw_info(domain >= 0 && domain < index(_data->incoming.size()), "invalid interface receiver domain");
        const auto &packets = _data->useAccepted ? _data->accepted : _data->incoming;
        return packets[domain];
    }
    void InterfaceCoupling::Commit() noexcept
    {
        auto session = _data->Enter(true);
        _data->incoming.swap(_data->accepted);
        _data->useAccepted = true;
    }
    void InterfaceCoupling::Discard() noexcept
    {
        auto session = _data->Enter(true);
        _data->useAccepted = true;
    }
    InterfaceDiagnostics InterfaceCoupling::Diagnostics() const
    {
        auto session = _data->Enter();
        InterfaceDiagnostics result;
        for (const auto &connection : _data->connections)
            result.queryPoints += connection.receiverRows.RowCount();
        result.changedDonors = _data->changedDonors;
        index counts[2]{result.queryPoints, result.changedDonors};
        MPI::AllreduceInPlace<MPI::Reduction::Sum>(
            MPI::ContiguousView<index>{counts, 2}, _data->mpi.comm);
        result.queryPoints = counts[0];
        result.changedDonors = counts[1];
        result.maximumGap = _data->maximumGap;
        result.maximumGap = MPI::AllreduceValue<MPI::Reduction::Max>(
            result.maximumGap, _data->mpi.comm);
        result.mapBuilds = _data->mapBuilds;
        return result;
    }
    void InterfaceCoupling::Refresh(const std::vector<InterfaceDomainInput> &domains, real time, StateEpoch epoch)
    {
        auto session = _data->Enter();
        auto &data = *_data;
        data.useAccepted = false;
        std::vector<std::vector<BoundaryInterfaceSource>> sources(
            data.discrete.size());
        MPI::LocalStage(data.mpi, [&]
                        {
            DNDS_check_throw_info(domains.size()==data.discrete.size()&&std::isfinite(time),"invalid interface refresh descriptors/time");
            for(index iDomain=0;iDomain<index(domains.size());++iDomain)
            {
                const auto &input=domains[iDomain];
                DNDS_check_throw_info(bool(input.values)==bool(data.discrete[iDomain]),
                                      "interface refresh has incomplete or changed domain participation");
                if (input.values)
                    input.values->Validate();
                if(data.discrete[iDomain]&&data.discrete[iDomain]->HasInterfaces())
                {
                    if(!data.incoming[iDomain])
                        data.incoming[iDomain]=data.discrete[iDomain]->BuildBoundaryInputs(data.workers[iDomain]);
                    else data.discrete[iDomain]->BeginBoundaryUpdate(*data.incoming[iDomain]);
                }
            } });
        for (auto &connection : data.connections)
        {
            auto &donor = data.discrete[connection.donor];
            auto &receiver = data.discrete[connection.receiver];
            const auto donorPose = data.motion[connection.donor].At(time).pose;
            const auto receiverPose = data.motion[connection.receiver].At(time).pose;
            const auto samePose = [](const auto &a, const auto &b)
            { return a.rotation == b.rotation && a.translation == b.translation; };
            const auto describe = [](const FiniteVolume::InterfaceDonorPublication &publication)
            {
                const auto &values = publication._means;
                std::vector<std::vector<index>> logical(
                    size_t(values.PhysicalFieldNumber()));
                for (index iField = 0; iField < values.PhysicalFieldNumber();
                     ++iField)
                    logical[size_t(iField)].assign(
                        values.PhysicalField(iField).Width(), -1);
                index first = 0;
                for (index iBlock = 0; iBlock < values.BlockNumber(); ++iBlock)
                {
                    const auto location = values.LocateBlock(iBlock);
                    for (index j = 0; j < location.componentNumber; ++j)
                        logical[size_t(location.physicalField)]
                               [size_t(location.firstComponent + j)] = first + j;
                    first += location.componentNumber;
                }
                std::vector<index> result{first, values.PhysicalFieldNumber()};
                for (const auto &field : logical)
                    result.push_back(index(field.size()));
                for (const auto &field : logical)
                    result.insert(result.end(), field.begin(), field.end());
                return result;
            };
            const auto &publication = domains[connection.donor].values;
            std::vector<index> localDescriptor;
            if (publication)
                localDescriptor = describe(*publication);
            int descriptorRoot = publication ? data.mpi.rank : data.mpi.size;
            MPI::Allreduce(MPI_IN_PLACE, &descriptorRoot, 1, MPI_INT, MPI_MIN,
                           data.mpi.comm);
            DNDS_check_throw_info(descriptorRoot >= 0 && descriptorRoot < data.mpi.size,
                                  "interface connection has no donor publication");
            // Resolve semantic component compatibility once, independently of
            // physical field grouping. Same-width equations are not sufficient:
            // species names, units, conservation groups and vector-frame roles
            // must match. The exact equation/material configuration was already
            // required equal in the constructor; this runtime map proves that
            // the published schema still matches the receiver.
            std::string localSchema;
            if (publication)
                localSchema = DescribeSchema(publication->Schema());
            if (connection.receiverComponents.empty())
            {
                std::string donorSchema = data.mpi.rank == descriptorRoot
                                              ? localSchema
                                              : std::string{};
                BroadcastString(donorSchema, descriptorRoot, data.mpi.comm);
                MPI::LocalStage(data.mpi, [&]
                                {
                    if(publication)
                        DNDS_check_throw_info(localSchema==donorSchema,
                                              "interface donor ranks publish different equation schemas"); });
                EquationSchema schema;
                MPI::LocalStage(data.mpi, [&]
                                { schema = ParseSchema(donorSchema); });
                int receiverRoot = receiver ? data.mpi.rank : data.mpi.size;
                MPI::Allreduce(MPI_IN_PLACE, &receiverRoot, 1, MPI_INT, MPI_MIN,
                               data.mpi.comm);
                DNDS_check_throw_info(
                    receiverRoot >= 0 && receiverRoot < data.mpi.size,
                    "interface connection has no receiver equation schema");
                std::vector<index> mapping;
                MPI::LocalStage(data.mpi, [&]
                                {
                    if(receiver)
                        mapping=schema.MapComponentsTo(receiver->Equation().Schema()); });
                index mappingSize = data.mpi.rank == receiverRoot
                                        ? index(mapping.size())
                                        : 0;
                MPI::Bcast(&mappingSize, 1, MPI_INT64_T, receiverRoot,
                           data.mpi.comm);
                DNDS_check_throw_info(
                    mappingSize > 0 &&
                        mappingSize <= std::numeric_limits<int>::max(),
                    "invalid interface equation component mapping size");
                if (data.mpi.rank != receiverRoot)
                    mapping.resize(size_t(mappingSize));
                MPI::Bcast(mapping.data(), int(mappingSize), MPI_INT64_T,
                           receiverRoot, data.mpi.comm);
                MPI::LocalStage(data.mpi, [&]
                                {
                    if(receiver)
                        DNDS_check_throw_info(
                            mapping==schema.MapComponentsTo(receiver->Equation().Schema()),
                            "interface receiver ranks resolve different equation mappings"); });
                connection.donorSchema = std::move(donorSchema);
                connection.receiverComponents = R::MetadataBuffer<index>(mapping);
            }
            else
                MPI::LocalStage(data.mpi, [&]
                                {
                    if(publication)
                        DNDS_check_throw_info(
                            localSchema==connection.donorSchema,
                            "interface donor equation schema changed after setup"); });
            int descriptorSize = data.mpi.rank == descriptorRoot
                                     ? int(localDescriptor.size())
                                     : 0;
            MPI::Bcast(&descriptorSize, 1, MPI_INT, descriptorRoot, data.mpi.comm);
            DNDS_check_throw_info(descriptorSize >= 2,
                                  "invalid interface donor field descriptor");
            std::vector<index> descriptor(static_cast<size_t>(descriptorSize),
                                          index{});
            if (data.mpi.rank == descriptorRoot)
                descriptor = localDescriptor;
            MPI::Bcast(descriptor.data(), descriptorSize, MPI_INT64_T,
                       descriptorRoot, data.mpi.comm);
            MPI::LocalStage(data.mpi, [&]
                            {
                if (publication)
                    DNDS_check_throw_info(
                        localDescriptor == descriptor,
                        "interface donor ranks use different physical field grouping");
                const index fields = descriptor[1];
                DNDS_check_throw_info(
                    fields > 0 && descriptor.size() >= size_t(2 + fields),
                    "invalid interface donor field count");
                index described = 0;
                for (index iField = 0; iField < fields; ++iField)
                {
                    const index width = descriptor[size_t(2 + iField)];
                    DNDS_check_throw_info(width > 0 &&
                                              described <= descriptor[0] - width,
                                          "invalid interface donor field width");
                    described += width;
                }
                DNDS_check_throw_info(
                    described == descriptor[0] &&
                        descriptor.size() == size_t(2 + fields + described),
                    "interface donor descriptor does not cover its equation");
                if (receiver)
                    DNDS_check_throw_info(
                        data.incoming[connection.receiver]
                                ->QuadratureValues()
                                .ComponentNumber() == descriptor[0] &&
                            index(connection.receiverComponents.size()) ==
                                descriptor[0],
                        "interface donor and receiver equation schemas differ");
                if (!connection.donorDescriptor.empty())
                    DNDS_check_throw_info(
                        connection.donorDescriptor == descriptor,
                        "interface donor physical grouping changed after setup"); });
            if (connection.donorDescriptor.empty())
                connection.donorDescriptor = descriptor;
            if (!connection.poses || !samePose(connection.poses->first, donorPose) || !samePose(connection.poses->second, receiverPose))
            {
                R::Array<real> points;
                R::Array<index> ids;
                std::optional<Geom::BoundaryPatch> patch;
                std::optional<CFV::FiniteVolumePlan> metrics;
                std::optional<CFV::VariationalReconstructionPlan> reconstruction;
                MPI::LocalStage(data.mpi, [&]
                                {
                    points=R::Array<real>(R::RowLayout::Dense(connection.receiverRows.RowCount(),3));
                    ids=R::Array<index>(R::RowLayout::Dense(connection.receiverRows.RowCount(),1));
                    if(donor)
                    {
                        patch=connection.patch->WithPose(donorPose);
                        metrics=donor->Metrics();
                        const auto polynomial=donor->PolynomialReconstruction();
                        if(polynomial) reconstruction=*polynomial;
                    }
                    if(receiver)
                    {
                        const auto frame=receiver->Motion().At(time);
                        auto p=points.Write(); auto key=ids.Write();
                        const auto reference = connection.receiverReferencePoints.Read();
                        CPU::InterfaceQueryRequest request;
                        request.referencePoints = reference.View<R::FixedRows<3>>();
                        request.worldPoints = p.View<R::FixedRows<3>>();
                        request.ids = key.View<R::FixedRows<1>>();
                        request.moving = frame.moving;
                        for (int d = 0; d < 3; ++d)
                        {
                            request.translation[size_t(d)] = frame.pose.translation[d];
                            for (int e = 0; e < 3; ++e)
                                request.rotation[size_t(3*d+e)] = frame.pose.rotation(d,e);
                        }
                        CPU::BuildInterfaceQueries(request);
                    } });
                Geom::InterfaceSearchSettings search;
                search.maxGap = connection.settings.maxGap;
                search.coarseBoxesPerRank = connection.settings.coarseBoxesPerRank;
                auto map = Geom::DistributedInterfacePointMap::Match(connection.context, std::move(patch), points, ids,
                                                                     connection.settings.routing == "coarseBoxes" ? Geom::InterfaceRoutingMode::CoarseBoxes : Geom::InterfaceRoutingMode::AllGatherBoundary, search);
                MPI::LocalStage(data.mpi, [&]
                                {
                    R::Array<index> donors(R::RowLayout::Dense(map.Size(), 1));
                    auto write = donors.Write();
                    const auto donorRows = write.View<R::FixedRows<1>>();
                    const auto previousRead = connection.lastDonors.Read();
                    const auto previous = previousRead.View<R::FixedRows<1>>();
                    for(index iPoint=0;iPoint<map.Size();++iPoint)
                    {
                        const auto match=map.Get(iPoint).geometry;
                        donorRows(iPoint,0)=match.donorOriginalId;
                        data.maximumGap=std::max(data.maximumGap,match.distance);
                        if(previous.RowCount()&&previous(iPoint,0)!=donorRows(iPoint,0)) ++data.changedDonors;
                    }
                    connection.lastDonors=std::move(donors);
                    ++data.mapBuilds;
                    ++connection.mapGeneration; });
                connection.fields.clear();
                connection.sampling.reset();
                connection.sampling.emplace(std::move(metrics),
                                            std::move(reconstruction),
                                            std::move(map));
                MPI::LocalStage(data.mpi, [&]
                                {
                    const index fields = connection.donorDescriptor[1];
                    index offset = 2 + fields;
                    connection.fields.resize(size_t(fields));
                    for (index iField = 0; iField < fields; ++iField)
                    {
                        auto &field = connection.fields[size_t(iField)];
                        const index width =
                            connection.donorDescriptor[size_t(2 + iField)];
                        std::vector<index> logicalComponents(size_t(width), index{});
                        for (index j = 0; j < width; ++j)
                        {
                            const index donorComponent =
                                connection.donorDescriptor[size_t(offset + j)];
                            DNDS_check_throw_info(
                                donorComponent >= 0 &&
                                    donorComponent <
                                        index(connection.receiverComponents.size()),
                                "interface donor field contains an invalid logical component");
                            logicalComponents[size_t(j)] =
                                connection.receiverComponents[size_t(donorComponent)];
                        }
                        field.logicalComponents = R::MetadataBuffer<index>(logicalComponents);
                        offset += width;
                        field.workspace =
                            connection.sampling->BuildWorkspace(width);
                        field.values = R::Array<real>(R::RowLayout::Dense(
                            connection.receiverRows.RowCount(), width));
                    } });
                for (index iField = 0;
                     iField < index(connection.fields.size()); ++iField)
                {
                    auto &field = connection.fields[size_t(iField)];
                    field.result = connection.sampling->BindResult(
                        *field.workspace, field.values,
                        R::TransferMode::Persistent, int(iField));
                }
                connection.poses = std::make_pair(donorPose, receiverPose);
            }
            MPI::LocalStage(data.mpi, [&]
                            {
                for (index iField = 0;
                     iField < index(connection.fields.size()); ++iField)
                {
                    std::optional<CFV::DonorInputs> input;
                    if (publication)
                    {
                        const auto &mean = publication->_means.PhysicalField(iField);
                        switch (publication->_kind)
                        {
                        case FiniteVolume::InterfaceDonorPublication::Kind::Constant:
                            input.emplace(mean);
                            break;
                        case FiniteVolume::InterfaceDonorPublication::Kind::Gradient:
                            input.emplace(CFV::DonorInputs::Gradient(
                                mean, publication->_gradients[size_t(iField)]));
                            break;
                        case FiniteVolume::InterfaceDonorPublication::Kind::Variational:
                            input.emplace(
                                mean,
                                publication->_coefficients->PhysicalField(iField));
                            break;
                        }
                    }
                    connection.sampling->EvaluateDonors(
                        input,
                        *connection.fields[size_t(iField)].workspace);
                } });
            std::vector<R::TransferOperation> transfers;
            transfers.reserve(connection.fields.size());
            for (auto &field : connection.fields)
                transfers.push_back(field.result->Begin());
            for (auto &transfer : transfers)
                transfer.Wait();
            MPI::LocalStage(data.mpi, [&]
                            {
                if(!receiver) return;
                // Every schema-declared vector triple is absolute and expressed
                // in donor axes. Scalars (including absolute total energy) are
                // invariant under this rigid change of coordinate axes.
                const auto donorFrame=data.motion[connection.donor].At(time);
                const auto receiverFrame=data.motion[connection.receiver].At(time);
                std::vector<R::Array<real>::ReadLease> reads;
                reads.reserve(connection.fields.size());
                for (const auto &field : connection.fields)
                    reads.push_back(field.values.Read());
                auto write = data.incoming[connection.receiver]
                                 ->QuadratureValues()
                                 .WriteOwned();
                auto faceWrite = data.incoming[connection.receiver]
                                     ->FaceCenterValues()
                                     .WriteOwned();
                const auto outputLayout = write.ComponentLayout();
                const auto faceOutputLayout = faceWrite.ComponentLayout();
                std::vector<CPU::InterfacePacketFieldView> fieldViews;
                fieldViews.reserve(connection.fields.size());
                for (index iField = 0;
                     iField < index(connection.fields.size()); ++iField)
                    fieldViews.push_back({
                        reads[size_t(iField)].View<R::UniformTableRows>(),
                        connection.fields[size_t(iField)].logicalComponents.HostView()});
                const R::MetadataBuffer<CPU::InterfacePacketFieldView> descriptors(fieldViews);
                const auto rowRead = connection.receiverRows.Read();
                CPU::InterfacePacketRequest request;
                request.rows = rowRead.View<R::FixedRows<1>>();
                request.fields = descriptors.HostView();
                request.vectorStarts = connection.vectorStarts.HostView();
                request.quadrature = outputLayout.View();
                request.faceCenters = faceOutputLayout.View();
                request.rotation.donorActive = donorFrame.moving;
                request.rotation.receiverActive = receiverFrame.moving;
                for (int d = 0; d < 3; ++d)
                    for (int e = 0; e < 3; ++e)
                    {
                        request.rotation.donor[size_t(3*d+e)] = donorFrame.pose.rotation(d,e);
                        request.rotation.receiver[size_t(3*d+e)] = receiverFrame.pose.rotation(e,d);
                    }
                CPU::FillInterfacePacket(request);
                DNDS_check_throw_info(
                    connection.mapGeneration > 0 &&
                        !connection.fields.empty() &&
                        connection.fields.front().workspace &&
                        connection.fields.front().workspace->plan,
                    "interface publication lacks a retained matching route");
                sources[size_t(connection.receiver)].push_back(
                    {connection.donor, connection.receiver,
                     connection.mapGeneration, donorPose, receiverPose,
                     connection.fields.front().workspace->plan}); });
        }
        MPI::LocalStage(data.mpi, [&]
                        {
            for(index iDomain=0;iDomain<index(data.discrete.size());++iDomain)
                if(data.incoming[iDomain]) data.discrete[iDomain]->PublishBoundaryInputs(
                    *data.incoming[iDomain],time,epoch,
                    std::move(sources[size_t(iDomain)])); });
    }
}
