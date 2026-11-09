#include "Cauchy/Interface/InterfaceCoupling.hpp"
#include "CFV/Plans/DistributedInterfaceSamplingPlan.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    struct InterfaceCoupling::Data
    {
        struct Connection
        {
            index donor, receiver;
            InterfaceSettings settings;
            std::shared_ptr<R::CommunicationContext> context;
            std::optional<Geom::BoundaryPatch> patch;
            struct ReceiverRow
            {
                index row;
                bool faceCenter;
            };
            // A single distributed match/sample batch, with explicit output
            // destinations in the quadrature and face-centered packet tables.
            std::vector<ReceiverRow> receiverRows;
            std::optional<std::pair<Geom::RigidPose, Geom::RigidPose>> poses;
            std::optional<CFV::DistributedInterfaceSamplingPlan> sampling;
            std::optional<CFV::DistributedSamplingWorkspace> workspace;
            R::Array<real> values;
            std::optional<R::TransferBinding> result;
            std::vector<index> lastDonors;
        };
        MPIInfo mpi;
        std::vector<std::optional<FiniteVolume>> discrete;
        std::vector<RigidMotion> motion;
        std::vector<std::optional<BoundaryInputs>> incoming;
        std::vector<std::optional<BoundaryInputs>> accepted;
        bool useAccepted = false;
        std::vector<Connection> connections;
        index changedDonors = 0;
        uint64_t mapBuilds = 0;
        real maximumGap = 0;
        explicit Data(MPI_Comm comm) : mpi(comm) {}
    };
    InterfaceCoupling::InterfaceCoupling(const SimulationConfig &config, MPI_Comm coupling,
                                         const std::vector<const FiniteVolume *> &localDomains)
        : _data(std::make_shared<Data>(coupling))
    {
        auto &data = *_data;
        MPI::LocalStage(data.mpi, [&]
                        {
            DNDS_check_throw_info(localDomains.size()==config.Domains().size(),"interface domain descriptor count differs");
            data.discrete.resize(localDomains.size());
            data.incoming.resize(localDomains.size());
            data.accepted.resize(localDomains.size());
            for(const auto &domain:config.Domains()) data.motion.emplace_back(domain.settings.motion,domain.settings.mesh.dimension);
            for(index iDomain=0;iDomain<index(localDomains.size());++iDomain)
                if(localDomains[iDomain])
                {
                    data.discrete[iDomain]=*localDomains[iDomain];
                    if(localDomains[iDomain]->HasInterfaces()) data.incoming[iDomain]=localDomains[iDomain]->BuildBoundaryInputs();
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
            connection.context = std::make_shared<R::CommunicationContext>(coupling);
            // Patch construction is domain-collective. A local exception must
            // propagate to the fatal boundary, not to a coupling reduction
            // while another donor rank is still constructing the patch.
            {
                if (auto &fv = data.discrete[connection.donor])
                    connection.patch = fv->BuildBoundaryPatch(settings.donorZone);
                if (auto &fv = data.discrete[connection.receiver])
                {
                    const auto &points = fv->BoundaryPoints();
                    for (index iPoint = 0; iPoint < index(points.size()); ++iPoint)
                        if (points[iPoint].interface && points[iPoint].zone == settings.receiverZone)
                            connection.receiverRows.push_back({iPoint, false});
                    const auto &faces = fv->BoundaryFaces();
                    for (index iFace = 0; iFace < index(faces.size()); ++iFace)
                        if (faces[iFace].interface && faces[iFace].zone == settings.receiverZone)
                            connection.receiverRows.push_back({iFace, true});
                }
            }
            index counts[2]{connection.patch ? connection.patch->Size() : 0, index(connection.receiverRows.size())};
            MPI_Allreduce(MPI_IN_PLACE, counts, 2, MPI_INT64_T, MPI_SUM, coupling);
            DNDS_check_throw_info(counts[0] > 0 && counts[1] > 0, "interface donor patch or receiver boundary quadrature is globally empty; check zone IDs");
            data.connections.push_back(std::move(connection));
        }
    }
    const BoundaryInputs *InterfaceCoupling::Inputs(index domain) const
    {
        DNDS_check_throw_info(domain >= 0 && domain < index(_data->incoming.size()), "invalid interface receiver domain");
        const auto &packets = _data->useAccepted ? _data->accepted : _data->incoming;
        return packets[domain] ? &*packets[domain] : nullptr;
    }
    void InterfaceCoupling::Commit() noexcept
    {
        _data->incoming.swap(_data->accepted);
        _data->useAccepted = true;
    }
    void InterfaceCoupling::Discard() noexcept { _data->useAccepted = true; }
    InterfaceDiagnostics InterfaceCoupling::Diagnostics() const
    {
        InterfaceDiagnostics result;
        for (const auto &connection : _data->connections)
            result.queryPoints += index(connection.receiverRows.size());
        result.changedDonors = _data->changedDonors;
        index counts[2]{result.queryPoints, result.changedDonors};
        MPI_Allreduce(MPI_IN_PLACE, counts, 2, MPI_INT64_T, MPI_SUM, _data->mpi.comm);
        result.queryPoints = counts[0];
        result.changedDonors = counts[1];
        result.maximumGap = _data->maximumGap;
        MPI_Allreduce(MPI_IN_PLACE, &result.maximumGap, 1, MPI_DOUBLE, MPI_MAX, _data->mpi.comm);
        result.mapBuilds = _data->mapBuilds;
        return result;
    }
    void InterfaceCoupling::Refresh(const std::vector<InterfaceDomainInput> &domains, real time, StateEpoch epoch)
    {
        auto &data = *_data;
        data.useAccepted = false;
        MPI::LocalStage(data.mpi, [&]
                        {
            DNDS_check_throw_info(domains.size()==data.discrete.size()&&std::isfinite(time),"invalid interface refresh descriptors/time");
            for(index iDomain=0;iDomain<index(domains.size());++iDomain)
            {
                const auto &input=domains[iDomain];
                DNDS_check_throw_info(bool(input.discrete)==bool(data.discrete[iDomain])&&bool(input.mean)==bool(input.discrete)&&bool(input.coefficients)==bool(input.discrete),
                                      "interface refresh has incomplete or changed domain participation");
                if(input.discrete)
                    DNDS_check_throw_info(&input.discrete->Mesh()==&data.discrete[iDomain]->Mesh(),"interface refresh replaced its prepared reference mesh");
                if(data.discrete[iDomain]&&data.discrete[iDomain]->HasInterfaces())
                {
                    if(!data.incoming[iDomain]) data.incoming[iDomain]=data.discrete[iDomain]->BuildBoundaryInputs();
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
            if (!connection.poses || !samePose(connection.poses->first, donorPose) || !samePose(connection.poses->second, receiverPose))
            {
                R::Array<real> points;
                R::Array<index> ids;
                std::optional<Geom::BoundaryPatch> patch;
                std::optional<CFV::VariationalReconstructionPlan> reconstruction;
                MPI::LocalStage(data.mpi, [&]
                                {
                    points=R::Array<real>(R::RowLayout::Dense(connection.receiverRows.size(),3));
                    ids=R::Array<index>(R::RowLayout::Dense(connection.receiverRows.size(),1));
                    if(donor)
                    {
                        patch=connection.patch->WithPose(donorPose);
                        reconstruction=donor->Reconstruction();
                    }
                    if(receiver)
                    {
                        const auto frame=receiver->Motion().At(time);
                        auto p=points.Write(); auto key=ids.Write();
                        for(index iPoint=0;iPoint<index(connection.receiverRows.size());++iPoint)
                        {
                            const auto row=connection.receiverRows[iPoint];
                            const auto &point = row.faceCenter ? receiver->BoundaryFaces()[row.row] : receiver->BoundaryPoints()[row.row];
                            const auto world=frame.WorldPoint(point.point);
                            for(int d=0;d<3;++d) p.View()(iPoint,d)=world[d];
                            key.View()(iPoint,0)=iPoint;
                        }
                    } });
                Geom::InterfaceSearchSettings search;
                search.maxGap = connection.settings.maxGap;
                search.coarseBoxesPerRank = connection.settings.coarseBoxesPerRank;
                auto map = Geom::DistributedInterfacePointMap::Match(connection.context, std::move(patch), points, ids,
                                                                     connection.settings.routing == "coarseBoxes" ? Geom::InterfaceRoutingMode::CoarseBoxes : Geom::InterfaceRoutingMode::AllGatherBoundary, search);
                MPI::LocalStage(data.mpi, [&]
                                {
                    std::vector<index> donors(map.Size());
                    for(index iPoint=0;iPoint<map.Size();++iPoint)
                    {
                        const auto match=map.Get(iPoint).geometry;
                        donors[iPoint]=match.donorOriginalId;
                        data.maximumGap=std::max(data.maximumGap,match.distance);
                        if(!connection.lastDonors.empty()&&connection.lastDonors[iPoint]!=donors[iPoint]) ++data.changedDonors;
                    }
                    connection.lastDonors=std::move(donors);
                    ++data.mapBuilds; });
                connection.result.reset();
                connection.sampling.emplace(std::move(reconstruction), std::move(map));
                MPI::LocalStage(data.mpi, [&]
                                {
                    connection.workspace=connection.sampling->BuildWorkspace(5);
                    connection.values=R::Array<real>(R::RowLayout::Dense(connection.receiverRows.size(),5)); });
                connection.result = connection.sampling->BindResult(*connection.workspace, connection.values);
                connection.poses = std::make_pair(donorPose, receiverPose);
            }
            const auto &input = domains[connection.donor];
            MPI::LocalStage(data.mpi, [&]
                            { connection.sampling->EvaluateDonors(input.mean, input.coefficients, *connection.workspace); });
            connection.result->Begin().Wait();
            MPI::LocalStage(data.mpi, [&]
                            {
                if(!receiver) return;
                // Values are absolute momentum in the donor mesh axes. Density
                // and absolute total energy need no moving-grid correction.
                const auto donorFrame=data.motion[connection.donor].At(time);
                const auto receiverFrame=data.motion[connection.receiver].At(time);
                const auto read=connection.values.Read();
                auto write=data.incoming[connection.receiver]->flow.Write();
                auto faceWrite=data.incoming[connection.receiver]->faceFlow.Write();
                const auto values=read.View<R::UniformTableRows>();
                const auto output=write.View<R::UniformTableRows>();
                const auto faceOutput=faceWrite.View<R::UniformTableRows>();
                for(index iPoint=0;iPoint<index(connection.receiverRows.size());++iPoint)
                {
                    FlowState value;
                    for(int j=0;j<5;++j) value[j]=values(iPoint,j);
                    value=receiverFrame.LocalState(donorFrame.WorldState(value));
                    const auto row = connection.receiverRows[iPoint];
                    for(int j=0;j<5;++j)
                        if (row.faceCenter) faceOutput(row.row,j)=value[j];
                        else output(row.row,j)=value[j];
                } });
        }
        MPI::LocalStage(data.mpi, [&]
                        {
            for(index iDomain=0;iDomain<index(data.discrete.size());++iDomain)
                if(data.incoming[iDomain]) data.discrete[iDomain]->PublishBoundaryInputs(*data.incoming[iDomain],time,epoch); });
    }
}
