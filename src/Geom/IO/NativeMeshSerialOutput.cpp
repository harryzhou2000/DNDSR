#include "Geom/IO/NativeMeshSerialOutput.hpp"
#include "Geom/Transition/FieldTransfer.hpp"
#include "DNDS/Runtime/RowBuilder.hpp"
#include <iomanip>
#include <numeric>
#include <sstream>

namespace DNDS::Geom
{
    void NativeMeshSerialOutput::ValidateSource(const UnstructuredMeshState &source) const
    {
        source.Validate();
        const auto retained = _sourceSnapshot.lock();
        DNDS_check_throw_info(retained && retained == source.Snapshot(Runtime::Device::Host())._state,
                              "native serial-output rows are stale; rebuild output after changing mesh geometry or topology");
    }

    NativeMeshSerialOutput NativeMeshSerialOutput::Build(const UnstructuredMeshState &source, const MPIInfo &mpi, int reader)
    {
        NativeMeshSerialOutput result;
        std::optional<EntitySpace> cells, nodes;
        std::optional<MeshReadAccess> retained;
        Runtime::Array<index> globalCones;
        std::string signature;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            source.Validate();
            DNDS_check_throw_info(reader >= 0 && reader < mpi.size, "native serial-output reader rank is invalid");
            cells = source.Entities(EntityKind::Cell);
            nodes = source.Entities(EntityKind::Node);
            EntitySpace::CheckCommunicator(cells->_mpi.comm, mpi.comm);
            EntitySpace::CheckCommunicator(nodes->_mpi.comm, mpi.comm);
            retained.emplace(source.Snapshot(Runtime::Device::Host()).AcquireRead());
            const auto view = retained->HostView();
            result._mpi = mpi;
            result.readerRank = reader;
            result.dimension = view.Dimension();
            result.periodic = view.IsPeriodic();
            result.periodicInfo = view.Periodic();
            result._cells = cells->NumOwned();
            result._nodes = nodes->NumOwned();
            result._cellStamp = cells->Stamp();
            result._nodeStamp = nodes->Stamp();
            result._sourceSnapshot = source.Snapshot(Runtime::Device::Host())._state;
            result.cellGlobal = std::make_shared<GlobalOffsetsMapping>(*cells->_global);
            result.nodeGlobal = std::make_shared<GlobalOffsetsMapping>(*nodes->_global);
            Runtime::RowBuilder builder(size_t(cells->NumOwned()), sizeof(index));
            for (index cell = 0; cell < cells->NumOwned(); ++cell)
            {
                const auto cone = view.CellNodes(cell);
                builder.ResizeRow(size_t(cell), cone.size());
                auto row = builder.Row<index>(size_t(cell));
                for (size_t slot = 0; slot < cone.size(); ++slot)
                {
                    DNDS_check_throw_info(cone[slot] >= 0 && cone[slot] < nodes->NumLocal(), "serial output requires locally available owned-cell nodes");
                    row.data[slot] = nodes->LocalToGlobal(cone[slot]);
                }
            }
            globalCones = Runtime::Array<index>::FromSharedStorage(builder.Commit());
            std::ostringstream inventory;
            inventory << source.MeshNamespace() << ':' << reader << ':' << result.dimension << ':' << result.periodic;
            for (const auto &global : {result.cellGlobal, result.nodeGlobal})
                for (auto count : global->RLengths()) inventory << ':' << count;
            inventory << std::setprecision(std::numeric_limits<real>::max_digits10);
            if (result.periodic)
                for (int frame = 0; frame < 4; ++frame)
                {
                    for (int j = 0; j < 9; ++j) inventory << ':' << result.periodicInfo.rotation[frame].map().data()[j];
                    for (int j = 0; j < 3; ++j) inventory << ':' << result.periodicInfo.translation[frame].map()[j] << ':' << result.periodicInfo.rotationCenter[frame].map()[j];
                }
            signature = inventory.str();
            DNDS_check_throw_info(signature.size() <= size_t(INT_MAX), "native serial-output inventory exceeds MPI range"); });
        int length = int(signature.size());
        MPI_Bcast(&length, 1, MPI_INT, 0, mpi.comm);
        std::string expected;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { expected = mpi.rank == 0 ? signature : std::string(size_t(length), '\0'); });
        MPI_Bcast(expected.data(), length, MPI_CHAR, 0, mpi.comm);
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { DNDS_check_throw_info(signature == expected, "native serial-output mesh, reader rank or periodic frames differ across ranks"); });

        auto context = std::make_shared<Runtime::CommunicationContext>(mpi.comm);
        auto route = [&](const std::shared_ptr<GlobalOffsetsMapping> &global)
        {
            std::vector<std::vector<index>> send, receive;
            std::vector<index> pull;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                send.resize(size_t(mpi.size));
                receive.resize(size_t(mpi.size));
                send[size_t(reader)].resize(size_t(global->RLengths()[size_t(mpi.rank)]));
                std::iota(send[size_t(reader)].begin(), send[size_t(reader)].end(), index(0));
                if (mpi.rank == reader)
                {
                    pull.resize(size_t(global->globalSize()));
                    std::iota(pull.begin(), pull.end(), index(0));
                    for (int rank = 0; rank < mpi.size; ++rank)
                    {
                        auto &rows = receive[size_t(rank)];
                        rows.resize(size_t(global->RLengths()[size_t(rank)]));
                        std::iota(rows.begin(), rows.end(), global->ROffsets()[size_t(rank)]);
                    }
                } });
            auto transfer = Runtime::RowTransferPlan::Known(context, size_t(global->RLengths()[size_t(mpi.rank)]),
                                                            mpi.rank == reader ? size_t(global->globalSize()) : 0, std::move(send), std::move(receive));
            auto mapping = std::make_shared<OffsetAscendIndexMapping>(global->ROffsets()[size_t(mpi.rank)],
                                                                      global->RLengths()[size_t(mpi.rank)], pull, *global, mpi);
            return std::make_pair(std::move(transfer), std::move(mapping));
        };
        auto cellRoute = route(result.cellGlobal), nodeRoute = route(result.nodeGlobal);
        result.cellPull = std::move(cellRoute.second);
        result.nodePull = std::move(nodeRoute.second);
        auto gather = [&](const auto &values, const Runtime::RowTransferPlan &transfer, const Runtime::ElementSchema &schema)
        {
            using Array = std::decay_t<decltype(values)>;
            auto layout = transfer.DestinationLayout(values.Layout());
            if (!values.Layout().IsVariable())
                layout = Runtime::RowLayout::Dense(transfer.DestinationRows(), values.Layout().Width());
            std::optional<Array> output;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            { output.emplace(layout); });
            Runtime::TransferOptions options;
            options.mode = Runtime::TransferMode::Transient;
            options.context = context;
            Runtime::PayloadPlan(transfer, values.Layout(), output->Layout(), schema)
                .Bind(values, *output, Runtime::TransferTransport::Direct, options)
                .Begin()
                .Wait();
            return std::move(*output);
        };
        const auto &snapshot = source.Snapshot(Runtime::Device::Host()).Live();
        result.coordinates = gather(snapshot.coords->owned, nodeRoute.first, Runtime::ElementSchema::Arithmetic<real>());
        result.cellNodes = gather(globalCones, cellRoute.first, Runtime::ElementSchema::Arithmetic<index>());
        const auto info = Runtime::ElementSchema::For<ElemInfo>();
        result.cellInfo = gather(snapshot.cellInfo->owned, cellRoute.first, info);
        if (result.periodic)
        {
            const auto bits = Runtime::ElementSchema::For<NodePeriodicBits>();
            result.cellNodesPbi = gather(snapshot.cellNodesPbi->owned, cellRoute.first, bits);
        }
        return result;
    }
}
