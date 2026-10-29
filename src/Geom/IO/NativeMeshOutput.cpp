#include "Geom/IO/NativeMeshOutput.hpp"
#include "Geom/Transition/FieldTransfer.hpp"
#include <iomanip>
#include <sstream>

namespace DNDS::Geom
{
    NativeMeshOutput NativeMeshOutput::Build(const UnstructuredMeshState &source, const MPIInfo &mpi)
    {
        NativeMeshOutput output;
        std::optional<MeshReadAccess> retained;
        std::optional<EntitySpace> nodes, cells;
        Runtime::Array<uint8_t> ownedMask, ghostMask;
        std::string signature;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            source.Validate();
            retained.emplace(source.Snapshot(Runtime::Device::Host()).AcquireRead());
            nodes.emplace(source.Entities(EntityKind::Node));
            cells.emplace(source.Entities(EntityKind::Cell));
            std::ostringstream inventory;
            const auto view = retained->HostView();
            inventory << source.MeshNamespace() << ':' << view.Dimension() << ':' << view.IsPeriodic();
            for (const auto *space : {&*nodes, &*cells})
            {
                EntitySpace::CheckCommunicator(space->_mpi.comm, mpi.comm);
                for (auto count : space->_global->RLengths()) inventory << ':' << count;
            }
            if (view.IsPeriodic())
            {
                inventory << std::setprecision(std::numeric_limits<real>::max_digits10);
                const auto &periodic = view.Periodic();
                for (int frame = 0; frame < 4; ++frame)
                {
                    for (int j = 0; j < 9; ++j) inventory << ':' << periodic.rotation[frame].map().data()[j];
                    for (int j = 0; j < 3; ++j) inventory << ':' << periodic.translation[frame].map()[j] << ':' << periodic.rotationCenter[frame].map()[j];
                }
            }
            signature = inventory.str();
            DNDS_check_throw_info(signature.size() <= size_t(INT_MAX), "native output inventory exceeds MPI range"); });
        int length = int(signature.size());
        MPI_Bcast(&length, 1, MPI_INT, 0, mpi.comm);
        std::string expected;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { expected = mpi.rank == 0 ? signature : std::string(size_t(length), '\0'); });
        MPI_Bcast(expected.data(), length, MPI_CHAR, 0, mpi.comm);
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            DNDS_check_throw_info(signature == expected, "native output mesh identity, distribution or periodic frames differ between ranks");
            const auto view = retained->HostView();
            if (!view.IsPeriodic()) return;
            ownedMask = Runtime::Array<uint8_t>(Runtime::RowLayout::Dense(size_t(nodes->NumOwned()), 1));
            ghostMask = Runtime::Array<uint8_t>(Runtime::RowLayout::Dense(size_t(nodes->NumGhost()), 1));
            auto owned = ownedMask.Write(), ghost = ghostMask.Write();
            for (size_t row = 0; row < ownedMask.RowCount(); ++row) owned.View()(row, 0) = 1;
            for (size_t row = 0; row < ghostMask.RowCount(); ++row) ghost.View()(row, 0) = 0;
            if (view.IsPeriodic())
                for (index cell = 0; cell < cells->NumOwned(); ++cell)
                {
                    const auto cone = view.CellNodes(cell);
                    const auto bits = view.CellNodePeriodicBits(cell);
                    for (size_t slot = 0; slot < cone.size(); ++slot)
                    {
                        const auto node = cone[slot];
                        DNDS_check_throw_info(node >= 0 && node < nodes->NumLocal() && uint8_t(bits[slot]) < 8,
                                              "native output has unavailable node or invalid periodic image");
                        auto &mask = node < nodes->NumOwned() ? owned.View()(size_t(node), 0) : ghost.View()(size_t(node - nodes->NumOwned()), 0);
                        mask |= uint8_t(1u << uint8_t(bits[slot]));
                    }
                } });
        const auto view = retained->HostView();
        std::optional<Runtime::RowTransferPlan> route;
        if (view.IsPeriodic())
        {
            std::vector<Runtime::RowRequest> requests;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                for (index row = nodes->NumOwned(); row < nodes->NumLocal(); ++row)
                {
                    const auto global = nodes->LocalToGlobal(row);
                    requests.push_back({nodes->Owner(global), nodes->OwnerLocalRow(global), row - nodes->NumOwned()});
                } });
            // Output may run on an isolated congruent communicator. The
            // immutable maps describe rank/row addresses, not the MPI context.
            auto communication = std::make_shared<Runtime::CommunicationContext>(mpi.comm);
            route.emplace(communication, size_t(nodes->NumOwned()), size_t(nodes->NumGhost()), requests);
            Runtime::PayloadPlan reverse(route->Reverse(), ghostMask.Layout(), ownedMask.Layout(), Runtime::ElementSchema::Arithmetic<uint8_t>());
            reverse.Bind(ghostMask, ownedMask, Runtime::Reducer::Builtin<uint8_t>(Runtime::Reduction::BitOr),
                         Runtime::TransferTransport::Direct, {Runtime::TransferMode::Transient})
                .Begin()
                .Wait();
        }
        index localNodes = nodes->NumOwned();
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            if (!view.IsPeriodic()) return;
            auto mask = ownedMask.Read();
            for (size_t row = 0; row < ownedMask.RowCount(); ++row)
                for (uint8_t image = 1; image < 8; ++image)
                    if (mask.View()(row, 0) & (1u << image)) localNodes = CheckedSize::Add(localNodes, index(1)); });
        auto points = EntitySpace::CreateOwned(EntityKind::Node, mpi, localNodes);
        index pointOffset = 0;
        MPI_Exscan(&localNodes, &pointOffset, 1, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
        if (mpi.rank == 0)
            pointOffset = 0;
        output.nodeOffset = pointOffset;
        output.globalNodes = points.NumGlobal();
        output.globalCells = cells->NumGlobal();
        index localCells = cells->NumOwned();
        MPI_Exscan(&localCells, &output.cellOffset, 1, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
        if (mpi.rank == 0)
            output.cellOffset = 0;
        Runtime::Array<index> labels, ghostLabels;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            output.coordinates = Runtime::Array<real>(Runtime::RowLayout::Dense(size_t(localNodes), 3));
            output.pointSourceRows.resize(size_t(localNodes));
            if (!view.IsPeriodic())
            {
                auto coordinates = output.coordinates.Write();
                for (index row = 0; row < nodes->NumOwned(); ++row)
                {
                    output.pointSourceRows[size_t(row)] = row;
                    for (size_t component = 0; component < 3; ++component)
                        coordinates.View()(size_t(row), component) = view.Coord(row)[component];
                }
                return;
            }
            labels = Runtime::Array<index>(Runtime::RowLayout::Dense(size_t(nodes->NumOwned()), 8));
            ghostLabels = Runtime::Array<index>(Runtime::RowLayout::Dense(size_t(nodes->NumGhost()), 8));
            auto mask = ownedMask.Read();
            auto write = labels.Write();
            auto coordinates = output.coordinates.Write();
            index nextImage = nodes->NumOwned();
            for (index row = 0; row < nodes->NumOwned(); ++row)
                for (uint8_t image = 0; image < 8; ++image)
                {
                    auto &label = write.View()(size_t(row), image);
                    label = UnInitIndex;
                    if (!(mask.View()(size_t(row), 0) & (1u << image))) continue;
                    const auto destination = image == 0 ? row : nextImage++;
                    label = pointOffset + destination;
                    output.pointSourceRows[size_t(destination)] = row;
                    const tPoint point = image == 0 ? tPoint(view.Coord(row)) : view.Periodic().GetCoordByBits(view.Coord(row), NodePeriodicBits{image});
                    for (size_t component = 0; component < 3; ++component) coordinates.View()(size_t(destination), component) = point[component];
                }
            DNDS_check_throw_info(nextImage == localNodes, "native output periodic point count mismatch"); });
        if (view.IsPeriodic())
        {
            Runtime::PayloadPlan(*route, labels.Layout(), ghostLabels.Layout(), Runtime::ElementSchema::Arithmetic<index>())
                .Bind(labels, ghostLabels, Runtime::TransferTransport::Direct, {Runtime::TransferMode::Transient})
                .Begin()
                .Wait();
        }
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            auto owned = labels.Read(), ghost = ghostLabels.Read();
            output.offsets.reserve(size_t(localCells) + 1);
            output.offsets.push_back(0);
            for (index cell = 0; cell < localCells; ++cell)
            {
                auto cone = view.CellNodes(cell);
                std::vector<index> globals(cone.size());
                for (size_t slot = 0; slot < cone.size(); ++slot)
                {
                    const auto node = cone[slot];
                    if (!view.IsPeriodic()) globals[slot] = nodes->LocalToGlobal(node);
                    else
                    {
                        const auto image = uint8_t(view.CellNodePeriodicBits(cell)[slot]);
                        globals[slot] = node < nodes->NumOwned() ? owned.View()(size_t(node), image) : ghost.View()(size_t(node - nodes->NumOwned()), image);
                        DNDS_check_throw_info(globals[slot] >= 0 && globals[slot] < output.globalNodes, "native output image label unavailable");
                    }
                }
                const auto [type, vertices] = Elem::ToVTKVertsAndData(view.GetCellElement(cell), globals);
                output.types.push_back(type);
                output.connectivity.insert(output.connectivity.end(), vertices.begin(), vertices.end());
                output.offsets.push_back(index(output.connectivity.size()));
            } });
        index count = index(output.connectivity.size()), prefix = 0;
        MPI_Exscan(&count, &prefix, 1, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
        if (mpi.rank == 0)
            prefix = 0;
        MPI_Allreduce(&count, &output.globalConnectivity, 1, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
        for (auto &offset : output.offsets)
            offset += prefix;
        return output;
    }
}
