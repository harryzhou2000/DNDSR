#include "NativeMeshClosure.hpp"
#include "NativeConnectivity.hpp"
#include "FieldTransfer.hpp"
#include "DNDS/Runtime/RowBuilder.hpp"

namespace DNDS::Geom
{
    NativeCellClosure BuildNativeCellClosure(const NativeMeshState &source, const MPIInfo &mpi)
    {
        DNDS_check_throw_info(mpi.comm != MPI_COMM_NULL, "native cell closure requires a communicator");
        std::optional<MeshReadAccess> retained;
        std::optional<NativeRelation> initial;
        std::optional<Runtime::Array<ElemInfo>> metadata;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            source.Validate();
            retained.emplace(source.Snapshot(Runtime::Device::Host()).AcquireRead());
            const auto view = retained->HostView();
            const auto cells = source.Entities(EntityKind::Cell), nodes = source.Entities(EntityKind::Node);
            Runtime::RowBuilder rows(size_t(cells.NumOwned()), sizeof(index));
            for (index cell = 0; cell < cells.NumOwned(); ++cell)
            {
                const auto entries = view.CellNodes(cell);
                rows.ResizeRow(size_t(cell), entries.size());
                auto output = rows.Row<index>(size_t(cell));
                for (size_t slot = 0; slot < entries.size(); ++slot) output[slot] = nodes.LocalToGlobal(entries[slot]);
            }
            initial.emplace(Adj::Cell2Node, cells, nodes, Runtime::Array<index>::FromSharedStorage(rows.Commit()));
            metadata.emplace(Runtime::RowLayout::Dense(size_t(cells.NumOwned()), 1));
            auto write = metadata->Write();
            for (index cell = 0; cell < cells.NumOwned(); ++cell) write.View()(size_t(cell), 0) = view.GetCellInfo(cell); });
        NativeConnectivity setup(mpi);
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { setup.Register(initial->Read()); });
        auto support = NativeConnectivity::Inverse(initial->Read(), mpi);
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { setup.Register(support.Read()); });
        GhostSpec spec{{{EntityKind::Cell, {Adj::Cell2Node, support.Read().Kind()}, EntityKind::Cell},
                        {EntityKind::Cell, {Adj::Cell2Node, support.Read().Kind(), Adj::Cell2Node}, EntityKind::Node}}};
        auto publication = setup.BuildGhosted(setup.Evaluate(CompiledGhostTree::compile(spec)));
        MPI_Datatype type = MPI_DATATYPE_NULL;
        MPI_Type_contiguous(ElemInfo::CommMult(), ElemInfo::CommType(), &type);
        MPI_Type_commit(&type);
        Runtime::ElementSchema schema(type, sizeof(ElemInfo), "Geom.ElemInfo.v1", Runtime::ElementSchema::TypeIdentity<ElemInfo>());
        MPI_Type_free(&type);
        auto ghosts = publication.routes.at(EntityKind::Cell).TransferRows(publication.spaces.at(EntityKind::Cell), *metadata, schema);
        ghosts = Runtime::Array<ElemInfo>::FromStorage(Runtime::RowLayout::Dense(ghosts.Rows(), 1), ghosts.Storage());
        return {std::move(publication.relations.at(Adj::Cell2Node)), {std::move(*metadata), std::move(ghosts)}};
    }
}
