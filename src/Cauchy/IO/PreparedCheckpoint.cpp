#include "Cauchy/IO/PreparedCheckpoint.hpp"
#include "Cauchy/IO/StateArchive.hpp"
#include "DNDS/Serializer/SerializerH5.hpp"

namespace DNDS::Cauchy
{
    struct PreparedCheckpoint::Data
    {
        DomainStateSnapshot snapshot;
        Json metadata;
        Runtime::Array<index> ids;
        CheckpointMetadata::MeshEnvelope mesh;
        std::shared_ptr<Runtime::CommunicationContext> communication;
        bool detached;
        Data(DomainStateSnapshot state, Json description, Runtime::Array<index> cellIds,
             CheckpointMetadata::MeshEnvelope envelope,
             std::shared_ptr<Runtime::CommunicationContext> context, bool copied)
            : snapshot(std::move(state)), metadata(std::move(description)), ids(std::move(cellIds)),
              mesh(std::move(envelope)), communication(std::move(context)), detached(copied) {}
    };
    PreparedCheckpoint::PreparedCheckpoint(DomainStateSnapshot snapshot, Json metadata, Runtime::Array<index> ids,
                                           CheckpointMetadata::MeshEnvelope mesh, std::shared_ptr<Runtime::CommunicationContext> communication, bool detached)
        : _data(std::make_shared<Data>(std::move(snapshot), std::move(metadata), std::move(ids),
                                       std::move(mesh), std::move(communication), detached))
    {
        DNDS_check_throw_info(_data->communication && _data->snapshot.Means().Device() == Runtime::Device::Host(),
                              "prepared checkpoint requires host fields and an owned communicator context");
    }
    bool PreparedCheckpoint::Detached() const { return _data->detached; }
    void PreparedCheckpoint::Write(const std::string &path, const Json &resolved, const Json &outputController) const
    {
        const auto &data = *_data;
        const auto &snapshot = data.snapshot;
        const MPIInfo mpi(data.communication->Communicator());
        auto serializer = std::make_shared<Serializer::SerializerH5>(mpi);
        serializer->OpenFile(path, false);
        serializer->WriteInt("cauchy_format", 5);
        serializer->WriteString("resolved_config", resolved.dump());
        serializer->WriteString("metadata_json", data.metadata.dump());
        if (!outputController.is_null())
            serializer->WriteString("output_controller_json", outputController.dump());
        serializer->WriteString("domain_name", snapshot.Name());
        serializer->WriteReal("accepted_time", snapshot.Time());
        serializer->WriteReal("last_accepted_dt", snapshot.LastAcceptedDt());
        serializer->WriteIndex("accepted_step", index(snapshot.Epoch().acceptedStep));
        Runtime::WriteArray(data.ids, serializer, "cell_ids", mpi);
        Runtime::WriteArray(data.mesh.topology, serializer, "mesh_topology", mpi);
        Runtime::WriteArray(data.mesh.coordinates, serializer, "mesh_coordinates", mpi);
        Runtime::WriteArray(data.mesh.periodicTransforms, serializer, "periodic_transforms", mpi);
        WriteEquationArrays(snapshot.Means(), serializer, "state_field_", mpi);
        if (snapshot.Reconstruction())
            WriteEquationArrays(*snapshot.Reconstruction(), serializer, "reconstruction_field_", mpi);
        if (snapshot.LimitedVRHistory())
            WriteStateArray(*snapshot.LimitedVRHistory(), serializer, "limited_vr_history", mpi, Runtime::ExecutionContext::Host());
        if (snapshot.PreviousState())
        {
            serializer->WriteReal("history_dt", snapshot.PreviousDt());
            WriteEquationArrays(*snapshot.PreviousState(), serializer, "history_field_", mpi);
        }
        serializer->CloseFile();
    }
}
