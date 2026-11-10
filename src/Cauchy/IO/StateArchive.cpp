#include "DNDS/MPI/Failure.hpp"
#include "Cauchy/IO/StateArchive.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"

namespace DNDS::Cauchy
{
    using Json = nlohmann::ordered_json;
    Json DescribeEquationStorage(const EquationDofs &state)
    {
        state.Validate();
        Json blocks = Json::array(), widths = Json::array();
        for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            widths.push_back(state.PhysicalField(iField).Width());
        for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
        {
            const auto location = state.LocateBlock(iBlock);
            blocks.push_back({{"name", state.Schema().blocks[iBlock].name}, {"field", location.physicalField}, {"first", location.firstComponent}, {"width", location.componentNumber}});
        }
        return {{"version", 1}, {"widths", widths}, {"blocks", blocks}};
    }
    void WriteEquationArrays(const EquationDofs &state, const Serializer::SerializerBaseSSP &serializer,
                             const std::string &prefix, const MPIInfo &mpi)
    {
        for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            Runtime::WriteArray(state.PhysicalField(iField).Owned(), serializer, prefix + std::to_string(iField), mpi);
    }
    StateArchive StateArchive::Read(const Serializer::SerializerBaseSSP &serializer,
                                    const std::string &prefix, Json descriptor, const MPIInfo &mpi,
                                    Serializer::ArrayGlobalOffset shard)
    {
        StateArchive result;
        result.storage = std::move(descriptor);
        MPI::LocalStage(mpi, [&]
                        { DNDS_check_throw_info(result.storage.is_object() && result.storage.value("version", 0) == 1 &&
                                                    result.storage.at("widths").is_array() && result.storage.at("blocks").is_array() &&
                                                    !result.storage.at("widths").empty() && result.storage.at("widths").size() <= result.storage.at("blocks").size(),
                                                "invalid checkpoint equation storage descriptor"); });
        for (index iField = 0; iField < index(result.storage.at("widths").size()); ++iField)
            result.fields.push_back(Runtime::ReadArray<real>(serializer, prefix + std::to_string(iField), mpi, shard));
        return result;
    }
    bool StateArchive::Compatible(const EquationDofs &destination) const
    {
        try
        {
            const auto expected = DescribeEquationStorage(destination);
            if (storage.value("version", 0) != 1 || storage.at("blocks").size() != expected.at("blocks").size() ||
                storage.at("widths").size() != fields.size() || fields.empty())
                return false;
            for (const auto &field : fields)
                if (field.Layout().IsVariable())
                    return false;
            std::vector<std::vector<std::pair<index, index>>> ranges(fields.size());
            for (index iBlock = 0; iBlock < destination.BlockNumber(); ++iBlock)
            {
                const auto &block = storage.at("blocks").at(iBlock);
                const auto field = block.at("field").get<index>(), first = block.at("first").get<index>(), width = block.at("width").get<index>();
                if (block.at("name") != expected.at("blocks").at(iBlock).at("name") ||
                    width != destination.LocateBlock(iBlock).componentNumber || field < 0 || field >= index(fields.size()) || first < 0 || width < 0)
                    return false;
                const auto capacity = fields[field].Layout().Width();
                if (first > capacity || width > capacity - first)
                    return false;
                ranges[field].push_back({first, first + width});
            }
            for (index iField = 0; iField < index(fields.size()); ++iField)
            {
                if (fields[iField].Layout().IsVariable() || fields[iField].RowCount() != fields[0].RowCount() ||
                    fields[iField].Layout().Width() != storage.at("widths").at(iField).get<index>())
                    return false;
                auto &segments = ranges[iField];
                std::sort(segments.begin(), segments.end());
                index end = 0;
                for (const auto &segment : segments)
                {
                    if (segment.first != end)
                        return false;
                    end = segment.second;
                }
                if (end != fields[iField].Layout().Width())
                    return false;
            }
            return true;
        }
        catch (const Json::exception &)
        {
            MPI::AbortIfSolverFailure("Cauchy equation archive validation failed");
            return false;
        }
    }
    void StateArchive::Restore(EquationDofs &destination, const Runtime::RowTransferPlan &route,
                               const std::shared_ptr<Runtime::CommunicationContext> &context) const
    {
        const MPIInfo mpi(route.Communicator());
        MPI::LocalStage(mpi, [&]
                        { DNDS_check_throw_info(Compatible(destination) && fields[0].RowCount() == route.SourceRows() &&
                                                    index(destination.Space().OwnedRows()) == route.DestinationRows(),
                                                "checkpoint equation fields or row mapping differ"); });
        const bool direct = storage == DescribeEquationStorage(destination);
        std::vector<Runtime::Array<real>> mapped;
        MPI::LocalStage(mpi, [&]
                        {
            if (!direct)
                for (const auto &field : fields)
                    mapped.emplace_back(Runtime::RowLayout::Dense(route.DestinationRows(), field.Layout().Width())); });
        for (index iField = 0; iField < index(fields.size()); ++iField)
        {
            auto &target = direct ? destination.PhysicalField(iField).Owned() : mapped[iField];
            Runtime::PayloadPlan(route, fields[iField].Layout(), target.Layout(), Runtime::ElementSchema::Arithmetic<real>())
                .Bind(fields[iField], target, Runtime::TransferTransport::Direct, {Runtime::TransferMode::Transient, context, 0})
                .Begin()
                .Wait();
        }
        MPI::LocalStage(mpi, [&]
                        {
                if (direct) return;
                auto write = destination.WriteOwned();
                for (index iBlock = 0; iBlock < destination.BlockNumber(); ++iBlock)
                {
                    const auto &block = storage.at("blocks").at(iBlock);
                    const auto read = mapped[block.at("field").get<index>()].Read();
                    const auto input = read.View<Runtime::UniformTableRows>();
                    const auto output = write.Block(iBlock);
                    const index first = block.at("first").get<index>();
                    for (index iCell = 0; iCell < route.DestinationRows(); ++iCell)
                        for (index j = 0; j < output.ComponentNumber(); ++j)
                            output(iCell, j) = input(iCell, first + j);
                } });
    }
}
