#include "Cauchy/State/EquationDofCommunication.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include <nlohmann/json.hpp>
#include <climits>

namespace DNDS::Cauchy
{
    void AgreeEquationBinding(const EquationDofs &state, MPI_Comm communicator, const std::string &operationControls)
    {
        const MPIInfo mpi(communicator);
        std::string descriptor, reference;
        MPI::LocalStage(mpi, [&]
                        {
            state.Validate();
            using Json = nlohmann::ordered_json;
            Json value{{"family", state.Schema().family}, {"energy", state.Schema().energyConvention}, {"mass", state.Schema().physicalAccumulation}};
            value["fields"] = Json::array(); value["blocks"] = Json::array();
            value["operationControls"] = operationControls;
            for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                value["fields"].push_back({state.Fields().Name(iField), state.PhysicalField(iField).Width()});
            for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
            {
                const auto location = state.LocateBlock(iBlock); const auto &block = state.Schema().blocks[iBlock];
                Json names = Json::array();
                for (const auto &component : block.components) names.push_back({component.name, component.units, component.conservationGroup});
                value["blocks"].push_back({block.name, location.physicalField, location.firstComponent, names, block.vectorStarts});
            }
            descriptor = value.dump();
            DNDS_check_throw_info(descriptor.size() <= size_t(INT_MAX), "equation binding descriptor is too large"); });
        int count = mpi.rank == 0 ? int(descriptor.size()) : 0;
        MPI_Bcast(&count, 1, MPI_INT, 0, communicator);
        MPI::LocalStage(mpi, [&]
                        { reference = mpi.rank == 0 ? descriptor : std::string(size_t(count), '\0'); });
        MPI_Bcast(reference.data(), count, MPI_CHAR, 0, communicator);
        int match = descriptor == reference;
        MPI_Allreduce(MPI_IN_PLACE, &match, 1, MPI_INT, MPI_MIN, communicator);
        DNDS_check_throw_info(match, "equation physical grouping/schema/operation controls differ across ranks");
    }
}
