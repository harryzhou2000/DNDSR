#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Interface/InterfaceCoupling.hpp"
#include "Cauchy/Discretization/FiniteVolumeReconstructionStateInternal.hpp"
#include "../CFV/ReconstructionFixture.hpp"

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
using I = DNDS::index;
static DNDS::MPIInfo mpi;

TEST_CASE("Resident interface refresh reuses routes in colocated and dislocated placement")
{
    const R::ExecutionContext placement{R::Device::CUDA(0), nullptr};
    for (const bool split : {false, true})
    {
        if (split && mpi.size < 2)
            continue;
        std::array<std::vector<int>, 2> ranks;
        for (int rank = 0; rank < mpi.size; ++rank)
            for (int domain = 0; domain < 2; ++domain)
                if (!split || rank % 2 == domain)
                    ranks[domain].push_back(rank);
        C::Json input{
            {"domains", C::Json::array({{{"name", "a"}, {"ranks", ranks[0]}}, {{"name", "b"}, {"ranks", ranks[1]}}})},
            {"defaults", {{"mesh", {{"dimension", 3}}}, {"boundaries", {{"7", {{"kind", "interface"}}}}}}},
            {"interfaces", C::Json::array({{{"donor", "a"}, {"receiver", "b"}, {"donorZone", 7}, {"receiverZone", 7}, {"maxGap", 1e-8}, {"refreshCadence", "nonlinearIteration"}},
                                           {{"donor", "b"}, {"receiver", "a"}, {"donorZone", 7}, {"receiverZone", 7}, {"maxGap", 1e-8}, {"refreshCadence", "nonlinearIteration"}}})}};
        // Explicit resident FV placement isolates production coupling from
        // time integration; the domain test covers the complete driver path.
        const auto config = C::SimulationConfig::ResolveCollective(input, mpi.comm);
        std::array<MPI_Comm, 2> communicators{MPI_COMM_NULL, MPI_COMM_NULL};
        for (int domain = 0; domain < 2; ++domain)
            MPI_Comm_split(mpi.comm, !split || mpi.rank % 2 == domain ? 0 : MPI_UNDEFINED, mpi.rank, &communicators[domain]);
        {
            std::vector<std::optional<C::FiniteVolume>> plans(2);
            std::vector<std::optional<C::EquationDofs>> values(2);
            std::array<std::shared_ptr<C::FiniteVolume::ReconstructionState>, 2> reconstruction;
            for (int domain = 0; domain < 2; ++domain)
                if (communicators[domain] != MPI_COMM_NULL)
                {
                    auto mesh = CFVTest::CartesianHexes(DNDS::MPIInfo(communicators[domain]), 2);
                    auto settings = config.Domains()[domain].settings;
                    settings.numerics.reconstruction = "variational";
                    settings.numerics.cfv.maxOrder = 1;
                    settings.numerics.cfv.SORInstead = false;
                    const auto equation = C::EquationSystem::Resolve(settings.equation);
                    const C::FiniteVolume host(mesh, settings, equation);
                    plans[domain] = host.Placed(placement.device, placement);
                    values[domain] = C::EquationDofs::Coupled(equation.Schema(), plans[domain]->Metrics().BuildField(5));
                    reconstruction[domain] = plans[domain]->BuildReconstructionState(*values[domain]);
                }
            auto communication = std::make_shared<R::CommunicationContext>(mpi.comm, false);
            C::InterfaceCoupling coupling(config, communication, plans);
            uint64_t initialMaps = 0;
            for (int round = 0; round < 2; ++round)
            {
                std::vector<C::InterfaceDomainInput> donors(2);
                for (int domain = 0; domain < 2; ++domain)
                    if (plans[domain])
                    {
                        auto &array = values[domain]->CoupledField().Owned();
                        R::Array<double> host(array.Layout());
                        {
                            auto write = host.Write();
                            for (I iCell = 0; iCell < host.RowCount(); ++iCell)
                                for (I j = 0; j < 5; ++j)
                                    write.View()(iCell, j) = j == 0 ? 2 + domain + round : j == 4 ? 3
                                                                                                  : 0;
                        }
                        array.CopyFrom(host, placement).Wait();
                        donors[domain].values = plans[domain]->PublishVariationalInterfaceDonor(*values[domain], reconstruction[domain]->WarmValues());
                    }
                C::StateEpoch epoch;
                epoch.trial = 1;
                epoch.couplingIteration = round;
                auto operation = round == 0 ? coupling.BeginRefresh(donors, 0, epoch) : coupling.BeginValueRefresh(donors, 0, epoch);
                operation.Complete();
                for (int domain = 0; domain < 2; ++domain)
                    if (plans[domain])
                    {
                        const auto packet = operation.Inputs(domain);
                        REQUIRE(packet);
                        CHECK(packet->QuadratureValues().Device() == placement.device);
                        plans[domain]->ValidateBoundaryInputs(*packet, 0);
                        for (const auto *field : {&packet->QuadratureValues(), &packet->FaceCenterValues()})
                        {
                            const auto host = field->CoupledField().Owned().To(R::Device::Host(), placement);
                            const auto read = host.Read();
                            for (I iRow = 0; iRow < host.RowCount(); ++iRow)
                            {
                                CHECK(read.View()(iRow, 0) == doctest::Approx(2 + (1 - domain) + round));
                                CHECK(read.View()(iRow, 4) == doctest::Approx(3));
                            }
                        }
                    }
                const auto diagnostic = coupling.Diagnostics();
                if (!round)
                    initialMaps = diagnostic.mapBuilds;
                else
                    CHECK(diagnostic.mapBuilds == initialMaps);
            }
        }
        for (auto &comm : communicators)
            if (comm != MPI_COMM_NULL)
                MPI_Comm_free(&comm);
    }
}

int main(int argc, char **argv)
{
    DNDS::MPI::Init_thread(&argc, &argv);
    mpi.setWorld();
    int result;
    {
        doctest::Context context(argc, argv);
        result = context.run();
    }
    DNDS::MPI::Finalize();
    return result;
}
