#include "Cauchy/Core/Simulation.hpp"
#include "DNDS/MPI/Failure.hpp"

// Deliberately fatal subprocess fixtures. The CTest wrapper requires both the
// expected diagnostic and a nonzero, non-timeout exit; abort is not test success
// by itself. Peers deliberately remain in MPI when rank zero fails.
int main(int argc, char **argv)
{
    DNDS::MPI::Init_thread(&argc, &argv);
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    try
    {
        const std::string mode = argc == 2 ? argv[1] : "";
        DNDS_check_throw_info(mpi.size >= 2, "failure fixture needs at least two ranks");
        if (mode == "factory")
        {
            auto config = DNDS::Cauchy::SimulationConfig::ResolveCollective(DNDS::Cauchy::Json::object(), mpi.comm);
            DNDS::Cauchy::Simulation simulation(config, mpi.comm, [&](auto, const auto &, const auto &) -> DNDS::ssp<DNDS::Geom::UnstructuredMesh>
                                                {
                if (mpi.rank == 0) throw std::runtime_error("injected domain factory failure");
                MPI_Barrier(mpi.comm);
                throw std::runtime_error("factory peer escaped unexpectedly"); });
        }
        else if (mode == "transfer")
        {
            namespace R = DNDS::Runtime;
            R::Array<double> source(R::RowLayout::Dense(1, 1)), destination(R::RowLayout::Dense(1, 1));
            {
                auto write = source.Write();
                write.View()(0, 0) = mpi.rank;
            }
            auto context = std::make_shared<R::CommunicationContext>(mpi.comm);
            R::RowTransferPlan route(context, 1, 1, {{(mpi.rank + 1) % mpi.size, 0, 0}});
            auto binding = R::PayloadPlan(route, source.Layout(), destination.Layout(), R::ElementSchema::Arithmetic<double>())
                               .Bind(source, destination);
            auto operation = binding.Begin();
            if (mpi.rank == 0)
                throw std::runtime_error("injected error with active transfer");
            operation.Wait();
            MPI_Barrier(mpi.comm);
        }
        else
            throw std::runtime_error("unknown failure fixture mode");
    }
    catch (const std::exception &error)
    {
        DNDS::MPI::Abort(mpi.comm, error.what());
    }
    DNDS::MPI::Finalize();
    return 0;
}
