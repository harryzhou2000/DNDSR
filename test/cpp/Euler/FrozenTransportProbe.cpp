// Compile unchanged against original and refactored Euler. No mesh or solver.
#include "Euler/Gas.hpp"
#if __has_include("DNDS/MPI/MPI.hpp")
#    include "DNDS/MPI/MPI.hpp"
#else
#    include "DNDS/MPI.hpp"
#endif
#include <fstream>
#include <cstring>

namespace
{
    using J = nlohmann::ordered_json;
    using DNDS::real;

    template <class Vector>
    J Bits(const Vector &values)
    {
        J result = J::array();
        for (Eigen::Index component = 0; component < values.size(); ++component)
        {
            const real value = values[component];
            DNDS_check_throw_info(std::isfinite(value), "nonfinite frozen transport output");
            uint64_t bits = 0;
            static_assert(sizeof(bits) == sizeof(value));
            std::memcpy(&bits, &value, sizeof(value));
            result.push_back(bits);
        }
        return result;
    }

    template <int dim>
    void Append(J &rows)
    {
        Eigen::Vector<real, dim + 2> state;
        Eigen::Vector<real, dim> normal;
        Eigen::Matrix<real, dim, dim + 2> gradient;
        state.setZero();
        state[0] = 1.7;
        real kinetic = 0;
        for (int direction = 0; direction < dim; ++direction)
        {
            state[direction + 1] = state[0] * (.2 - .07 * direction);
            kinetic += .5 * DNDS::sqr(state[direction + 1]) / state[0];
            normal[direction] = .4 + .3 * direction;
            for (int component = 0; component < dim + 2; ++component)
                gradient(direction, component) = .13 * (direction + 1) - .07 * (component + 1) + .03 * direction * component;
        }
        state[dim + 1] = 2.3 / .4 + kinetic;
        normal.normalize();
        for (bool adiabatic : {false, true})
            for (real ratio : {0., .6})
                for (bool qcr : {false, true})
                {
                    Eigen::Vector<real, dim + 2> flux;
                    flux.setZero();
                    DNDS::Euler::Gas::ViscousFlux_IdealGas<dim>(state, gradient, normal, adiabatic,
                                                                1.4, 1.4, .023, ratio, qcr, .017, 3.5, flux);
                    rows.push_back(J{{"dimension", dim}, {"adiabatic", adiabatic}, {"ratio", ratio}, {"qcr", qcr}, {"state_bits", Bits(state)}, {"normal_bits", Bits(normal)}, {"flux_bits", Bits(flux)}});
                }
    }
}

int main(int argc, char **argv)
{
    const int initialized = DNDS::MPI::Init_thread(&argc, &argv);
    if (initialized != MPI_SUCCESS)
        return initialized;
    try
    {
        DNDS_check_throw_info(argc == 2, "argument: new output JSON path");
        int ranks = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &ranks);
        DNDS_check_throw_info(ranks == 1, "pointwise transport probe is serial");
        J rows = J::array();
        Append<2>(rows);
        Append<3>(rows);
        std::ofstream stream(argv[1]);
        DNDS_check_throw_info(bool(stream), "cannot create transport probe record");
        stream << rows.dump() << '\n';
        DNDS_check_throw_info(bool(stream), "cannot write transport probe record");
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    DNDS::MPI::Finalize();
    return 0;
}
