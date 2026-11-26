#include "Euler/Physics/ConstVolTrajectory.hpp"

#ifdef DNDS_USE_CANTERA
#    include "cantera/zerodim.h"
#endif

namespace DNDS::Euler::Reactive0D
{
    std::vector<StateSample> runCanteraTrajectory(const ConstVolCase &c, const std::string &mechPath,
                                                  const StateSample &initial,
                                                  const std::vector<StateSample> &times)
    {
#ifdef DNDS_USE_CANTERA
        validateCaseScales(c);
        DNDS_check_throw_info(c.U.size() > 0, "ConstVolTrajectory Cantera reference requires a nonempty state vector");
        DNDS_check_throw_info(!times.empty(), "ConstVolTrajectory Cantera reference requires nonempty sample times");
        auto sol = Cantera::newSolution(mechPath, "", "none");
        auto gas = sol->thermo();
        gas->setMassFractions_NoNorm(initial.Y.data());
        gas->setState_TD(initial.T, c.rho0 * c.U[0]);
        auto reactor = Cantera::newReactorBase("IdealGasReactor", sol);
        Cantera::ReactorNet net(reactor);

        std::vector<StateSample> out;
        out.reserve(times.size());
        double lastTime = -std::numeric_limits<double>::infinity();
        for (const auto &target : times)
        {
            DNDS_check_throw_info(std::isfinite(target.tPhys) && target.tPhys >= lastTime,
                                  "ConstVolTrajectory Cantera reference sample times must be monotone");
            if (target.tPhys > 0)
                net.advance(target.tPhys);
            auto &th = *reactor->phase()->thermo();
            std::vector<double> Y(th.nSpecies());
            th.getMassFractions(Y.data());
            out.push_back({target.tCode, target.tPhys, th.temperature(), th.pressure(), std::move(Y)});
            lastTime = target.tPhys;
        }
        return out;
#else
        (void)c;
        (void)mechPath;
        (void)initial;
        (void)times;
        DNDS_check_throw_info(false, "ConstVolTrajectory Cantera reference requires DNDS_USE_CANTERA=ON");
        return {};
#endif
    }

    void cleanupCantera()
    {
#ifdef DNDS_USE_CANTERA
        Cantera::appdelete();
#endif
    }
}
