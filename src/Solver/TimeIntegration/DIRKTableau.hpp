#pragma once

#include "DNDS/Core/Defines.hpp"
#include <vector>
#include <string>

namespace DNDS::ODE
{
    /// Stiffly accurate DIRK coefficients, preserving Euler coefficient precision.
    /// Tableaus describe arithmetic only; the caller owns serialized stage solves.
    struct DIRKTableau
    {
        std::vector<std::vector<real>> a;
        std::vector<real> c;
        explicit DIRKTableau(const std::string &method)
        {
            if (method == "trapz")
            {
                a = {{0, 0}, {.5, .5}};
                c = {0, 1};
            }
            else if (method == "esdirk2")
            {
                const real gamma = 1 - std::sqrt(2.) * .5, b2 = (1 - 2 * gamma) / (4 * gamma);
                a = {{0, 0, 0}, {gamma, gamma, 0}, {1 - b2 - gamma, b2, gamma}};
                c = {0, 2 * gamma, 1};
            }
            else if (method == "esdirk3")
            {
                const real gamma = real(1767732205903) / real(4055673282236);
                a = {{0, 0, 0, 0}, {gamma, gamma, 0, 0}, {real(2746238789719) / real(10658868560708), real(-640167445237) / real(6845629431997), gamma, 0}, {real(1471266399579) / real(7840856788654), real(-4482444167858) / real(7529755066697), real(11266239266428) / real(11593286722821), gamma}};
                c = {0, 2 * gamma, a[2][0] + a[2][1] + gamma, 1};
            }
            else if (method == "esdirk4")
            {
                // Preserve Euler's six-stage ESDIRK4 coefficients verbatim,
                // including its published decimal precision and stage times.
                a = {{0, 0, 0, 0, 0, 0}, {.25, .25, 0, 0, 0, 0}, {.137776, -.055776, .25, 0, 0, 0}, {.1446368660269822, -.2239319076133447, .4492950415863626, .25, 0, 0}, {.09825878328356477, -.5915442428196704, .8101210205756877, .283164405707806, .25, 0}, {.1579162951616714, 0, .1867589405240008, .6805652953093346, -.2752405309950067, .25}};
                c = {0, .5, .332, .62, .849999966747388, 1};
            }
            else
                DNDS_check_throw_info(false, "unavailable native time method: " + method);
        }
    };
}
