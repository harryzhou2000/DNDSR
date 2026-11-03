#pragma once

#include "DNDS/Core/Defines.hpp"

namespace DNDS::Cauchy
{
    struct ComponentSchema
    {
        std::string name, units, conservationGroup;
    };
    struct EquationBlockSchema
    {
        std::string name;
        std::vector<ComponentSchema> components;
        /// Physical polar-vector triples, independent of geometric dimension.
        std::vector<index> vectorStarts;
    };
    /// Logical component meaning and order, independent of physical arrays.
    /// Build before binding state; published bindings keep an immutable copy.
    struct EquationSchema
    {
        std::string family, energyConvention, physicalAccumulation;
        std::vector<EquationBlockSchema> blocks;
    };
}
