#pragma once

#include "DNDS/Core/Defines.hpp"
#include <algorithm>

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

        /// Resolve setup/config names to logical indices, independently of
        /// physical field grouping/layout. Accept a unique component name or
        /// block.component; empty selects all. Unknown, ambiguous and duplicate
        /// selections fail before numerical work. No name lookup is done in a
        /// cell loop or at each residual evaluation.
        std::vector<index> ResolveComponents(const std::vector<std::string> &names) const
        {
            std::vector<index> selected;
            selected.reserve(names.size());
            for (const auto &name : names)
            {
                index found = -1, ordinal = 0;
                for (const auto &block : blocks)
                    for (const auto &component : block.components)
                    {
                        if (name == component.name || name == block.name + "." + component.name)
                        {
                            DNDS_check_throw_info(found < 0, "ambiguous equation component: " + name);
                            found = ordinal;
                        }
                        ++ordinal;
                    }
                DNDS_check_throw_info(found >= 0, "unknown equation component: " + name);
                DNDS_check_throw_info(std::find(selected.begin(), selected.end(), found) == selected.end(), "duplicate equation component: " + name);
                selected.push_back(found);
            }
            return selected;
        }
    };
}
