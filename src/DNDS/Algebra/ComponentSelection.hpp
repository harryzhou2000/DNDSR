#pragma once

#include "DNDS/Core/Defines.hpp"
#include <algorithm>
#include <vector>

namespace DNDS::Algebra
{
    /// Validated component set, independent of field storage and physics.
    /// Empty input means every component. An explicit permutation of the full
    /// set is normalized to that same representation. No communication occurs;
    /// distributed callers must agree the normalized selection before use.
    class ComponentSelection
    {
        std::vector<index> _components;

    public:
        ComponentSelection(index width, std::vector<index> components = {})
            : _components(std::move(components))
        {
            DNDS_check_throw_info(width > 0, "component selection needs a positive field width");
            std::sort(_components.begin(), _components.end());
            DNDS_check_throw_info(_components.empty() ||
                                      (_components.front() >= 0 && _components.back() < width &&
                                       std::adjacent_find(_components.begin(), _components.end()) == _components.end()),
                                  "component selection contains an invalid or duplicate index");
            if (index(_components.size()) == width)
                _components.clear();
        }
        bool All() const { return _components.empty(); }
        bool Includes(index component) const { return All() || std::binary_search(_components.begin(), _components.end(), component); }
        const std::vector<index> &Indices() const { return _components; }
    };
}
