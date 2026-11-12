#pragma once

#include "DNDS/Algebra/DofAlgebra.hpp"
#include <set>

namespace DNDS::Algebra
{
    /// Named numerical blocks on one domain's row space. Physical component
    /// meaning belongs to the equation, not this algebra container. Different
    /// blocks may use different uniform physical layouts. Ghosts never enter
    /// norms or pointwise updates. Copies of this object share field storage.
    class DofBundle
    {
        DofSpace _space;
        std::vector<std::pair<std::string, DofField<real>>> _blocks;

        std::vector<index> Selection(const std::vector<std::string> &names) const
        {
            std::vector<index> selected;
            std::set<std::string> seen;
            if (names.empty())
                for (index i = 0; i < index(_blocks.size()); ++i)
                    selected.push_back(i);
            else
                for (const auto &name : names)
                {
                    DNDS_check_throw_info(seen.insert(name).second, "duplicate DOF block selection: " + name);
                    selected.push_back(Find(name));
                }
            return selected;
        }
        index Find(const std::string &name) const
        {
            for (index i = 0; i < index(_blocks.size()); ++i)
                if (_blocks[i].first == name)
                    return i;
            DNDS_check_throw_info(false, "unknown DOF block: " + name);
            return -1;
        }
        void CheckPair(const DofBundle &other, const std::vector<index> &selected) const
        {
            Validate();
            other.Validate();
            DNDS_check_throw_info(_space.Same(other._space) && _blocks.size() == other._blocks.size(), "DOF bundle spaces/blocks differ");
            for (auto i : selected)
            {
                DNDS_check_throw_info(_blocks[i].first == other._blocks[i].first &&
                                          _blocks[i].second.Width() == other._blocks[i].second.Width() &&
                                          _blocks[i].second.Device() == other._blocks[i].second.Device(),
                                      "DOF bundle block schema differs");
                for (index j = 0; j < index(other._blocks.size()); ++j)
                    if (i != j)
                        DNDS_check_throw_info(!Detail::Overlaps(_blocks[i].second.Owned().Storage(), other._blocks[j].second.Owned().Storage()), "cross-block DOF alias is unsupported");
            }
        }

    public:
        explicit DofBundle(DofSpace space) : _space(std::move(space)) {}
        const DofSpace &Space() const { return _space; }
        index BlockNumber() const { return index(_blocks.size()); }
        /// Label physical arrays for local allocation diagnostics. Shared
        /// allocations are counted once even when several bundles name them.
        void SetName(const std::string &prefix, Runtime::AllocationSite site = {}) const
        {
            for (const auto &entry : _blocks)
            {
                entry.second.Owned().SetName(prefix + "/" + entry.first + "/owned", site);
                entry.second.Ghost().SetName(prefix + "/" + entry.first + "/ghost", site);
            }
        }
        void Validate() const
        {
            for (index i = 0; i < index(_blocks.size()); ++i)
            {
                const auto &entry = _blocks[i];
                entry.second.Validate();
                DNDS_check_throw_info(entry.second.Space().Same(_space), "DOF block row space changed");
                for (index j = 0; j < i; ++j)
                    DNDS_check_throw_info(!Detail::Overlaps(entry.second.Owned().Storage(), _blocks[j].second.Owned().Storage()) &&
                                              !Detail::Overlaps(entry.second.Owned().Storage(), _blocks[j].second.Ghost().Storage()) &&
                                              !Detail::Overlaps(entry.second.Ghost().Storage(), _blocks[j].second.Owned().Storage()) &&
                                              !Detail::Overlaps(entry.second.Ghost().Storage(), _blocks[j].second.Ghost().Storage()),
                                          "DOF block storage was rebound to an alias");
            }
        }
        void Add(std::string name, DofField<real> field)
        {
            field.Validate();
            DNDS_check_throw_info(!name.empty() && field.Space().Same(_space), "DOF block name/space is invalid");
            for (const auto &entry : _blocks)
            {
                DNDS_check_throw_info(entry.first != name, "duplicate DOF block: " + name);
                DNDS_check_throw_info(!Detail::Overlaps(entry.second.Owned().Storage(), field.Owned().Storage()) &&
                                          !Detail::Overlaps(entry.second.Ghost().Storage(), field.Owned().Storage()) &&
                                          !Detail::Overlaps(entry.second.Owned().Storage(), field.Ghost().Storage()) &&
                                          !Detail::Overlaps(entry.second.Ghost().Storage(), field.Ghost().Storage()),
                                      "DOF blocks overlap");
            }
            _blocks.emplace_back(std::move(name), std::move(field));
        }
        DofField<real> &Block(const std::string &name) { return _blocks[Find(name)].second; }
        const DofField<real> &Block(const std::string &name) const { return _blocks[Find(name)].second; }
        const std::string &Name(index iBlock) const
        {
            DNDS_check_throw_info(iBlock >= 0 && iBlock < BlockNumber(), "DOF block index out of range");
            return _blocks[iBlock].first;
        }
        /// Fresh fields sharing only row-space identity and immutable shapes.
        DofBundle Clone() const
        {
            Validate();
            DofBundle result(_space);
            for (const auto &entry : _blocks)
            {
                auto field = DofField<real>(_space, entry.second.Owned().Layout(), entry.second.Device());
                Algebra::Copy(field, entry.second);
                result.Add(entry.first, std::move(field));
            }
            return result;
        }
        void Fill(real value, const std::vector<std::string> &selection = {})
        {
            const auto selected = Selection(selection);
            Validate();
            for (auto i : selected)
                Algebra::Fill(_blocks[i].second, value);
        }
        void Copy(const DofBundle &other, const std::vector<std::string> &selection = {})
        {
            const auto selected = Selection(selection);
            CheckPair(other, selected);
            for (auto i : selected)
                Algebra::Copy(_blocks[i].second, other._blocks[i].second);
        }
        void Axpy(real alpha, const DofBundle &other, const std::vector<std::string> &selection = {})
        {
            const auto selected = Selection(selection);
            CheckPair(other, selected);
            for (auto i : selected)
                Algebra::Axpy(_blocks[i].second, alpha, other._blocks[i].second);
        }
        real DotLocal(const DofBundle &other, const std::vector<std::string> &selection = {}) const
        {
            const auto selected = Selection(selection);
            CheckPair(other, selected);
            real sum = 0;
            for (auto i : selected)
                sum += Algebra::DotLocal(_blocks[i].second, other._blocks[i].second);
            return sum;
        }
        /// Exactly one reduction on this domain communicator, not one per
        /// block. Simulation-level reductions must count a domain only once.
        real DotGlobal(const DofBundle &other, const std::vector<std::string> &selection = {}) const
        {
            const real local = DotLocal(other, selection);
            real global = 0;
            MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, _space.Communicator());
            return global;
        }
    };
}
