#pragma once

#include "DNDS/Runtime/Array.hpp"
#include "DNDS/Runtime/CommunicationContext.hpp"
#include <string>

namespace DNDS::Algebra
{
    /// One prepared local row ordering. Share this descriptor between fields;
    /// equal row counts or communicator membership do not establish identity.
    class DofSpace
    {
    public:
        struct Partition
        {
            size_t owned, ghost;
            std::string name;
        };

    private:
        std::shared_ptr<const Partition> _partition;
        std::shared_ptr<Runtime::CommunicationContext> _communication;

    public:
        explicit DofSpace(size_t owned, size_t ghost = 0, std::string name = {})
            : _partition(std::make_shared<const Partition>(Partition{owned, ghost, std::move(name)})) {}
        DofSpace(std::shared_ptr<Runtime::CommunicationContext> communication, size_t owned,
                 size_t ghost = 0, std::string name = {})
            : DofSpace(owned, ghost, std::move(name))
        {
            DNDS_check_throw_info(communication, "distributed DOF space needs a communication context");
            _communication = std::move(communication);
        }
        size_t OwnedRows() const { return _partition->owned; }
        size_t GhostRows() const { return _partition->ghost; }
        const std::string &Name() const { return _partition->name; }
        bool Same(const DofSpace &other) const { return _partition == other._partition; }
        bool Distributed() const { return bool(_communication); }
        MPI_Comm Communicator() const
        {
            DNDS_check_throw_info(_communication, "global algebra reduction needs a distributed DOF space");
            return _communication->Communicator();
        }
        /// Numeric commands retain row identity without performing MPI resource
        /// destruction on a CPU worker or CUDA retirement callback.
        const std::shared_ptr<const Partition> &Pin() const { return _partition; }
    };

    namespace Detail
    {
        inline bool Overlaps(const Runtime::Buffer &a, const Runtime::Buffer &b)
        {
            if (!a.Bytes() || !b.Bytes() || a.GetDevice() != b.GetDevice())
                return false;
            const auto x = reinterpret_cast<uintptr_t>(a.LegacyPointer());
            const auto y = reinterpret_cast<uintptr_t>(b.LegacyPointer());
            return x <= y ? y - x < a.Bytes() : x - y < b.Bytes();
        }
    }

    /// Composed shallow owner. Assignment shares Array handles; Copy() performs
    /// numerical copying. Algebra never reads ghost values or row padding.
    template <class T = real, size_t Components = Runtime::DynamicExtent>
    class DofField
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "DOF algebra supports float and double");
        static_assert(Components == Runtime::DynamicExtent || Components > 0);

    public:
        using value_type = T;
        using Policy = std::conditional_t<Components == Runtime::DynamicExtent, Runtime::TableRows, Runtime::FixedRows<Components>>;
        using Array = Runtime::Array<T, Policy>;
        static constexpr size_t static_components = Components;

    private:
        DofSpace _space;
        Array _owned, _ghost;
        static Runtime::RowLayout Layout(size_t rows, size_t width, size_t stride)
        {
            DNDS_check_throw_info(width && width != Runtime::DynamicExtent &&
                                      (Components == Runtime::DynamicExtent || width == Components),
                                  "DOF component width is missing or differs from its static width");
            return Runtime::RowLayout::Dense(rows, width, stride);
        }

    public:
        DofField(DofSpace space, size_t width = Components, size_t stride = 0,
                 Runtime::Device device = Runtime::Device::Host())
            : _space(std::move(space)), _owned(Layout(_space.OwnedRows(), width, stride), device),
              _ghost(Layout(_space.GhostRows(), width, stride), device) {}
        DofField(DofSpace space, Array owned, Array ghost)
            : _space(std::move(space)), _owned(std::move(owned)), _ghost(std::move(ghost)) { Validate(); }
        const DofSpace &Space() const { return _space; }
        size_t Width() const { return _owned.Layout().Width(); }
        Runtime::Device Device() const { return _owned.GetDevice(); }
        Array &Owned() { return _owned; }
        const Array &Owned() const { return _owned; }
        Array &Ghost() { return _ghost; }
        const Array &Ghost() const { return _ghost; }
        void Validate() const
        {
            const auto owned = _owned.Layout(), ghost = _ghost.Layout();
            DNDS_check_throw_info(!owned.IsVariable() && !ghost.IsVariable() && owned.Width() && owned.Width() == ghost.Width() &&
                                      owned.Rows() == _space.OwnedRows() && ghost.Rows() == _space.GhostRows() &&
                                      _owned.GetDevice() == _ghost.GetDevice() && _owned.IsValid(),
                                  "DOF field shape, placement or owned values are invalid");
            DNDS_check_throw_info(!Detail::Overlaps(_owned.Storage(), _ghost.Storage()), "DOF owned and ghost storage overlap");
        }
    };
}
