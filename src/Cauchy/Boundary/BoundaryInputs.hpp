#pragma once

#include "Cauchy/Physics/EquationSystem.hpp"
#include "Cauchy/State/EquationDofs.hpp"
#include "Cauchy/Core/StateEpoch.hpp"
#include "Geom/Spatial/ElementGeometry.hpp"
#include <optional>

namespace DNDS::Cauchy
{
    class FiniteVolume;
    class InterfaceDonorPublication;
    class ThermalPolicy;
    namespace Detail
    {
        struct BoundaryPointLayout;
    }
    /// Receiver reference-frame point row; quadrature=-1 denotes the maintained
    /// face center. This is prepared geometry, validated by its FiniteVolume.
    struct BoundaryPoint
    {
        index cell, face, quadrature;
        int zone;
        Vector3 point, normal;
        bool interface;
    };
    /// One completed inter-domain matching/sampling source retained by a
    /// boundary publication. Poses are exact values for this physical stage;
    /// routeGeneration changes whenever matching is rebuilt. routeOwner pins
    /// the distributed match, return route and donor geometry snapshot after
    /// the controller returns. Payload arrays remain owned by BoundaryInputs.
    struct BoundaryInterfaceSource
    {
        index donorDomain = -1, receiverDomain = -1;
        uint64_t routeGeneration = 0;
        Geom::RigidPose donorPose, receiverPose;
        std::shared_ptr<const void> routeOwner;
        /// Retains this rank's sampled donor fields and recorded generations.
        /// Empty on receiver-only ranks: remote allocations are owned on their
        /// donor ranks, never represented by a local pointer. The packet epoch
        /// and frozen route identify that remote sampling round. This retains
        /// storage, not a read lease or immutable copy of future donor values.
        std::shared_ptr<const InterfaceDonorPublication> localDonor;
    };
    /// Explicit incoming equation packet in receiver axes. QuadratureValues()
    /// holds RHS/reconstruction rows; FaceCenterValues() holds independently
    /// sampled approximate-Jacobian rows. Neither is an average of the other.
    /// Both carry the complete schema in canonical logical order and are
    /// published atomically; only interface rows are required. The coupling
    /// controller owns matching, interpolation and transfers.
    ///
    /// BeginBoundaryUpdate clears interface rows to NaN and withdraws the old
    /// publication; fill/transfer every required row, then PublishBoundaryInputs.
    /// This catches incomplete updates without a row-sized coverage bitmap.
    /// The packet adds no communication to a domain by itself. Copies share
    /// payload storage but retain their actual publication identity; mutation
    /// invalidates old copies through native allocation/shape generations.
    struct BoundaryInputs
    {
    private:
        friend class FiniteVolume;
        EquationDofs _quadrature;
        EquationDofs _faceCenters;
        EquationPropertyKey _property;
        std::shared_ptr<const ThermalPolicy> _thermal;
        std::shared_ptr<const Detail::BoundaryPointLayout> _layout;
        struct Publication
        {
            real time;
            StateEpoch epoch;
            std::vector<uint64_t> versions, shapes, faceVersions, faceShapes;
            std::vector<BoundaryInterfaceSource> sources;
        };
        std::shared_ptr<const Publication> _publication;
        bool _updating = false;
        BoundaryInputs(EquationDofs quadrature, EquationDofs faceCenters,
                       EquationPropertyKey property,
                       std::shared_ptr<const ThermalPolicy> thermal)
            : _quadrature(std::move(quadrature)),
              _faceCenters(std::move(faceCenters)),
              _property(std::move(property)), _thermal(std::move(thermal)) {}

    public:
        /// O(1) identity of a completed publication, preserved by packet alias.
        /// This does not test live payload versions: ValidateBoundaryInputs
        /// separately rejects a write or shape edit after publication.
        bool SamePublication(const BoundaryInputs &other) const
        {
            return _publication && _publication == other._publication;
        }
        /// Managed complete equation rows. Mutable access is intentional for a
        /// begun controller update; any later write invalidates publication by
        /// native array generations. No raw pointer escapes this packet API.
        EquationDofs &QuadratureValues() { return _quadrature; }
        const EquationDofs &QuadratureValues() const { return _quadrature; }
        EquationDofs &FaceCenterValues() { return _faceCenters; }
        const EquationDofs &FaceCenterValues() const { return _faceCenters; }
        /// Describes the frozen source snapshot supplied by the controller;
        /// these queries never refresh a field or assert current donor state.
        real Time() const
        {
            DNDS_check_throw_info(_publication, "boundary packet is unpublished");
            return _publication->time;
        }
        StateEpoch Epoch() const
        {
            DNDS_check_throw_info(_publication, "boundary packet is unpublished");
            return _publication->epoch;
        }
        /// Immutable source identities for an interface-controller packet.
        /// A manually prepared physical/test packet may have no sources.
        const std::vector<BoundaryInterfaceSource> &InterfaceSources() const
        {
            DNDS_check_throw_info(_publication,
                                  "boundary packet is unpublished");
            return _publication->sources;
        }
    };
}
