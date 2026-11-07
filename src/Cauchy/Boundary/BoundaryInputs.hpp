#pragma once

#include "Cauchy/Physics/FlowState.hpp"
#include "Cauchy/Core/StateEpoch.hpp"
#include "DNDS/Runtime/Array/Array.hpp"
#include <optional>

namespace DNDS::Cauchy
{
    class FiniteVolume;
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
    /// Explicit incoming boundary packet, currently flow5 in receiver axes.
    /// flow holds RHS quadrature rows, faceFlow holds independently sampled
    /// face-center rows for approximate Jacobians. Neither is an average of
    /// the other. Both are published together; only interface rows are required.
    /// More equation blocks can accompany flow without entering its allocation.
    /// The coupling controller owns matching, interpolation and transfers.
    ///
    /// BeginBoundaryUpdate clears interface rows to NaN and withdraws the old
    /// publication; fill/transfer every required row, then PublishBoundaryInputs.
    /// This catches incomplete updates without a row-sized coverage bitmap.
    /// The packet adds no communication to a domain by itself. Copies share
    /// payload storage but retain their actual publication identity; mutation
    /// invalidates old copies through native allocation/shape generations.
    struct BoundaryInputs
    {
        Runtime::Array<real> flow;
        Runtime::Array<real> faceFlow;

    private:
        friend class FiniteVolume;
        std::shared_ptr<const Detail::BoundaryPointLayout> _layout;
        struct Publication
        {
            real time;
            StateEpoch epoch;
            uint64_t version, shape, faceVersion, faceShape;
        };
        std::optional<Publication> _publication;
        bool _updating = false;

    public:
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
    };
}
