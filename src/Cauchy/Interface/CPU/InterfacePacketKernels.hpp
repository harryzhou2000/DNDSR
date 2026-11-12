#pragma once

#include "DNDS/Algebra/DofComponentView.hpp"
#include "DNDS/Core/Span.hpp"
#include <array>

namespace DNDS::Cauchy::CPU
{
    /// One receiver packet destination selected by the coupling facade.
    struct InterfacePacketRow
    {
        index row = 0;
        bool faceCenter = false;
    };

    /// Borrowed donor samples and their resolved receiver component ordinals.
    /// The enclosing access scopes and component vectors must outlive the call.
    struct InterfacePacketFieldView
    {
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks,
                           Runtime::UniformTableRows>
            values;
        SpanView<const index> receiverComponents;
    };

    /// Prepared row-major vector transform from donor axes to receiver axes.
    struct InterfacePacketRotation
    {
        std::array<real, 9> donor{1, 0, 0, 0, 1, 0, 0, 0, 1};
        std::array<real, 9> receiver{1, 0, 0, 0, 1, 0, 0, 0, 1};
        bool donorActive = false, receiverActive = false;
    };

    /// Pure receiver-side arithmetic. Ownership, access leases, schema
    /// validation, communication, and publication stay in InterfaceCoupling.
    struct InterfacePacketRequest
    {
        Runtime::ArrayView<const InterfacePacketRow, Runtime::AccessBoundsChecks,
                           Runtime::FixedRows<1>>
            rows;
        SpanView<const InterfacePacketFieldView> fields;
        SpanView<const index> vectorStarts;
        Algebra::DofComponentView<real> quadrature;
        Algebra::DofComponentView<real> faceCenters;
        InterfacePacketRotation rotation;
    };

    void FillInterfacePacket(const InterfacePacketRequest &request);
}
