#pragma once

#include "DNDS/Algebra/DofComponentView.hpp"
#include "DNDS/Core/Span.hpp"
#include "Cauchy/Interface/InterfacePacketView.hpp"
#include <array>

namespace DNDS::Cauchy::CPU
{
    /// Borrowed donor samples and their resolved receiver component ordinals.
    /// The enclosing access scopes and component vectors must outlive the call.
    struct InterfacePacketFieldView
    {
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks,
                           Runtime::UniformTableRows>
            values;
        SpanView<const index> receiverComponents;
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
