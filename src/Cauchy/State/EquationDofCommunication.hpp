#pragma once

#include "Cauchy/State/EquationDofs.hpp"

namespace DNDS::Cauchy
{
    /// Collective setup check before binding a sequence of physical fields.
    /// Logical meaning, physical field order/width and block offsets must agree
    /// exactly across ranks. Local row counts and physical layouts may differ.
    /// Optional opaque operation controls join the same exact agreement;
    /// workspace builders use them before binding a conditional halo sequence.
    /// Call on the controller with the same communicator on every domain rank;
    /// this is not a hot-kernel check or a payload exchange.
    void AgreeEquationBinding(const EquationDofs &state, MPI_Comm communicator, const std::string &operationControls = {});
}
