#pragma once

#include "DNDS/Core/Defines.hpp"

namespace DNDS::ODE
{
    namespace Detail
    {
        // Field backends can name time integration storage. Plain vector backends only allocate.
        template <class Operations, class State>
        auto NewTimeStorage(const State &prototype, const char *name, int) -> decltype(Operations::NewVector(prototype, name))
        {
            return Operations::NewVector(prototype, name);
        }
        template <class Operations, class State>
        State NewTimeStorage(const State &prototype, const char *, long)
        {
            return Operations::NewVector(prototype);
        }
    }

    /// Normalized equation r = resOther + rhsScale * F(U,t) - inverseDt * (M U).
    /// inverseDt is the reciprocal physical step (zero for steady or spatial-only
    /// evaluation). The equation adapter applies M to state operands only.
    /// resOther contains no contribution from this unknown or its current RHS.
    /// Other unknowns/history must remain frozen while this equation is used,
    /// including throughout a spatial FAS cycle. No MPI or physics is hidden here.
    struct TimeEquationCoefficients
    {
        real inverseDt = 0;
        real rhsScale = 1;
        real time = 0;

        void Validate() const
        {
            DNDS_check_throw_info(std::isfinite(inverseDt) && inverseDt >= 0 &&
                                      std::isfinite(rhsScale) && rhsScale > 0 && std::isfinite(time),
                                  "invalid normalized time equation coefficients");
        }

        /// Output must have distinct storage. Only owned values are produced;
        /// storage/layout and access synchronization belong to Operations.
        template <class Operations, class State>
        void AssembleResidual(State &out, const State &resOther, const State &state, const State &rate, const Operations &operations = {}) const
        {
            Validate();
            Operations::RequireDistinct(out, resOther);
            Operations::RequireDistinct(out, state);
            Operations::RequireDistinct(out, rate);
            operations.AssembleTimeResidual(out, resOther, state, rate, inverseDt, rhsScale);
        }
    };

    /// Borrowed view of one equation. Its method owner keeps resOther alive.
    /// coupledStage is an unknown within a nonlinear problem, not a serialized
    /// DIRK stage. Finish the update before asking the method for another view.
    template <class State>
    struct CoupledStageEquation
    {
        State &state;
        const State &resOther;
        TimeEquationCoefficients coefficients;
        int coupledStage = 0;
    };
}
