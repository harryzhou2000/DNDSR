#pragma once

#include "DNDS/Runtime/Execution/KernelViews.hpp"

namespace DNDS::Cauchy
{
    /// Borrow the optional owned-plus-ghost cell viscosities and DES shield.
    /// The host operator retains the whole Array for the duration of the flux.
    struct CellCenterViscosityView
    {
        using Values = Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::TableRows>;
        Values values;
        bool useViscosity = true;

        template <class View>
        static CellCenterViscosityView From(const View &view, bool useViscosity = true)
        {
            if (!view.RowCount())
                return {};
            return {Runtime::ProjectKernelView<Runtime::TableRows>(view.DynamicView()), useViscosity};
        }

        /// A negative return selects pointwise transport. Boundary samples
        /// have no adjacent cell on the exterior side.
        DNDS_DEVICE_CALLABLE real AtFace(index cell, index other) const
        {
            return !useViscosity || !values.RowCount() || other == UnInitIndex ? -1 : .5 * values(cell, 0) + .5 * values(other, 0);
        }

        DNDS_DEVICE_CALLABLE real ShieldAtFace(index cell, index other) const
        {
            const real own = values(cell, 1);
            return other == UnInitIndex ? own : .5 * own + .5 * values(other, 1);
        }
    };
}
