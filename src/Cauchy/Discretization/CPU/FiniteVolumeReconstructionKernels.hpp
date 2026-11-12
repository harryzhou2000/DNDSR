#pragma once
#include "Cauchy/Discretization/FiniteVolumeStencil.hpp"
#include "Cauchy/Discretization/FiniteVolumeViews.hpp"
#include "Cauchy/Physics/EquationSystem.hpp"
#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "Geom/Access/MeshView.hpp"
namespace DNDS::Cauchy::CPU
{
    /// Flow-five characteristics only. Scalar extension components are not
    /// passed through identity matrices. Periodic equation frames are handled
    /// at the face, independently of polynomial modal transformations.
    index PrepareFaceCharacteristicMatrices(FiniteVolumeStencilView stencil, Geom::UnstructuredMeshView mesh,
                                            CFV::FiniteVolumeHostView metric, RHSFieldView u, index firstComponent,
                                            Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::TableRows> l,
                                            Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::TableRows> r,
                                            const EquationSystem &equation);
}
