#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Discretization/LimitedVRInternal.hpp"
#include "Cauchy/Physics/ReconstructionPositivity.hpp"
#include "DNDS/Algebra/DofAlgebra.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    using FiniteVolumeDetail::Load;
    using FiniteVolumeDetail::RotateState;
    FiniteVolume::LimitedVRGeometry::LimitedVRGeometry(const Data &data)
        : gates(CFV::LimitedVRPlan::Prepare(data.metrics))
    {
        DNDS_check_throw_info(data.reconstruction,
                              "LVR geometry requires variational reconstruction");
        // Face IDs, parents, center metrics and periodic maps are shared with
        // the ordinary FV publication. The selected gradient operator and its
        // mutable execution workspace belong to the common FV path as well.
    }
    FiniteVolume::LimitedVRWorkspace::LimitedVRWorkspace(const Data &data, const EquationDofs &state)
        : scalars(data.metrics.BuildField(2)),
          branches(data.metrics.BuildField(data.settings.numerics.limitedVR.GateWidth())), seeds(data.metrics.BuildField(data.settings.numerics.limitedVR.GateWidth())), temporary(data.metrics.BuildField(data.settings.numerics.limitedVR.GateWidth())),
          base(R::RowLayout::Dense(data.metrics.NumCell(), 2), data.metrics.Device()),
          alpha(R::RowLayout::Dense(data.metrics.NumCell(), 1), data.metrics.Device()),
          branchHalo(data.BindHalo(branches)), seedHalo(data.BindHalo(seeds)), temporaryHalo(data.BindHalo(temporary))
    {
        const auto modes = data.reconstruction->NRecDOF();
        for (index field = 0; field < state.PhysicalFieldNumber(); ++field)
        {
            const auto &mean = state.PhysicalField(field);
            targetCoefficients.push_back(data.reconstruction->BuildCoefficients(
                mean.Width(), FiniteVolumeDetail::FieldLayout(mean.Owned().Layout(), data.metrics.NumCell(),
                                                              mean.Width() * modes)));
        }
        if (data.metrics.Device() == R::Device::Host())
        {
            auto penalty = alpha.Write();
            for (index iCell = 0; iCell < data.metrics.NumCell(); ++iCell)
                penalty.View()(iCell, 0) = 0;
        }
        // Device allocation is already zero-initialized. Private status and
        // sensor scratch is allocated lazily at the first queued Build.
    }
}
