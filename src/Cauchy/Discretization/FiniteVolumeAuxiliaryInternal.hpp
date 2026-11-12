#pragma once

#include "Cauchy/Discretization/FiniteVolumeInternal.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    struct MeanGradients::Data
    {
        std::weak_ptr<const void> owner;
        EquationDofOperations::Binding binding;
        std::optional<EquationDofOperations::Binding> origin;
        std::vector<CFV::FVField> fields;
        std::vector<R::Array<real>> boundary;
        std::vector<std::vector<index>> vectors;
        std::vector<R::TransferBinding> meanHalo, gradientHalo;
        std::vector<uint64_t> sourceGhost, ownedVersion, ghostVersion, ownedShape, ghostShape;
        real time = 0;
        bool ghostsReady = false;
        Data(std::weak_ptr<const void> plan, const EquationDofs &state) : owner(std::move(plan)), binding(state) {}
        void Check(const EquationDofs &state, const std::vector<CFV::FVField> &values,
                   const std::shared_ptr<const void> &plan, real atTime, bool ghosts) const
        {
            DNDS_check_throw_info(owner.lock() == plan && origin && time == atTime && values.size() == fields.size(), "mean gradient publication is missing or belongs to another plan/time");
            origin->Check(state, true);
            DNDS_check_throw_info(!ghosts || ghostsReady, "mean gradient halo is not published");
            for (index iField = 0; iField < index(values.size()); ++iField)
            {
                const auto &a = values[iField], &b = fields[iField];
                DNDS_check_throw_info(a.Space().Same(b.Space()) && a.Owned().SharedStorage() == b.Owned().SharedStorage() &&
                                          a.Ghost().SharedStorage() == b.Ghost().SharedStorage() && a.Owned().Version() == ownedVersion[iField] &&
                                          a.Owned().ShapeVersion() == ownedShape[iField] && a.Ghost().ShapeVersion() == ghostShape[iField] &&
                                          state.PhysicalField(iField).Ghost().Version() == sourceGhost[iField] &&
                                          (!ghosts || a.Ghost().Version() == ghostVersion[iField]),
                                      "mean gradient source or output changed");
            }
        }
    };
    namespace AuxiliaryDetail
    {
        template <class Values>
        auto EquationAuxiliaryArrays(Values &values)
        {
            return std::array<const R::Array<real> *, 9>{values.meanThermodynamics ? &*values.meanThermodynamics : nullptr,
                                                         values.faceSpectralRadius ? &*values.faceSpectralRadius : nullptr,
                                                         values.faceWaveJump ? &*values.faceWaveJump : nullptr,
                                                         values.facePressureMinimum ? &*values.facePressureMinimum : nullptr,
                                                         values.cellSpectralSum ? &values.cellSpectralSum->Owned() : nullptr,
                                                         values.cellWaveJump ? &values.cellWaveJump->Owned() : nullptr,
                                                         values.cellPressureMinimum ? &values.cellPressureMinimum->Owned() : nullptr,
                                                         values.derivedOwned ? &values.derivedOwned->Values() : nullptr,
                                                         values.derivedGhost ? &values.derivedGhost->Values() : nullptr};
        }
    }
    struct AuxiliaryValues::EquationOrigin
    {
        struct Stamp
        {
            R::ArrayStorage::Shared storage;
            uint64_t version = 0, shape = 0;
        };
        std::weak_ptr<const void> owner;
        EquationDofOperations::Binding source;
        std::vector<uint64_t> ghostVersions;
        std::array<Stamp, 9> outputs;
        std::array<R::ArrayStorage::Shared, 3> ghosts;
        std::vector<R::Array<real>> gradientInputs;
        std::vector<std::array<uint64_t, 2>> gradientVersions;
        std::optional<BoundaryInputs> boundary;
        real time;
        EquationOrigin(std::weak_ptr<const void> plan, const EquationDofs &state, const AuxiliaryValues &values, real atTime,
                       std::optional<BoundaryInputs> trace)
            : owner(std::move(plan)), source(state.NumericAlias()), boundary(std::move(trace)), time(atTime)
        {
            for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                ghostVersions.push_back(state.PhysicalField(iField).Ghost().Version());
            const auto arrays = AuxiliaryDetail::EquationAuxiliaryArrays(values);
            for (int k = 0; k < 9; ++k)
                if (arrays[k])
                    outputs[k] = {arrays[k]->SharedStorage(), arrays[k]->Version(), arrays[k]->ShapeVersion()};
            int k = 0;
            for (const auto *field : {&values.cellSpectralSum, &values.cellWaveJump, &values.cellPressureMinimum})
                ghosts[k++] = *field ? (*field)->Ghost().SharedStorage() : nullptr;
        }
    };
}
