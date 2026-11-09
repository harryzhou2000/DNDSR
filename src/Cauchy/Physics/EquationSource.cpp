#include "DNDS/MPI/Failure.hpp"
#include "Cauchy/Physics/EquationSource.hpp"
#include "Cauchy/Physics/EquationProperties.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include <limits>

namespace DNDS::Cauchy
{
    void AdvanceChemicalSource(const EquationSystem &equation, const EquationDofs &state, EquationDofs &values,
                               Physics::Chemistry::ChemicalWorkerPool::Lease worker, real dt,
                               const Physics::Chemistry::ReactorStepSettings &controls, real sourceScale)
    {
        equation.ValidateStateSchema(state);
        state.RequireCompatible(values);
        EquationDofOperations::RequireDistinct(state, values);
        controls.Validate();
        DNDS_check_throw_info(equation.HasMixture() && state.Device() == Runtime::Device::Host() &&
                                  std::isfinite(dt) && dt >= 0 && std::isfinite(sourceScale) && sourceScale >= 0,
                              "unavailable or invalid constant-volume chemical step");
        auto evaluator = equation.Mixture().Borrow(std::move(worker));
        const auto input = state.ReadOwned();
        std::vector<EquationBlockArrayView<const real>> from;
        std::vector<EquationBlockArrayView<real>> to;
        for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
            from.push_back(input.Block(iBlock).owned);
        std::optional<EquationDofs::WriteScope> output;
        try
        {
            output.emplace(values.WriteOwned());
            for (index iBlock = 0; iBlock < values.BlockNumber(); ++iBlock)
                to.push_back(output->Block(iBlock));
        }
        catch (...)
        {
            MPI::AbortIfSolverFailure("Cauchy chemical source operation failed");
            if (output)
                output->CancelUnmodified();
            throw;
        }
        const auto flow = input.Block("flow").owned;
        const int independent = equation.Mixture().SpeciesNumber() - 1;
        const auto species = independent ? input.Block("species").owned : EquationBlockArrayView<const real>{};
        const auto target = independent ? output->Block("species") : EquationBlockArrayView<real>{};
        auto rhoY = evaluator.SpeciesInput();
        for (index iCell = 0; iCell < flow.RowCount(); ++iCell)
        {
            for (index iBlock = 0; iBlock < index(from.size()); ++iBlock)
                for (index j = 0; j < from[iBlock].ComponentNumber(); ++j)
                    to[iBlock](iCell, j) = from[iBlock](iCell, j);
            if (dt == 0 || sourceScale == 0)
                continue;
            FlowState stateFlow;
            for (int j = 0; j < 5; ++j)
                stateFlow[j] = flow(iCell, j);
            for (int k = 0; k < independent; ++k)
                rhoY[k] = species(iCell, k);
            const auto updated = evaluator.AdvanceConstVolume(stateFlow[0], Physics::Thermodynamics::ConservativeSpecificInternalEnergy<3>(stateFlow),
                                                              {rhoY.data, independent}, dt, controls, sourceScale);
            for (int k = 0; k < independent; ++k)
                target(iCell, k) = updated.rhoY[k];
        }
    }
    void EvaluateChemicalSource(const EquationSystem &equation, const EquationDofs &state,
                                EquationDofs &values, Physics::Chemistry::ChemicalWorkerPool::Lease worker,
                                Runtime::Array<real> *jacobian, real sourceScale, const DerivedFields *temperatureGuess)
    {
        equation.ValidateStateSchema(state);
        state.RequireCompatible(values);
        DNDS_check_throw_info(equation.HasMixture() && state.Device() == Runtime::Device::Host(), "chemical source has no kernel for the requested profile/backend");
        DNDS_check_throw_info(std::isfinite(sourceScale) && sourceScale >= 0, "chemical source scale must be finite and nonnegative");
        EquationDofOperations::RequireDistinct(state, values);
        const index components = state.ComponentNumber();
        // Resolve logical component locations explicitly. Physical field names,
        // grouping and segment offsets do not determine equation ordering.
        std::vector<index> columns;
        index first = 0, flowFirst = -1, speciesFirst = -1;
        for (const auto &block : state.Schema().blocks)
        {
            if (block.name == "flow")
                flowFirst = first;
            if (block.name == "species")
                speciesFirst = first;
            first += index(block.components.size());
        }
        for (index j = 0; j < 5; ++j)
            columns.push_back(flowFirst + j);
        const int species = equation.Mixture().SpeciesNumber() - 1;
        for (int k = 0; k < species; ++k)
            columns.push_back(speciesFirst + k);
        if (jacobian)
        {
            DNDS_check_throw_info(components > 0 && components <= std::numeric_limits<index>::max() / components, "source Jacobian row width overflows");
            DNDS_check_throw_info(jacobian->GetDevice() == state.Device() && jacobian->RowCount() == index(state.Space().OwnedRows()) &&
                                      !jacobian->Layout().IsVariable() && jacobian->Layout().Width() == components * components,
                                  "chemical source Jacobian shape/backend differs");
            for (const auto *bound : std::array<const EquationDofs *, 2>{&state, &values})
                for (index iField = 0; iField < bound->PhysicalFieldNumber(); ++iField)
                    for (const auto *array : {&bound->PhysicalField(iField).Owned(), &bound->PhysicalField(iField).Ghost()})
                        DNDS_check_throw_info(!Algebra::Detail::Overlaps(jacobian->Storage(), array->Storage()), "source Jacobian aliases equation storage");
        }
        auto evaluator = equation.MakePointEvaluator(std::move(worker), {false, false, false, true, bool(jacobian)});
        std::optional<DerivedFields::ReadScope> guess;
        if (temperatureGuess)
        {
            DNDS_check_throw_info(!temperatureGuess->GhostPart(), "source temperature guess must be owned");
            guess.emplace(temperatureGuess->ReadTemperatureGuess(state, equation));
        }
        auto input = state.ReadOwned();
        const auto flow = input.Block("flow").owned;
        const auto rhoY = species ? input.Block("species").owned : EquationBlockArrayView<const real>{};
        std::optional<Runtime::Array<real>::WriteLease> jacobianWrite;
        std::optional<EquationDofs::WriteScope> output;
        try
        {
            if (jacobian)
                jacobianWrite.emplace(jacobian->Write());
            output.emplace(values.WriteOwned());
        }
        catch (...)
        {
            MPI::AbortIfSolverFailure("Cauchy chemical source operation failed");
            if (jacobianWrite)
                jacobianWrite->CancelUnmodified();
            throw;
        }
        const auto matrix = jacobianWrite ? jacobianWrite->View<Runtime::UniformTableRows>() : Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows>{};
        // This operator replaces the full source, not just its chemical block.
        // Resolve nonchemical outputs once; future RANS/other blocks must not
        // retain values from a previous use of this scratch allocation.
        std::vector<EquationBlockArrayView<real>> nonchemical;
        for (index iBlock = 0; iBlock < values.BlockNumber(); ++iBlock)
            if (values.Schema().blocks[iBlock].name != "species")
                nonchemical.push_back(output->Block(iBlock));
        const auto sourceSpecies = species ? output->Block("species") : EquationBlockArrayView<real>{};
        const auto oldT = guess ? guess->Field(DerivedQuantity::Temperature) : EquationBlockArrayView<const real>{};
        auto composition = evaluator.SpeciesInput();
        // Any evaluation failure can leave scratch partial. Do not cancel
        // leases or publish that failed call as a valid physical result.
        for (index iCell = 0; iCell < flow.RowCount(); ++iCell)
        {
            for (const auto &block : nonchemical)
                for (index j = 0; j < block.ComponentNumber(); ++j)
                    block(iCell, j) = 0;
            if (jacobian)
                for (index j = 0; j < components * components; ++j)
                    matrix(iCell, j) = 0;
            if (sourceScale == 0)
            {
                for (int k = 0; k < species; ++k)
                    sourceSpecies(iCell, k) = 0;
                continue;
            }
            FlowState pointFlow;
            for (int j = 0; j < 5; ++j)
                pointFlow[j] = flow(iCell, j);
            for (int k = 0; k < species; ++k)
                composition[k] = rhoY(iCell, k);
            const auto point = evaluator.Evaluate(pointFlow, {composition.data, composition.nSpecies}, guess ? oldT(iCell, 0) : 0);
            for (int k = 0; k < species; ++k)
            {
                const real rate = sourceScale * point.chemicalSource[k];
                DNDS_check_throw_info(std::isfinite(rate), "scaled chemical source is nonfinite");
                sourceSpecies(iCell, k) = rate;
                if (jacobian)
                    for (index j = 0; j < index(columns.size()); ++j)
                    {
                        const real derivative = sourceScale * point.sourceJacobian(k, int(j));
                        DNDS_check_throw_info(std::isfinite(derivative), "scaled chemical Jacobian is nonfinite");
                        matrix(iCell, speciesFirst + k + components * columns[j]) = derivative;
                    }
            }
        }
    }
}
