#include "Cauchy/State/DerivedFields.hpp"
#include "Cauchy/Physics/EquationSystem.hpp"
#include "Cauchy/Physics/EquationProperties.hpp"
#include <algorithm>
#include <mutex>

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    namespace
    {
        int Ordinal(DerivedQuantity quantity)
        {
            const int ordinal = int(quantity);
            DNDS_check_throw_info(ordinal >= 0 && ordinal < DerivedQuantityNumber, "unknown derived quantity");
            return ordinal;
        }
    }
    DerivedRequest::DerivedRequest(std::initializer_list<DerivedQuantity> quantities)
    {
        for (const auto quantity : quantities)
            enabled[Ordinal(quantity)] = true;
    }
    bool DerivedRequest::Has(DerivedQuantity quantity) const { return enabled[Ordinal(quantity)]; }
    int DerivedRequest::ComponentNumber() const
    {
        return int(std::count(enabled.begin(), enabled.end(), true));
    }

    struct DerivedFields::Data
    {
        struct SourceStamp
        {
            std::weak_ptr<const R::ArrayStorage> storage;
            uint64_t version, shape;
        };
        struct Origin
        {
            std::vector<SourceStamp> sources;
            EquationPropertyKey model;
            uint64_t outputVersion, outputShape;
        };
        Algebra::DofSpace space;
        DerivedRequest request;
        bool ghost;
        std::array<int, DerivedQuantityNumber> columns;
        R::Array<real> values;
        mutable std::mutex mutex;
        std::optional<Origin> origin;
        Data(Algebra::DofSpace rowSpace, DerivedRequest selection, bool ghostPart)
            : space(std::move(rowSpace)), request(selection), ghost(ghostPart)
        {
            columns.fill(-1);
            int column = 0;
            for (int quantity = 0; quantity < DerivedQuantityNumber; ++quantity)
                if (request.enabled[quantity])
                    columns[quantity] = column++;
            if (column)
                values = R::Array<real>(R::RowLayout::Dense(ghost ? space.GhostRows() : space.OwnedRows(), column));
        }
        void Validate(const EquationDofs &state, const EquationSystem &equation) const
        {
            equation.ValidateStateSchema(state);
            const auto profile = equation.ResolveKernelProfile(state.Device());
            DNDS_check_throw_info(profile == EquationKernelProfile::CompressiblePerfectGasCPU ||
                                      profile == EquationKernelProfile::CompressibleIdealMixtureCPU,
                                  "derived refresh has no kernel for the requested equation profile");
            DNDS_check_throw_info(space.Same(state.Space()), "derived field row space differs from state");
        }
        bool Current(const EquationDofs &state, const EquationSystem &equation) const
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!origin || origin->model != equation.PropertyKey() ||
                origin->sources.size() != size_t(state.PhysicalFieldNumber()) ||
                origin->outputVersion != values.Version() || origin->outputShape != values.ShapeVersion())
                return false;
            for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            {
                const auto &field = state.PhysicalField(iField);
                const auto &input = ghost ? field.Ghost() : field.Owned();
                const auto &stamp = origin->sources[iField];
                if (stamp.storage.lock() != input.SharedStorage() || stamp.version != input.Version() || stamp.shape != input.ShapeVersion())
                    return false;
            }
            return true;
        }
        bool CompatibleTemperature(const EquationSystem &equation) const
        {
            std::lock_guard<std::mutex> lock(mutex);
            return origin && origin->model == equation.PropertyKey() &&
                   origin->outputVersion == values.Version() && origin->outputShape == values.ShapeVersion();
        }
    };
    DerivedFields::DerivedFields(Algebra::DofSpace space, DerivedRequest request, bool ghost)
        : _data(std::make_shared<Data>(std::move(space), request, ghost)) {}
    bool DerivedFields::Enabled() const { return _data->request.ComponentNumber() != 0; }
    bool DerivedFields::GhostPart() const { return _data->ghost; }
    index DerivedFields::RowNumber() const { return index(_data->ghost ? _data->space.GhostRows() : _data->space.OwnedRows()); }
    const DerivedRequest &DerivedFields::Request() const { return _data->request; }
    const R::Array<real> &DerivedFields::Values() const { return _data->values; }
    void DerivedFields::PreflightRefresh() const
    {
        if (!Enabled())
            return;
        auto writer = _data->values.Write();
        writer.CancelUnmodified();
    }
    bool DerivedFields::Current(const EquationDofs &state, const EquationSystem &equation) const
    {
        _data->Validate(state, equation);
        return Enabled() && _data->Current(state, equation);
    }
    DerivedFields::ReadScope::ReadScope(std::shared_ptr<const Data> data, bool guessOnly)
        : _data(std::move(data)), _read(_data->values.Read()), _guessOnly(guessOnly) {}
    EquationBlockArrayView<const real> DerivedFields::ReadScope::Field(DerivedQuantity quantity) const
    {
        return {_read.View<R::UniformTableRows>(), Column(quantity), 1};
    }
    int DerivedFields::ReadScope::Column(DerivedQuantity quantity) const
    {
        const int column = _data->columns[Ordinal(quantity)];
        DNDS_check_throw_info(column >= 0 && (!_guessOnly || quantity == DerivedQuantity::Temperature), "derived field absent or unavailable as a stale property");
        return column;
    }
    R::ArrayView<const real, R::AccessBoundsChecks, R::UniformTableRows> DerivedFields::ReadScope::View() const
    {
        DNDS_check_throw_info(!_guessOnly, "stale-temperature scope does not expose other property columns");
        return _read.View<R::UniformTableRows>();
    }
    DerivedFields::ReadScope DerivedFields::ReadCurrent(const EquationDofs &state, const EquationSystem &equation) const
    {
        _data->Validate(state, equation);
        DNDS_check_throw_info(Enabled(), "derived bundle is disabled");
        ReadScope read(_data, false);
        read._source.emplace(_data->ghost ? state.ReadWithGhost() : state.ReadOwned());
        DNDS_check_throw_info(_data->Current(state, equation), "derived properties are stale or unpublished");
        return read;
    }
    DerivedFields::ReadScope DerivedFields::ReadTemperatureGuess(const EquationDofs &state, const EquationSystem &equation) const
    {
        _data->Validate(state, equation);
        DNDS_check_throw_info(Request().Has(DerivedQuantity::Temperature), "derived temperature guess is disabled");
        ReadScope read(_data, true);
        DNDS_check_throw_info(_data->CompatibleTemperature(equation),
                              "temperature guess has no compatible publication");
        return read;
    }

    DerivedPointValues EvaluateDerivedPoint(const EquationSystem &equation, const FlowState &state)
    {
        auto evaluator = equation.MakePointEvaluator();
        return EvaluateDerivedPoint(evaluator, state);
    }
    DerivedPointValues EvaluateDerivedPoint(EquationPointEvaluator &evaluator, const FlowState &state,
                                            Physics::Chemistry::ConstSpeciesBufferView rhoY, real temperatureGuess)
    {
        const auto point = evaluator.Evaluate(state, rhoY, temperatureGuess);
        const auto &thermo = point.thermo;
        return {thermo.temperature, thermo.pressure, thermo.soundSpeed, point.transport.viscosity,
                point.transport.conductivity, thermo.cp, thermo.cv, thermo.gasConstant, point.diffusionBound};
    }
    bool RefreshDerivedFields(const EquationDofs &state, const EquationSystem &equation,
                              DerivedFields &output, bool onlyIfStale)
    {
        return RefreshDerivedFields(state, equation, output, Physics::Chemistry::ChemicalWorkerPool::Lease{}, onlyIfStale);
    }
    bool RefreshDerivedFields(const EquationDofs &state, const EquationSystem &equation,
                              DerivedFields &output, Physics::Chemistry::ChemicalWorkerPool::Lease worker, bool onlyIfStale)
    {
        auto &data = *output._data;
        data.Validate(state, equation);
        if (!output.Enabled())
            return false;
        auto source = data.ghost ? state.ReadWithGhost() : state.ReadOwned();
        if (onlyIfStale && data.Current(state, equation))
            return false;
        const bool diffusion = data.request.Has(DerivedQuantity::DiffusionBound);
        const bool transport = diffusion || data.request.Has(DerivedQuantity::MolecularViscosity) ||
                               data.request.Has(DerivedQuantity::ThermalConductivity);
        auto evaluator = equation.MakePointEvaluator(std::move(worker), {transport, diffusion, false});
        auto rhoY = evaluator.SpeciesInput();
        EquationBlockArrayView<const real> species;
        if (rhoY.nSpecies)
        {
            const auto input = source.Block("species");
            species = data.ghost ? input.ghost : input.owned;
        }
        auto writer = std::make_optional(data.values.Write());
        const bool useTemperatureGuess = data.request.Has(DerivedQuantity::Temperature) && data.CompatibleTemperature(equation);
        {
            std::lock_guard<std::mutex> lock(data.mutex);
            data.origin.reset();
        }
        const auto input = source.Block("flow");
        const auto rows = data.ghost ? input.ghost : input.owned;
        const auto values = writer->View<R::UniformTableRows>();
        for (index iRow = 0; iRow < output.RowNumber(); ++iRow)
        {
            FlowState point;
            for (int component = 0; component < 5; ++component)
                point[component] = rows(iRow, component);
            for (int k = 0; k < rhoY.nSpecies; ++k)
                rhoY[k] = species(iRow, k);
            const real temperatureGuess = useTemperatureGuess ? values(iRow, data.columns[int(DerivedQuantity::Temperature)]) : 0;
            const auto properties = EvaluateDerivedPoint(evaluator, point, rhoY, temperatureGuess);
            for (int quantity = 0; quantity < DerivedQuantityNumber; ++quantity)
                if (data.columns[quantity] >= 0)
                    values(iRow, data.columns[quantity]) = properties[quantity];
        }
        writer.reset();
        DerivedFields::Data::Origin origin{{}, equation.PropertyKey(), data.values.Version(), data.values.ShapeVersion()};
        for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
        {
            const auto &field = state.PhysicalField(iField);
            const auto &input = data.ghost ? field.Ghost() : field.Owned();
            origin.sources.push_back({input.SharedStorage(), input.Version(), input.ShapeVersion()});
        }
        {
            std::lock_guard<std::mutex> lock(data.mutex);
            data.origin = std::move(origin);
        }
        return true;
    }
}
