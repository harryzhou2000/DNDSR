#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Cauchy/State/EquationDofCommunication.hpp"
#include "Geom/Quadratures/Quadrature.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    namespace
    {
        EquationDofs BindFields(const EquationDofs &prototype, Algebra::DofBundle fields,
                                index modes)
        {
            auto schema = prototype.Schema();
            std::vector<EquationDofBinding> bindings;
            for (index iBlock = 0; iBlock < prototype.BlockNumber(); ++iBlock)
            {
                const auto location = prototype.LocateBlock(iBlock);
                auto &block = schema.blocks[iBlock];
                bindings.push_back({block.name, prototype.Fields().Name(location.physicalField),
                                    location.firstComponent * modes});
                if (modes != 1)
                {
                    auto components = std::move(block.components);
                    block.components.clear();
                    block.vectorStarts.clear();
                    for (const auto &component : components)
                        for (index mode = 0; mode < modes; ++mode)
                            block.components.push_back({component.name + ":mode" + std::to_string(mode),
                                                        component.units, "polynomial-coefficient"});
                }
            }
            if (modes != 1)
                schema.family = "polynomial:" + schema.family;
            return {std::move(schema), std::move(fields), std::move(bindings)};
        }
    }

    std::vector<FiniteVolume::Workspace::Field> FiniteVolume::Workspace::BuildFields(const Data &data, const EquationDofs &state, bool meanOnly)
    {
        std::vector<Field> result;
        result.reserve(size_t(state.PhysicalFieldNumber()));
        for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
        {
            const auto &physical = state.PhysicalField(iField);
            std::vector<index> vectors;
            for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
            {
                const auto location = state.LocateBlock(iBlock);
                if (location.physicalField == iField)
                    for (const auto first : state.Schema().blocks[iBlock].vectorStarts)
                        vectors.push_back(location.firstComponent + first);
            }
            std::sort(vectors.begin(), vectors.end());
            Field field;
            field.vectorStarts = vectors;
            field.boundaryValues = R::Array<real>(R::RowLayout::Dense(
                index(data.boundaryEntries.size()), index(physical.Width())));
            if (data.reconstruction)
            {
                field.reconstruction.emplace(data.reconstruction->BuildWorkspace(
                    index(physical.Width()),
                    FiniteVolumeDetail::FieldLayout(
                        physical.Owned().Layout(), data.metrics.NumCell(),
                        index(physical.Width()) * data.reconstruction->NRecDOF()),
                    vectors));
                field.reconstructionBoundary.emplace(
                    data.reconstruction->BuildBoundaryValues(index(physical.Width())));
                field.boundaryValues = field.reconstructionBoundary->values;
                Algebra::Fill(field.reconstruction->temporary, 0.);
            }
            if (!meanOnly && (data.settings.numerics.reconstruction == "gradient" ||
                              (data.equation.HasRANS() && data.settings.numerics.reconstruction == "constant")))
            {
                field.gradient.emplace(data.metrics.BuildField(index(physical.Width()) * data.metrics.Dimension()));
                field.gradientBoundary.emplace(data.gradient.BuildBoundaryValues(index(physical.Width())));
            }
            else if (!meanOnly && data.limitedVRGeometry)
                field.gradientBoundary.emplace(data.gradient.BuildBoundaryValues(index(physical.Width())));
            result.push_back(std::move(field));
        }
        return result;
    }
    void FiniteVolume::Workspace::BuildSourceSampling(const EquationDofs &state)
    {
        if (!owner->equation.HasRANS() && (!chemical || owner->equation.Mixture().SpeciesNumber() == 1))
            return;
        auto read = owner->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        const auto mesh = read.HostView();
        const int order = owner->settings.numerics.RHSIntegrationOrder();
        sourceOffsets.resize(size_t(owner->metrics.NumCell()) + 1, 0);
        for (index iCell = 0; iCell < owner->metrics.NumCell(); ++iCell)
        {
            const Geom::Elem::Quadrature quadrature(mesh.GetCellElement(iCell), order);
            sourceOffsets[iCell + 1] = sourceOffsets[iCell] + quadrature.GetNumPoints();
        }
        const auto count = sourceOffsets.back();
        R::Array<index> cells(R::RowLayout::Dense(count, 1));
        R::Array<real> points(R::RowLayout::Dense(count, 3));
        sourceWeights = R::Array<real>(R::RowLayout::Dense(count, 1));
        const auto geometry = owner->metrics.AcquireRead();
        const auto metric = geometry.HostView();
        {
            auto cellWrite = cells.Write();
            auto pointWrite = points.Write(), weightWrite = sourceWeights.Write();
            const auto c = cellWrite.View<R::FixedRows<1>>();
            const auto p = pointWrite.View<R::FixedRows<3>>();
            const auto w = weightWrite.View<R::FixedRows<1>>();
            for (index iCell = 0; iCell < owner->metrics.NumCell(); ++iCell)
            {
                Geom::tSmallCoords coords;
                mesh.GetCoordsOnCell(iCell, coords);
                const auto element = mesh.GetCellElement(iCell);
                const Geom::Elem::Quadrature quadrature(element, order);
                for (int iQ = 0; iQ < quadrature.GetNumPoints(); ++iQ)
                {
                    const index iPoint = sourceOffsets[iCell] + iQ;
                    const auto &shape = Geom::Elem::GetCachedShapeAtQuadrature(element.type, order, iQ);
                    const auto point = Geom::Elem::PPhysicsCoordD01Nj(coords, shape);
                    c(iPoint, 0) = iCell;
                    for (int d = 0; d < 3; ++d)
                        p(iPoint, d) = point[d];
                    w(iPoint, 0) = order == 1 ? metric.GetCellVol(iCell) : Geom::Elem::CellJacobianDet(mesh.Dimension(), coords, shape) * quadrature.GetWeight(iQ);
                }
            }
        }
        if (owner->reconstruction)
            sourceSampling.emplace(*owner->reconstruction, cells, points, owner->equation.HasRANS());
        sourceValuesAreMeans = owner->settings.numerics.reconstruction != "variational" && count == owner->metrics.NumCell();
        for (index iCell = 0; sourceValuesAreMeans && iCell < owner->metrics.NumCell(); ++iCell)
            sourceValuesAreMeans = sourceOffsets[iCell] == iCell && sourceOffsets[iCell + 1] == iCell + 1;
        for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
        {
            // Keep source sampling scratch independent of any particular
            // mean allocation so serialized stage states can reuse it.
            sourceValues.emplace_back(FiniteVolumeDetail::FieldLayout(state.PhysicalField(iField).Owned().Layout(), count,
                                                                      index(state.PhysicalField(iField).Width())));
            if (owner->equation.HasRANS())
            {
                sourceGradients.emplace_back();
                // One-point direct FV can read its ready physical gradient
                // field directly; polynomial/multi-point paths own samples.
                if (!sourceValuesAreMeans)
                    for (int d = 0; d < owner->metrics.Dimension(); ++d)
                        sourceGradients.back()[d] = R::Array<real>(sourceValues.back().Layout());
            }
        }
        if (owner->equation.HasRANS())
        {
            const index owned = owner->metrics.NumCell();
            R::Array<index> centerCells(R::RowLayout::Dense(owned, 1));
            R::Array<real> centers(R::RowLayout::Dense(owned, 3));
            {
                auto cellWrite = centerCells.Write();
                auto pointWrite = centers.Write();
                for (index iCell = 0; iCell < owned; ++iCell)
                {
                    cellWrite.View()(iCell, 0) = iCell;
                    const auto center = metric.GetCellCent(iCell);
                    for (int d = 0; d < 3; ++d)
                        pointWrite.View()(iCell, d) = center[d];
                }
            }
            if (owner->reconstruction)
                sourceJacobianSampling.emplace(*owner->reconstruction, centerCells, centers);
            for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            {
                sourceJacobianGradients.emplace_back();
                for (int d = 0; d < owner->metrics.Dimension(); ++d)
                    sourceJacobianGradients.back()[d] = R::Array<real>(FiniteVolumeDetail::FieldLayout(state.PhysicalField(iField).Owned().Layout(), owned, state.PhysicalField(iField).Width()));
            }
        }
    }
    EquationDofs FiniteVolume::Workspace::BindBoundary(const EquationDofs &state, const std::vector<Field> &fields, index rows)
    {
        const Algebra::DofSpace space(size_t(rows), 0, "equation boundary quadrature");
        Algebra::DofBundle bundle(space);
        for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            bundle.Add(state.Fields().Name(iField), CFV::FVField(space, fields[iField].boundaryValues,
                                                                 R::Array<real>(R::RowLayout::Dense(0, state.PhysicalField(iField).Width()))));
        return BindFields(state, std::move(bundle), 1);
    }
    FiniteVolume::Workspace::Workspace(std::shared_ptr<const Data> data, EquationDofs &state,
                                       std::optional<Physics::Chemistry::ChemicalWorkerPool> pool, bool meanTrial)
        : owner(std::move(data)), meanOnly(meanTrial), componentNumber(state.ComponentNumber()), physicalFieldNumber(state.PhysicalFieldNumber()),
          fields(BuildFields(*owner, state, meanOnly)),
          boundary(BindBoundary(state, fields, index(owner->boundaryEntries.size()))),
          thermal(owner->equation, owner->settings.numerics.densityFloor, owner->settings.numerics.temperatureFloor,
                  owner->settings.numerics.sensibleEnergyFloor, owner->settings.numerics.pressureFloor,
                  pool ? pool->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{}),
          diagnostics(owner->forceZones),
          workers(std::move(pool)),
          boundaryMomentumFlux(R::RowLayout::Dense(index(owner->boundaryEntries.size()), 3)),
          faceFlux(R::RowLayout::Dense(index(owner->stencilFaces.size()), state.ComponentNumber())),
          chemical(owner->settings.equation.source == "chemical")
    {
        index first = 0;
        logicalField.assign(size_t(state.ComponentNumber()), -1);
        logicalComponent.assign(size_t(state.ComponentNumber()), -1);
        for (const auto &block : state.Schema().blocks)
        {
            logicalFirst.push_back(first);
            const auto location = state.LocateBlock(index(logicalFirst.size() - 1));
            for (index j = 0; j < location.componentNumber; ++j)
            {
                logicalField[size_t(first + j)] = location.physicalField;
                logicalComponent[size_t(first + j)] = location.firstComponent + j;
            }
            if (block.name == "flow")
                flowFirst = first;
            if (block.name == "species")
                speciesFirst = first;
            if (block.name == "rans")
                ransFirst = first;
            for (const auto vector : block.vectorStarts)
                vectorStarts.push_back(first + vector);
            first += index(block.components.size());
        }
        if (!meanOnly && owner->settings.numerics.positivity)
        {
            if (owner->settings.numerics.reconstruction == "variational")
            {
                // Polynomial positivity retains only its own reference and
                // sample buffers. Direct P0/P1 policies do not manufacture
                // a modal prototype merely to reuse this implementation.
                auto coefficientPrototype = ReconstructionState::BuildCoefficients(*owner, state);
                positivity = std::make_shared<PositivityWorkspace>(owner, state, coefficientPrototype, workers);
            }
            // Conservative explicit flux blending is currently the
            // established pure-flow policy. Extended RANS/species stages
            // reject and retry their raw update instead of constructing a
            // false independently blended scalar system.
            if (!owner->equation.HasMixture() && !owner->equation.HasRANS() &&
                state.ComponentNumber() == 5 && state.PhysicalFieldNumber() == 1)
            {
                lowOrderResidual.emplace(EquationDofOperations::NewVector(state));
                lowOrderFaceFlux.emplace(R::RowLayout::Dense(index(owner->stencilFaces.size()), state.ComponentNumber()));
            }
        }
        if (!meanOnly && owner->limitedVRGeometry)
        {
            DNDS_check_throw_info(!owner->equation.HasMixture() && !owner->equation.HasRANS() &&
                                      state.ComponentNumber() == 5 && state.PhysicalFieldNumber() == 1,
                                  "LVR requires one coupled perfect-gas flow5 field");
            DNDS_check_throw_info(fields.front().reconstruction,
                                  "LVR requires a variational reconstruction workspace");
            limitedVR = std::make_unique<LimitedVRWorkspace>(*owner, *fields.front().reconstruction);
        }
        nativeDirectGradient = !meanOnly && owner->settings.numerics.reconstruction == "gradient" &&
                               (owner->settings.numerics.limiter == "none" ||
                                owner->settings.numerics.limiter == "Barth" ||
                                owner->settings.numerics.limiter == "WBAP");
        if (nativeDirectGradient)
        {
            directLimiterScratch.resize(size_t(6 * componentNumber));
            if (owner->settings.numerics.positivity)
            {
                index maximumPoints = 0;
                for (index iCell = 0; iCell < owner->metrics.NumCell(); ++iCell)
                    maximumPoints = std::max(
                        maximumPoints,
                        owner->positivityOffsets[iCell + 1] -
                            owner->positivityOffsets[iCell]);
                directPPSamples.resize(
                    size_t(maximumPoints * componentNumber));
            }
        }
        if (!meanOnly && (owner->settings.numerics.limiter == "WBAP" || limitedVR) && owner->settings.numerics.characteristicLimiter &&
            !owner->equation.HasMixture())
        {
            const auto flow = state.LocateBlock("flow");
            DNDS_check_throw_info(flow.componentNumber == 5, "characteristic limiter requires one five-component flow block");
            flowCharacteristicField = flow.physicalField;
            if (nativeDirectGradient)
                gradientFlowCharacteristic.emplace(owner->gradient.BuildFaceTransform(flow.firstComponent, 5));
            else if (owner->reconstruction)
                flowCharacteristic.emplace(owner->reconstruction->BuildFaceTransform(flow.firstComponent, 5));
        }
        if (owner->settings.numerics.reconstruction != "variational")
        {
            for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            {
                if (owner->settings.numerics.reconstruction == "constant" &&
                    !fields[iField].gradient)
                    continue;
                if (nativeDirectGradient)
                {
                    DNDS_check_throw_info(fields[iField].gradient,
                                          "explicit P1 has no prepared physical-gradient publication");
                    if (owner->settings.numerics.limiter == "none" &&
                        !owner->settings.numerics.positivity)
                        directGradients.push_back(*fields[iField].gradient);
                    else
                    {
                        directGradients.push_back(owner->metrics.BuildField(
                            FiniteVolumeDetail::FieldLayout(state.PhysicalField(iField).Owned().Layout(),
                                                            owner->metrics.NumCell(),
                                                            state.PhysicalField(iField).Width() * owner->metrics.Dimension())));
                        if (owner->settings.numerics.limiter == "WBAP")
                            directGradientInputHalo.push_back(owner->BindHalo(
                                *fields[iField].gradient, "cauchy-direct-gradient-input"));
                    }
                    directGradientHalo.push_back(owner->BindHalo(
                        directGradients.back(), "cauchy-direct-gradient"));
                    if (owner->equation.Viscous() &&
                        owner->settings.numerics.viscousGradient == "polynomial" &&
                        !owner->settings.numerics.viscousLimited &&
                        (owner->settings.numerics.limiter != "none" ||
                         owner->settings.numerics.positivity))
                    {
                        directViscousGradients.push_back(*fields[iField].gradient);
                        directViscousGradientHalo.push_back(owner->BindHalo(
                            directViscousGradients.back(),
                            "cauchy-direct-unlimited-viscous-gradient"));
                    }
                }
                else
                    directGradients.push_back(owner->metrics.BuildField(
                        FiniteVolumeDetail::FieldLayout(state.PhysicalField(iField).Owned().Layout(), owner->metrics.NumCell(),
                                                        state.PhysicalField(iField).Width() * owner->metrics.Dimension())));
                if (owner->equation.Viscous() && owner->settings.numerics.viscousGradient == "polynomial" &&
                    !owner->settings.numerics.viscousLimited && !nativeDirectGradient)
                    directViscousGradients.push_back(owner->metrics.BuildField(
                        FiniteVolumeDetail::FieldLayout(state.PhysicalField(iField).Owned().Layout(), owner->metrics.NumCell(),
                                                        state.PhysicalField(iField).Width() * owner->metrics.Dimension())));
            }
        }
        if (!meanOnly && owner->equation.Viscous() && owner->settings.numerics.viscousGradient == "primitive")
        {
            DNDS_check_throw_info(!owner->equation.HasMixture() && !owner->equation.HasRANS() &&
                                      state.ComponentNumber() == 5 && state.PhysicalFieldNumber() == 1,
                                  "primitive viscosity requires one perfect-gas flow5 field");
            primitive.emplace(owner->metrics.BuildField(5));
            primitiveGradient.emplace(owner->metrics.BuildField(5 * owner->metrics.Dimension()));
            primitiveBoundary.emplace(owner->gradient.BuildBoundaryValues(5));
            primitiveGradientHalo.emplace(owner->BindHalo(*primitiveGradient, "cauchy-primitive-gradient"));
        }
        BuildSourceSampling(state);
        if (nativeDirectGradient || ransFirst >= 0)
            directGradientPublications.resize(directGradients.size());
    }
    void FiniteVolume::Workspace::BindHalos(EquationDofs &state)
    {
        (void)state;
    }
}
