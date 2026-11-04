#include "Cauchy/Discretization/FiniteVolumeInternal.hpp"
#include "Cauchy/Physics/EquationProperties.hpp"
#include "Cauchy/Physics/EquationTransport.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "CFV/Math/PeriodicFrame.hpp"
#include "Geom/Quadratures/Quadrature.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include <limits>
#include <climits>

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    namespace
    {
        EquationDofs BindFields(const EquationDofs &prototype, Algebra::DofBundle fields, index modes)
        {
            auto schema = prototype.Schema();
            std::vector<EquationDofBinding> bindings;
            for (index iBlock = 0; iBlock < prototype.BlockNumber(); ++iBlock)
            {
                const auto location = prototype.LocateBlock(iBlock);
                auto &block = schema.blocks[iBlock];
                bindings.push_back({block.name, prototype.Fields().Name(location.physicalField), location.firstComponent * modes});
                if (modes != 1)
                {
                    auto components = std::move(block.components);
                    block.components.clear();
                    block.vectorStarts.clear(); // Basis coefficients are not ordinary adjacent vector triples.
                    for (const auto &component : components)
                        for (index mode = 0; mode < modes; ++mode)
                            block.components.push_back({component.name + ":mode" + std::to_string(mode), component.units, "polynomial-coefficient"});
                }
            }
            if (modes != 1)
                schema.family = "polynomial:" + schema.family;
            return {std::move(schema), std::move(fields), std::move(bindings)};
        }
        void AgreeDescriptor(const EquationDofs &state, MPI_Comm communicator)
        {
            const MPIInfo mpi(communicator);
            std::string descriptor, reference;
            MPI::LocalStage(mpi, [&]
                            {
                Json value{{"family", state.Schema().family}, {"energy", state.Schema().energyConvention}, {"mass", state.Schema().physicalAccumulation}};
                value["fields"] = Json::array(); value["blocks"] = Json::array();
                for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                    value["fields"].push_back({state.Fields().Name(iField), state.PhysicalField(iField).Width()});
                for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
                {
                    const auto location = state.LocateBlock(iBlock); const auto &block = state.Schema().blocks[iBlock];
                    Json names = Json::array();
                    for (const auto &component : block.components) names.push_back({component.name, component.units, component.conservationGroup});
                    value["blocks"].push_back({block.name, location.physicalField, location.firstComponent, names, block.vectorStarts});
                }
                descriptor = value.dump();
                DNDS_check_throw_info(descriptor.size() <= size_t(INT_MAX), "equation binding descriptor is too large"); });
            int count = mpi.rank == 0 ? int(descriptor.size()) : 0;
            MPI_Bcast(&count, 1, MPI_INT, 0, communicator);
            MPI::LocalStage(mpi, [&]
                            { reference = mpi.rank == 0 ? descriptor : std::string(size_t(count), '\0'); });
            MPI_Bcast(reference.data(), count, MPI_CHAR, 0, communicator);
            int match = descriptor == reference;
            MPI_Allreduce(MPI_IN_PLACE, &match, 1, MPI_INT, MPI_MIN, communicator);
            DNDS_check_throw_info(match, "equation physical grouping/schema differs across ranks");
        }
        void PreflightHalo(EquationDofs &fields)
        {
            auto read = fields.ReadOwned();
            std::vector<R::Array<real>::WriteLease> ghosts;
            ghosts.reserve(size_t(fields.PhysicalFieldNumber()));
            try
            {
                for (index iField = 0; iField < fields.PhysicalFieldNumber(); ++iField)
                    ghosts.emplace_back(fields.PhysicalField(iField).Ghost().Write());
            }
            catch (...)
            {
                for (auto &lease : ghosts)
                    lease.CancelUnmodified();
                throw;
            }
            for (auto &lease : ghosts)
                lease.CancelUnmodified();
        }
        /// Retains only point-local scratch. Logical block views are resolved
        /// once per operator; neither this helper nor the point loop flattens a
        /// distributed state or assumes adjacent physical flow/species storage.
        struct PointScratch
        {
            std::vector<real> values;
            Eigen::Matrix<real, Eigen::Dynamic, 3, Eigen::RowMajor> gradient;
            explicit PointScratch(index width) : values(size_t(width)), gradient(width, 3) {}
            FlowState Flow(index first) const
            {
                FlowState result{};
                for (int j = 0; j < 5; ++j)
                    result[j] = values[first + j];
                return result;
            }
            FlowGradient FlowDerivative(index first) const
            {
                FlowGradient result{};
                for (int j = 0; j < 5; ++j)
                    for (int d = 0; d < 3; ++d)
                        result[j][d] = gradient(first + j, d);
                return result;
            }
            Physics::Chemistry::ConstSpeciesBufferView Species(index first, int count) const
            {
                return {count ? values.data() + first : nullptr, count};
            }
        };
        void Gather(const std::vector<EquationBlockReadView> &blocks, index iCell, PointScratch &point)
        {
            index first = 0;
            for (const auto &block : blocks)
                for (index j = 0; j < block.ComponentNumber(); ++j)
                    point.values[first++] = block(iCell, j);
            point.gradient.setZero();
        }
        std::vector<EquationBlockReadView> BlockViews(const EquationDofs &state, const EquationDofs::ReadScope &scope)
        {
            std::vector<EquationBlockReadView> result;
            result.reserve(size_t(state.BlockNumber()));
            for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
                result.push_back(scope.Block(iBlock));
            return result;
        }
        void RotatePoint(PointScratch &point, const Geom::tGPoint &rotation, const std::vector<index> &vectorStarts)
        {
            // Scalar gradients rotate once, vector gradients on both indices.
            for (index j = 0; j < index(point.values.size()); ++j)
                point.gradient.row(j) = (point.gradient.row(j) * rotation.transpose()).eval();
            for (const auto first : vectorStarts)
            {
                const Geom::tPoint value{point.values[first], point.values[first + 1], point.values[first + 2]};
                const Geom::tPoint rotated = rotation * value;
                for (int d = 0; d < 3; ++d)
                    point.values[first + d] = rotated[d];
                point.gradient.middleRows(first, 3) = (rotation * point.gradient.middleRows(first, 3)).eval();
            }
        }
        void AdiabaticBoundary(PointScratch &point, index flowFirst, const FiniteVolumeDetail::Sample &sample,
                               const RigidFrame &frame, bool exterior)
        {
            if (sample.boundary.kind == BoundaryKind::Extrapolate)
                return;
            const auto wall = sample.boundary.WallVelocityAt({sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]}, &frame);
            const auto flow = point.Flow(flowFirst);
            const real rho = flow[0], factor = exterior ? 2 : 1;
            DNDS_check_throw_info(rho > 0, "wall producer requires positive density");
            Vector3 velocity{flow[1] / rho, flow[2] / rho, flow[3] / rho}, target = velocity;
            if (sample.boundary.kind == BoundaryKind::SlipWall)
            {
                real speed = 0;
                for (int d = 0; d < 3; ++d)
                    speed += (velocity[d] - wall[d]) * sample.normal[d];
                for (int d = 0; d < 3; ++d)
                    target[d] -= factor * speed * sample.normal[d];
            }
            else
                for (int d = 0; d < 3; ++d)
                    target[d] += factor * (wall[d] - velocity[d]);
            real energy = flow[4];
            for (int d = 0; d < 3; ++d)
            {
                point.values[flowFirst + 1 + d] = rho * target[d];
                energy += .5 * rho * (target[d] * target[d] - velocity[d] * velocity[d]);
            }
            // Same full internal energy and composition, independent of EOS.
            point.values[flowFirst + 4] = energy;
        }
        struct StreamJoin
        {
            const R::ExecutionStream &stream;
            int exceptions = std::uncaught_exceptions();
            ~StreamJoin()
            {
                if (std::uncaught_exceptions() > exceptions)
                    try
                    {
                        stream.Record().WaitHost();
                    }
                    catch (...)
                    {
                    }
            }
        };
    }

    struct FiniteVolume::EquationWorkspace
    {
        struct Field
        {
            CFV::ReconstructionWorkspace reconstruction;
            CFV::ReconstructionBoundaryValues boundary;
            std::optional<CFV::FVField> gradient;
            std::optional<R::Array<real>> gradientBoundary;
            std::vector<index> vectorStarts;
        };
        std::shared_ptr<const Data> owner;
        EquationDofs prototype;
        EquationDofOperations::Binding binding;
        std::vector<Field> fields, limited;
        EquationDofs coefficients, fluxCoefficients, boundary;
        std::vector<R::TransferBinding> meanHalo, coefficientHalo, fluxHalo;
        std::optional<Physics::Chemistry::ChemicalWorkerPool> workers;
        std::vector<index> logicalFirst, vectorStarts;
        index flowFirst = -1, speciesFirst = -1;
        std::vector<index> sourceOffsets;
        R::Array<real> sourceWeights;
        std::optional<CFV::PolynomialSamplingPlan> sourceSampling;
        std::vector<R::Array<real>> sourceValues;
        bool chemical;
        int iterations = 0;

        static std::vector<Field> BuildFields(const Data &data, const EquationDofs &state)
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
                Field field{data.reconstruction.BuildWorkspace(index(physical.Width()),
                                                               FiniteVolumeDetail::FieldLayout(physical.Owned().Layout(), data.metrics.NumCell(), index(physical.Width()) * data.reconstruction.NRecDOF()), vectors),
                            data.reconstruction.BuildBoundaryValues(index(physical.Width())),
                            {},
                            {},
                            vectors};
                Algebra::Fill(field.reconstruction.coefficients, 0.);
                Algebra::Fill(field.reconstruction.temporary, 0.);
                if (data.settings.numerics.reconstruction == "gradient")
                {
                    field.gradient.emplace(data.metrics.BuildField(index(physical.Width()) * data.metrics.Dimension()));
                    field.gradientBoundary.emplace(data.gradient.BuildBoundaryValues(index(physical.Width())));
                }
                result.push_back(std::move(field));
            }
            return result;
        }
        void BuildSourceSampling()
        {
            if (!chemical || owner->equation.Mixture().SpeciesNumber() == 1)
                return;
            auto read = owner->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
            const auto mesh = read.HostView();
            const int order = owner->settings.numerics.cfv.intOrder;
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
                        const auto &shape = Geom::Elem::detail::NBufferAtQuadrature.buf.at(element.type).at(order).at(iQ);
                        const auto point = Geom::Elem::PPhysicsCoordD01Nj(coords, shape);
                        c(iPoint, 0) = iCell;
                        for (int d = 0; d < 3; ++d)
                            p(iPoint, d) = point[d];
                        w(iPoint, 0) = Geom::Elem::CellJacobianDet(mesh.Dimension(), coords, shape) * quadrature.GetWeight(iQ);
                    }
                }
            }
            sourceSampling.emplace(owner->reconstruction, cells, points, false);
            for (index iField = 0; iField < prototype.PhysicalFieldNumber(); ++iField)
                sourceValues.emplace_back(FiniteVolumeDetail::FieldLayout(prototype.PhysicalField(iField).Owned().Layout(), count,
                                                                          index(prototype.PhysicalField(iField).Width())));
        }
        static EquationDofs BindCoefficients(const EquationDofs &state, const std::vector<Field> &fields, index modes)
        {
            Algebra::DofBundle bundle(state.Space());
            for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                bundle.Add(state.Fields().Name(iField), fields[iField].reconstruction.coefficients);
            return BindFields(state, std::move(bundle), modes);
        }
        static EquationDofs BindBoundary(const EquationDofs &state, const std::vector<Field> &fields, index rows)
        {
            const Algebra::DofSpace space(size_t(rows), 0, "equation boundary quadrature");
            Algebra::DofBundle bundle(space);
            for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                bundle.Add(state.Fields().Name(iField), CFV::FVField(space, fields[iField].boundary.values,
                                                                     R::Array<real>(R::RowLayout::Dense(0, state.PhysicalField(iField).Width()))));
            return BindFields(state, std::move(bundle), 1);
        }
        EquationWorkspace(std::shared_ptr<const Data> data, EquationDofs &state,
                          std::optional<Physics::Chemistry::ChemicalWorkerPool> pool)
            : owner(std::move(data)), prototype(state), binding(state), fields(BuildFields(*owner, state)),
              coefficients(BindCoefficients(state, fields, owner->reconstruction.NRecDOF())), fluxCoefficients(coefficients),
              boundary(BindBoundary(state, fields, owner->reconstruction.NumBoundaryQuadrature())), workers(std::move(pool)),
              chemical(owner->settings.equation.source == "chemical")
        {
            index first = 0;
            for (const auto &block : state.Schema().blocks)
            {
                logicalFirst.push_back(first);
                if (block.name == "flow")
                    flowFirst = first;
                if (block.name == "species")
                    speciesFirst = first;
                for (const auto vector : block.vectorStarts)
                    vectorStarts.push_back(first + vector);
                first += index(block.components.size());
            }
            if (owner->settings.numerics.limiter == "WBAP")
            {
                limited = fields;
                for (auto &field : limited)
                    field.reconstruction.coefficients = owner->metrics.BuildField(field.reconstruction.coefficients.Owned().Layout());
                fluxCoefficients = BindCoefficients(state, limited, owner->reconstruction.NRecDOF());
            }
            BuildSourceSampling();
        }
        void BindHalos(EquationDofs &state)
        {
            for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            {
                meanHalo.push_back(owner->metrics.BindHalo(state.PhysicalField(iField)));
                coefficientHalo.push_back(owner->metrics.BindHalo(fields[iField].reconstruction.coefficients));
                if (!limited.empty())
                    fluxHalo.push_back(owner->metrics.BindHalo(limited[iField].reconstruction.coefficients));
            }
        }

        void FillBoundary(const EquationDofs &state, bool polynomial, real time)
        {
            const auto mean = state.ReadOwned(), rec = coefficients.ReadOwned();
            const auto meanBlocks = BlockViews(state, mean), recBlocks = BlockViews(coefficients, rec);
            auto basisRead = owner->reconstruction.AcquireBasisRead();
            CFV::Math::SmallMatrix scratch;
            PointScratch point(state.ComponentNumber());
            const auto frame = owner->motion.At(time);
            const int modes = owner->reconstruction.NRecDOF();
            {
                auto output = boundary.WriteOwned();
                for (index iPoint = 0; iPoint < index(owner->boundaryEntries.size()); ++iPoint)
                {
                    const auto &sample = owner->samples[owner->boundaryEntries[iPoint]];
                    Gather(meanBlocks, sample.cell, point);
                    if (polynomial)
                    {
                        const auto basis = basisRead.FaceBasis(sample.face, sample.side, sample.point, 0, scratch);
                        for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
                            for (index j = 0; j < meanBlocks[iBlock].ComponentNumber(); ++j)
                                for (int mode = 0; mode < modes; ++mode)
                                    point.values[logicalFirst[iBlock] + j] += recBlocks[iBlock](sample.cell, j * modes + mode) * basis(0, mode);
                    }
                    AdiabaticBoundary(point, flowFirst, sample, frame, false);
                    for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
                    {
                        const auto target = output.Block(iBlock);
                        for (index j = 0; j < target.ComponentNumber(); ++j)
                            target(iPoint, j) = point.values[logicalFirst[iBlock] + j];
                    }
                }
            }
            // GradientPlan may select another face quadrature order. Its setup
            // already maps these rows to the canonical boundary quadrature.
            for (auto &field : fields)
                if (field.gradientBoundary)
                {
                    const auto source = field.boundary.values.Read();
                    auto target = field.gradientBoundary->Write();
                    const auto a = source.View<R::UniformTableRows>();
                    const auto b = target.View<R::UniformTableRows>();
                    for (index iPoint = 0; iPoint < b.RowCount(); ++iPoint)
                        for (index j = 0; j < b.Width(); ++j)
                            b(iPoint, j) = a(owner->gradientBoundaryPoints[iPoint], j);
                }
        }
        void ApplyRHS(const EquationDofs &state, EquationDofs &result, CFV::ReconstructionRegion region,
                      EquationPointEvaluator &evaluator, real time)
        {
            auto geometry = owner->metrics.AcquireRead();
            const auto metric = geometry.HostView();
            auto basisRead = owner->reconstruction.AcquireBasisRead();
            const bool ghosts = region != CFV::ReconstructionRegion::Core;
            const auto mean = ghosts ? state.ReadWithGhost() : state.ReadOwned();
            const auto rec = ghosts ? fluxCoefficients.ReadWithGhost() : fluxCoefficients.ReadOwned();
            const auto meanBlocks = BlockViews(state, mean), recBlocks = BlockViews(fluxCoefficients, rec);
            auto output = result.WriteOwned();
            std::vector<EquationBlockArrayView<real>> outputs;
            for (index iBlock = 0; iBlock < result.BlockNumber(); ++iBlock)
                outputs.push_back(output.Block(iBlock));
            const int modes = owner->reconstruction.NRecDOF(), dim = owner->metrics.Dimension();
            const bool viscous = owner->equation.Viscous();
            const int species = owner->equation.HasMixture() ? owner->equation.Mixture().SpeciesNumber() : 0;
            const int independent = std::max(0, species - 1);
            const index width = state.ComponentNumber();
            const auto frame = owner->motion.At(time);
            PointScratch left(width), right(width), face(width);
            std::vector<real> integrated(size_t(width), 0), flux(size_t(width), 0), speciesFlux(size_t(independent), 0);
            EquationTransportScratch transport(species);
            CFV::Math::SmallMatrix basisScratch;
            const auto reconstruct = [&](index iCell, const FiniteVolumeDetail::Sample &sample, int side, PointScratch &point)
            {
                Gather(meanBlocks, iCell, point);
                const auto basis = basisRead.FaceBasis(sample.face, side, sample.point, viscous ? 1 : 0, basisScratch);
                for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
                    for (index j = 0; j < meanBlocks[iBlock].ComponentNumber(); ++j)
                        for (int mode = 0; mode < modes; ++mode)
                        {
                            const real value = recBlocks[iBlock](iCell, j * modes + mode);
                            const index logical = logicalFirst[iBlock] + j;
                            point.values[logical] += value * basis(0, mode);
                            if (viscous)
                                for (int d = 0; d < dim; ++d)
                                    point.gradient(logical, d) += value * basis(1 + d, mode);
                        }
            };
            const auto compute = [&](index iCell)
            {
                std::fill(integrated.begin(), integrated.end(), 0);
                for (index entry = owner->offsets[iCell]; entry < owner->offsets[iCell + 1]; ++entry)
                {
                    const auto &sample = owner->samples[entry];
                    reconstruct(iCell, sample, sample.side, left);
                    if (sample.other != UnInitIndex)
                    {
                        reconstruct(sample.other, sample, 1 - sample.side, right);
                        RotatePoint(right, sample.rotation, vectorStarts);
                    }
                    else
                    {
                        right.values = left.values;
                        right.gradient = left.gradient;
                        AdiabaticBoundary(right, flowFirst, sample, frame, true);
                    }
                    const auto leftFlow = left.Flow(flowFirst), rightFlow = right.Flow(flowFirst);
                    const auto leftData = EquationPointEvaluator::FluxState(leftFlow, evaluator.Evaluate(leftFlow, left.Species(speciesFirst, independent), 0, {}));
                    const auto rightData = EquationPointEvaluator::FluxState(rightFlow, evaluator.Evaluate(rightFlow, right.Species(speciesFirst, independent), 0, {}));
                    const Vector3 normal{sample.normal[0], sample.normal[1], sample.normal[2]};
                    const auto grid = frame.GridVelocity({sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]});
                    const auto flowFlux = Riemann::Rusanov(leftData, rightData, normal, grid);
                    real speedLeft = 0, speedRight = 0;
                    for (int d = 0; d < 3; ++d)
                    {
                        speedLeft += (leftFlow[1 + d] / leftFlow[0] - grid[d]) * normal[d];
                        speedRight += (rightFlow[1 + d] / rightFlow[0] - grid[d]) * normal[d];
                    }
                    const real alpha = std::max(std::abs(speedLeft) + leftData.soundSpeed, std::abs(speedRight) + rightData.soundSpeed);
                    for (index j = 0; j < width; ++j)
                        flux[j] = .5 * (left.values[j] * speedLeft + right.values[j] * speedRight - alpha * (right.values[j] - left.values[j]));
                    for (int j = 0; j < 5; ++j)
                        flux[flowFirst + j] = flowFlux[j];
                    const bool wall = sample.other == UnInitIndex && sample.boundary.kind == BoundaryKind::NoSlipWall;
                    if (viscous && (sample.other != UnInitIndex || wall))
                    {
                        const real inverseDistance = wall && owner->settings.numerics.noGRPOnWall ? 0 : 1 / sample.viscousDistance;
                        for (index j = 0; j < width; ++j)
                        {
                            face.values[j] = .5 * (left.values[j] + right.values[j]);
                            for (int d = 0; d < 3; ++d)
                                face.gradient(j, d) = .5 * (left.gradient(j, d) + right.gradient(j, d)) +
                                                      normal[d] * (right.values[j] - left.values[j]) * inverseDistance;
                        }
                        const auto faceFlow = face.Flow(flowFirst);
                        const auto point = evaluator.Evaluate(faceFlow, face.Species(speciesFirst, independent), 0, {true, true, true, false, false});
                        struct SpeciesGradient
                        {
                            const PointScratch &point;
                            index first;
                            real operator()(int k, int d) const { return point.gradient(first + k, d); }
                        };
                        const auto diffusion = PrepareViscousFlux(faceFlow, face.FlowDerivative(flowFirst), point, face.Species(speciesFirst, independent),
                                                                  SpeciesGradient{face, speciesFirst}, normal, speciesFlux, transport, {wall, wall});
                        for (int j = 0; j < 5; ++j)
                            flux[flowFirst + j] -= diffusion[j];
                        for (int k = 0; k < independent; ++k)
                            flux[speciesFirst + k] -= speciesFlux[k];
                    }
                    for (index j = 0; j < width; ++j)
                        integrated[j] -= metric.GetFaceWeight(sample.point) * flux[j];
                }
                if (frame.moving)
                {
                    Gather(meanBlocks, iCell, face);
                    const auto source = frame.RotationSource(face.Flow(flowFirst));
                    for (int j = 0; j < 5; ++j)
                        integrated[flowFirst + j] += metric.GetCellVol(iCell) * source[j];
                }
                for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
                    for (index j = 0; j < outputs[iBlock].ComponentNumber(); ++j)
                        outputs[iBlock](iCell, j) = integrated[logicalFirst[iBlock] + j];
            };
            if (region == CFV::ReconstructionRegion::All)
                for (index iCell = 0; iCell < owner->metrics.NumCell(); ++iCell)
                    compute(iCell);
            else
                for (const auto iCell : region == CFV::ReconstructionRegion::Core ? owner->partition.Core() : owner->partition.HaloDependent())
                    compute(iCell);
        }
        void AddChemicalSource(const EquationDofs &state, EquationDofs &result, EquationPointEvaluator &evaluator)
        {
            if (!sourceSampling)
                return;
            for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                sourceSampling->Evaluate(state.PhysicalField(iField), fluxCoefficients.PhysicalField(iField), sourceValues[iField]);
            std::vector<R::Array<real>::ReadLease> sources;
            for (const auto &field : sourceValues)
                sources.push_back(field.Read());
            const auto weightRead = sourceWeights.Read();
            const auto weights = weightRead.View<R::FixedRows<1>>();
            auto output = result.WriteOwned();
            const auto speciesOutput = output.Block("species");
            const int independent = owner->equation.Mixture().SpeciesNumber() - 1;
            PointScratch point(state.ComponentNumber());
            std::vector<EquationBlockLocation> locations;
            for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
                locations.push_back(state.LocateBlock(iBlock));
            for (index iCell = 0; iCell < owner->metrics.NumCell(); ++iCell)
                for (index iPoint = sourceOffsets[iCell]; iPoint < sourceOffsets[iCell + 1]; ++iPoint)
                {
                    for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
                    {
                        const auto location = locations[iBlock];
                        const auto values = sources[location.physicalField].View<R::UniformTableRows>();
                        for (index j = 0; j < location.componentNumber; ++j)
                            point.values[logicalFirst[iBlock] + j] = values(iPoint, location.firstComponent + j);
                    }
                    const auto properties = evaluator.Evaluate(point.Flow(flowFirst), point.Species(speciesFirst, independent), 0, {false, false, false, true, false});
                    for (int k = 0; k < independent; ++k)
                        speciesOutput(iCell, k) += weights(iPoint, 0) * properties.chemicalSource[k];
                }
        }
    };

    std::shared_ptr<FiniteVolume::EquationWorkspace> FiniteVolume::BuildEquationWorkspace(EquationDofs &state,
                                                                                          std::optional<Physics::Chemistry::ChemicalWorkerPool> workers) const
    {
        const MPIInfo mpi(Metrics().Communicator());
        MPI::LocalStage(mpi, [&]
                        {
            _data->equation.ValidateStateSchema(state);
            _data->equation.ResolveKernelProfile(state.Device());
            DNDS_check_throw_info(state.Space().Same(Metrics().Space()), "equation FV state belongs to another row space");
            const auto &n = _data->settings.numerics;
            DNDS_check_throw_info((n.reconstruction == "constant" || n.reconstruction == "gradient" || n.reconstruction == "variational") &&
                                  n.reconstructionIterations >= 0, "invalid equation reconstruction controls");
            DNDS_check_throw_info(_data->flux.Kind() == Riemann::Solver::Rusanov && !n.positivity && !n.faceFallback &&
                (n.limiter == "none" || (n.limiter == "WBAP" && !n.characteristicLimiter)),
                "requested extended FV flux/PP/limiter profile is not connected yet");
            DNDS_check_throw_info(!_data->equation.Viscous() || n.viscousGradient == "polynomial", "extended viscosity requires the conservative polynomial gradient");
            DNDS_check_throw_info(!(_data->equation.Viscous() && n.limiter == "WBAP") || n.viscousLimited,
                                  "separate unlimited viscous polynomial is not connected in the equation workspace");
            DNDS_check_throw_info(_data->settings.equation.source == "none" ||
                (_data->settings.equation.source == "chemical" && _data->equation.HasMixture()), "unavailable equation volume source");
            for (const auto &sample : _data->samples)
                if (sample.other == UnInitIndex)
                    DNDS_check_throw_info(sample.boundary.kind == BoundaryKind::Extrapolate ||
                        ((sample.boundary.kind == BoundaryKind::SlipWall || sample.boundary.kind == BoundaryKind::NoSlipWall) && !sample.boundary.isothermal),
                        "requested equation boundary producer is not connected yet");
            if (_data->equation.HasMixture())
            {
                DNDS_check_throw_info(workers, "mixture FV requires explicit chemistry workers");
                auto checked = _data->equation.MakePointEvaluator(workers->Acquire(), {});
            }
            else DNDS_check_throw_info(!workers, "perfect-gas FV does not require chemistry workers"); });
        AgreeDescriptor(state, mpi.comm);
        std::shared_ptr<EquationWorkspace> result;
        MPI::LocalStage(mpi, [&]
                        { result = std::make_shared<EquationWorkspace>(_data, state, std::move(workers)); });
        result->BindHalos(state);
        return result;
    }
    const EquationDofs &FiniteVolume::ReconstructionGuess(const EquationWorkspace &work) const
    {
        DNDS_check_throw_info(work.owner == _data, "foreign equation workspace");
        return work.coefficients;
    }
    const EquationDofs &FiniteVolume::FluxReconstruction(const EquationWorkspace &work) const
    {
        DNDS_check_throw_info(work.owner == _data, "foreign equation workspace");
        return work.fluxCoefficients;
    }

    void FiniteVolume::Evaluate(EquationDofs &state, EquationDofs &residual, EquationWorkspace &work,
                                R::EagerEngine &engine, const R::ExecutionStream &stream, real time) const
    {
        const MPIInfo mpi(Metrics().Communicator());
        // Observe only this lane's producer prefix. MPI remains on the caller's
        // controller thread, including when the CPU launcher uses worker threads.
        MPI::LocalStage(mpi, [&]
                        { stream.Record().WaitHost(); });
        std::optional<EquationPointEvaluator> evaluator;
        MPI::LocalStage(mpi, [&]
                        {
            DNDS_check_throw_info(work.owner == _data && std::isfinite(time) && stream.Context().device == R::Device::Host(),
                                  "foreign equation workspace, invalid time or missing CPU backend");
            work.binding.Check(state);
            _data->equation.ValidateStateSchema(residual);
            state.RequireCompatible(residual);
            EquationDofOperations::RequireDistinct(state, residual);
            EquationDofOperations::RequireDistinct(work.coefficients, residual);
            if (!work.limited.empty()) EquationDofOperations::RequireDistinct(work.fluxCoefficients, residual);
            auto write = residual.WriteOwned(); write.CancelUnmodified();
            PreflightHalo(state); PreflightHalo(work.coefficients);
            if (!work.limited.empty()) PreflightHalo(work.fluxCoefficients);
            _data->partition.Validate();
            const bool viscous = _data->equation.Viscous();
            evaluator.emplace(_data->equation.MakePointEvaluator(work.workers ? work.workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{},
                {viscous, viscous, viscous, work.chemical, false})); });
        const auto local = [&](auto &&action)
        {
            if (_data->settings.implicit.collectiveChecks)
                MPI::LocalStage(mpi, action);
            else
                action();
        };
        const auto begin = [&](std::vector<R::TransferBinding> &bindings)
        {
            std::vector<R::TransferOperation> operations;
            operations.reserve(bindings.size());
            for (auto &binding : bindings)
                operations.push_back(engine.ChecksAccess() ? binding.BeginChecked(stream) : binding.BeginScheduled(stream.Context()));
            return operations;
        };
        const auto wait = [](std::vector<R::TransferOperation> &operations)
        { for (auto &operation : operations) operation.Wait(); operations.clear(); };
        const auto submit = [&](auto action)
        { engine.Submit(stream, [action = std::move(action)](auto)
                        { action(); }); };
        const bool overlap = _data->settings.execution.overlap;
        const auto rest = overlap ? CFV::ReconstructionRegion::HaloDependent : CFV::ReconstructionRegion::All;
        const auto sweep = [&](CFV::ReconstructionRegion region)
        {
            for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            {
                auto &field = work.fields[iField];
                _data->reconstruction.DoReconstructionIterTo(state.PhysicalField(iField), field.reconstruction.coefficients,
                                                             field.reconstruction, field.boundary, region);
            }
        };
        const auto gradient = [&](CFV::ReconstructionRegion region)
        {
            for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            {
                auto &field = work.fields[iField];
                _data->gradient.Evaluate(state.PhysicalField(iField), *field.gradientBoundary, *field.gradient, field.vectorStarts, region);
            }
        };
        // Locals captured by worker commands remain alive even on exceptions.
        StreamJoin join{stream};
        auto meanOperations = begin(work.meanHalo);
        std::vector<R::TransferOperation> firstCoefficients;
        const auto &numerics = _data->settings.numerics;
        const bool variational = numerics.reconstruction == "variational";
        const bool firstSweep = overlap && variational && numerics.reconstructionIterations > 0;
        if (firstSweep)
            firstCoefficients = begin(work.coefficientHalo);
        submit([&, variational]
               { work.FillBoundary(state, variational && _data->settings.numerics.reconstructionBoundaryFromPolynomial, time); });
        if (overlap && numerics.reconstruction == "gradient")
            submit([&]
                   { gradient(CFV::ReconstructionRegion::Core); });
        if (firstSweep)
            submit([&]
                   { sweep(CFV::ReconstructionRegion::Core); });
        wait(meanOperations);
        if (numerics.reconstruction == "gradient")
        {
            submit([&, rest]
                   { gradient(rest); });
            submit([&]
                   {
                for (auto &field : work.fields)
                    _data->gradient.ConvertToCoefficients(*field.gradient, _data->reconstruction, field.reconstruction.coefficients); });
            local([&]
                  { stream.Record().WaitHost(); });
        }
        else if (variational)
        {
            for (int iteration = 0; iteration < numerics.reconstructionIterations; ++iteration)
            {
                std::vector<R::TransferOperation> operations;
                if (iteration == 0 && firstSweep)
                    operations = std::move(firstCoefficients);
                else
                {
                    local([&]
                          { stream.Record().WaitHost(); });
                    if (numerics.reconstructionBoundaryFromPolynomial)
                        local([&]
                              { work.FillBoundary(state, true, time); });
                    operations = begin(work.coefficientHalo);
                    if (overlap)
                        submit([&]
                               { sweep(CFV::ReconstructionRegion::Core); });
                }
                wait(operations);
                submit([&, rest]
                       { sweep(rest); });
                local([&]
                      { stream.Record().WaitHost(); });
                bool converged = false;
                if (numerics.reconstructionTolerance > 0 && (iteration % 4 == 3 || iteration + 1 == numerics.reconstructionIterations))
                {
                    real norms[2]{};
                    local([&]
                          {
                        for (const auto &field : work.fields)
                        {
                            const auto aRead = field.reconstruction.coefficients.Owned().Read(), bRead = field.reconstruction.temporary.Owned().Read();
                            const auto a = aRead.View<R::UniformTableRows>(), b = bRead.View<R::UniformTableRows>();
                            for (index iCell = 0; iCell < a.RowCount(); ++iCell)
                                for (index j = 0; j < a.Width(); ++j)
                                {
                                    const real delta = b(iCell, j) - a(iCell, j);
                                    norms[0] += delta * delta; norms[1] += b(iCell, j) * b(iCell, j);
                                }
                        } });
                    MPI_Allreduce(MPI_IN_PLACE, norms, 2, MPI_DOUBLE, MPI_SUM, mpi.comm);
                    converged = std::sqrt(norms[0] / std::max(norms[1], real(1e-300))) <= numerics.reconstructionTolerance;
                }
                local([&]
                      { for (auto &field : work.fields) _data->reconstruction.CommitReconstruction(field.reconstruction); });
                work.iterations = iteration + 1;
                if (converged)
                    break;
            }
        }
        else
            local([&]
                  {
                stream.Record().WaitHost();
                for (auto &field : work.fields) Algebra::Fill(field.reconstruction.coefficients, 0.); });
        if (!work.limited.empty())
        {
            local([&]
                  { EquationDofOperations::Copy(work.fluxCoefficients, work.coefficients); });
            for (int order = numerics.cfv.maxOrder; order >= 1; --order)
            {
                auto operations = begin(work.fluxHalo);
                const auto limit = [&, order](CFV::ReconstructionRegion region)
                { for (auto &field : work.limited) _data->reconstruction.DoLimiterWBAP_3OrderTo(field.reconstruction, order, nullptr, region); };
                if (overlap)
                    submit([limit]
                           { limit(CFV::ReconstructionRegion::Core); });
                wait(operations);
                submit([limit, rest]
                       { limit(rest); });
                local([&]
                      { stream.Record().WaitHost(); });
                local([&]
                      { for (auto &field : work.limited) _data->reconstruction.CommitReconstruction(field.reconstruction); });
            }
        }
        auto operations = begin(work.limited.empty() ? work.coefficientHalo : work.fluxHalo);
        if (overlap)
            submit([&]
                   { work.ApplyRHS(state, residual, CFV::ReconstructionRegion::Core, *evaluator, time); });
        wait(operations);
        submit([&, rest]
               { work.ApplyRHS(state, residual, rest, *evaluator, time); });
        submit([&]
               { work.AddChemicalSource(state, residual, *evaluator); });
        local([&]
              { stream.Record().WaitHost(); });
    }
}
