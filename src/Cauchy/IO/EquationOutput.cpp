#include "Cauchy/Discretization/EquationFiniteVolumeInternal.hpp"
#include "Cauchy/Physics/EquationProperties.hpp"
#include "Cauchy/Physics/EquationPositivity.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include <fstream>
#include <iomanip>

namespace DNDS::Cauchy
{
    void FiniteVolume::WriteVisualization(const EquationDofs &state, const EquationWorkspace &workspace,
                                          const std::string &prefix, real time, int hdfDeflateLevel,
                                          const std::optional<Physics::Chemistry::ChemicalWorkerPool> &workers) const
    {
        using namespace EquationFVDetail;
        const MPIInfo mpi(Metrics().Communicator());
        const auto &equation = _data->equation;
        const int species = equation.HasMixture() ? equation.Mixture().SpeciesNumber() : 0;
        const int independent = std::max(0, species - 1);
        std::vector<std::string> names;
        std::vector<index> first;
        index flowFirst = -1, speciesFirst = -1, ransFirst = -1;
        std::optional<EquationPointEvaluator> evaluator;
        std::optional<EquationThermalPolicy> thermal;
        Runtime::Array<real> derived;
        MPI::LocalStage(mpi, [&]
                        {
            ValidateWorkspace(state, workspace);
            DNDS_check_throw_info(hdfDeflateLevel >= 0 && hdfDeflateLevel <= 9, "VTKHDF compression level must be in [0,9]");
            for (const auto &block : state.Schema().blocks)
            {
                first.push_back(index(names.size()));
                if (block.name == "flow") flowFirst = index(names.size());
                if (block.name == "species") speciesFirst = index(names.size());
                if (block.name == "rans") ransFirst = index(names.size());
                for (const auto &component : block.components) names.push_back(component.name);
            }
            for (const auto *name : {"p", "T", "u", "v", "w", "Mach"}) names.push_back(name);
            if (species)
                for (const auto &name : equation.Mixture().SpeciesNames()) names.push_back("Y:" + name);
            auto sorted = names; std::sort(sorted.begin(), sorted.end());
            DNDS_check_throw_info(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(), "ambiguous equation output field names");
            const auto &limits = _data->settings.numerics;
            thermal.emplace(equation, limits.densityFloor, limits.temperatureFloor, 0, limits.pressureFloor,
                workers ? workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{});
            evaluator.emplace(equation.MakePointEvaluator(workers ? workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{}, {equation.HasRANS()}));
            derived = Runtime::Array<real>(Runtime::RowLayout::Dense(Metrics().NumCell(), 6 + species)); });
        const auto means = state.ReadOwned();
        const auto blocks = BlockViews(state, means);
        PointScratch point(state.ComponentNumber());
        MPI::LocalStage(mpi, [&]
                        {
            auto write = derived.Write();
            const auto values = write.View<Runtime::UniformTableRows>();
            for (index iCell = 0; iCell < Metrics().NumCell(); ++iCell)
            {
                Gather(blocks, iCell, point);
                const auto flow = point.Flow(flowFirst);
                const auto properties = evaluator->Evaluate(flow, point.Species(speciesFirst, independent));
                values(iCell, 0) = properties.thermo.pressure;
                values(iCell, 1) = properties.thermo.temperature;
                real speedSquared = 0;
                for (int d = 0; d < 3; ++d)
                {
                    const real speed = flow[d + 1] / flow[0];
                    values(iCell, d + 2) = speed;
                    speedSquared += speed * speed;
                }
                values(iCell, 5) = std::sqrt(speedSquared) / properties.thermo.soundSpeed;
                for (int k = 0; k < species; ++k) values(iCell, 6 + k) = properties.fractions[k];
            } });
        const auto read = derived.Read();
        const auto values = read.View<Runtime::UniformTableRows>();
        const auto components = means.ComponentLayout();
        const auto conservative = components.View();
        auto outputSettings = _data->mesh->hdf5OutSetting;
        outputSettings.deflateLevel = hdfDeflateLevel;
        _data->mesh->PrintParallelVTKHDFDataArray(prefix, "", int(names.size()), 0, 0, 0, [&](int i)
                                                  { return names.at(i); }, [&](int i, index iCell)
                                                  { return i < state.ComponentNumber() ? conservative(iCell, i) : values(iCell, i - state.ComponentNumber()); }, {}, {}, {}, {}, {}, {}, time, mpi.comm, &outputSettings);
        MPI::LocalStage(mpi, [&]
                        {
            std::ofstream surface(prefix + "_surface_rank" + std::to_string(mpi.rank) + ".csv");
            DNDS_check_throw_info(bool(surface), "cannot open equation surface output");
            surface << "cell_id,zone,x,y,z,nx,ny,nz,weight,rho,p,u,v,w,T";
            if (species) for (const auto &name : equation.Mixture().SpeciesNames()) surface << ",Y:" << name;
            surface << '\n' << std::setprecision(17);
            const auto &coefficients = FluxReconstruction(workspace);
            const auto rec = coefficients.ReadOwned();
            const auto recBlocks = BlockViews(coefficients, rec);
            auto basis = _data->reconstruction.AcquireBasisRead();
            auto geometry = Metrics().AcquireRead(); const auto metric = geometry.HostView();
            CFV::Math::SmallMatrix scratch;
            const int modes = _data->reconstruction.NRecDOF();
            const auto frame = _data->motion.At(time);
            const auto *ransModel = equation.HasRANS() ? &equation.RANS() : nullptr;
            const auto ransMean = ransModel ? means.Block("rans") : EquationBlockReadView{};
            PointScratch cellMean(state.ComponentNumber());
            std::optional<WallDistanceRead> wallRead;
            if (ransModel && ransModel->ComponentNumber() == 2) wallRead.emplace(_data->wallDistance->AcquireRead());
            const auto walls = wallRead ? wallRead->HostView() : WallDistanceView{};
            for (const auto &sample : _data->samples)
            {
                if (sample.other != UnInitIndex) continue;
                Gather(blocks, sample.cell, point);
                const auto b = basis.FaceBasis(sample.face, sample.side, sample.point, 0, scratch);
                for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
                    for (index j = 0; j < blocks[iBlock].ComponentNumber(); ++j)
                        for (int mode = 0; mode < modes; ++mode)
                            point.values[first[iBlock] + j] += recBlocks[iBlock](sample.cell, j * modes + mode) * b(0, mode);
                if (!thermal->View().Admissible(point.values)) Gather(blocks, sample.cell, point);
                if (ransModel)
                    RANSStateView{ransModel->Kind(), ransModel->ComponentNumber()}.RepairReconstruction(ransMean[sample.cell], point.values.data() + ransFirst);
                if (wallRead) Gather(blocks, sample.cell, cellMean);
                BoundaryValue(point, flowFirst, speciesFirst, independent, sample, *evaluator, frame, time, false,
                              {ransModel, ransFirst, &cellMean, wallRead ? walls.cellMean(sample.cell, 0) : 0});
                const auto flow = point.Flow(flowFirst);
                const auto properties = evaluator->Evaluate(flow, point.Species(speciesFirst, independent));
                surface << Mesh().OriginalEntityId(Geom::EntityKind::Cell, sample.cell) << ',' << sample.zone;
                for (int d = 0; d < 3; ++d) surface << ',' << sample.ownPoint[d];
                for (int d = 0; d < 3; ++d) surface << ',' << sample.normal[d];
                surface << ',' << metric.GetFaceWeight(sample.point) << ',' << flow[0] << ',' << properties.thermo.pressure;
                for (int d = 0; d < 3; ++d) surface << ',' << flow[d + 1] / flow[0];
                surface << ',' << properties.thermo.temperature;
                for (int k = 0; k < species; ++k) surface << ',' << properties.fractions[k];
                surface << '\n';
            }
            DNDS_check_throw_info(bool(surface), "equation surface output write failed"); });
    }
}
