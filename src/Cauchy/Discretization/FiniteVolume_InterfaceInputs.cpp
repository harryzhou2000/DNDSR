#include "Cauchy/Core/NumericalFailure.hpp"
#include "Cauchy/Discretization/FiniteVolumeInternal.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeBoundaryKernels.hpp"
#include "Cauchy/State/CPU/FieldKernels.hpp"
#include "Cauchy/Physics/EquationPositivity.hpp"
#include "Cauchy/Physics/EquationRANS.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Geom/Interface/BoundaryInterface.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"

namespace DNDS::Cauchy
{
    namespace
    {
        EquationDofs BuildPacketRows(const EquationSchema &schema, index rows,
                                     const std::string &name)
        {
            index components = 0;
            for (const auto &block : schema.blocks)
                components += index(block.components.size());
            const Algebra::DofSpace space(size_t(rows), 0, name);
            return EquationDofs::Coupled(
                schema,
                Algebra::DofField<real>(
                    space, Runtime::RowLayout::Dense(0, components)),
                "equation");
        }
        void Snapshot(const EquationDofs &values, std::vector<uint64_t> &versions,
                      std::vector<uint64_t> &shapes)
        {
            versions.clear();
            shapes.clear();
            versions.reserve(size_t(values.PhysicalFieldNumber()));
            shapes.reserve(size_t(values.PhysicalFieldNumber()));
            for (index iField = 0; iField < values.PhysicalFieldNumber(); ++iField)
            {
                const auto &array = values.PhysicalField(iField).Owned();
                versions.push_back(array.Version());
                shapes.push_back(array.ShapeVersion());
            }
        }
        bool Current(const EquationDofs &values,
                     const std::vector<uint64_t> &versions,
                     const std::vector<uint64_t> &shapes)
        {
            if (versions.size() != size_t(values.PhysicalFieldNumber()) ||
                shapes.size() != versions.size())
                return false;
            for (index iField = 0; iField < values.PhysicalFieldNumber(); ++iField)
            {
                const auto &array = values.PhysicalField(iField).Owned();
                if (versions[size_t(iField)] != array.Version() ||
                    shapes[size_t(iField)] != array.ShapeVersion())
                    return false;
            }
            return true;
        }
    }
    Geom::BoundaryPatch FiniteVolume::BuildBoundaryPatch(int zone) const
    {
        std::vector<index> selected;
        MPI::LocalStage(MPIInfo(Metrics().Communicator()), [&]
                        {
        auto read=_data->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        const auto view=read.HostView();
        for(index iBnd=0;iBnd<view.NumBnd();++iBnd)
            if(view.GetBndZone(iBnd)==zone) selected.push_back(iBnd); });
        return Geom::BoundaryPatch::Create(_data->mesh, selected);
    }
    std::vector<BoundaryPoint> FiniteVolume::BoundaryPoints() const
    {
        Metrics().Validate();
        const auto read = _data->boundaryLayout->points.Read();
        const auto view = read.View<Runtime::FixedRows<1>>();
        std::vector<BoundaryPoint> result(view.RowCount());
        for (index iRow = 0; iRow < view.RowCount(); ++iRow)
            result[iRow] = view(iRow, 0);
        return result;
    }
    std::vector<BoundaryPoint> FiniteVolume::BoundaryFaces() const
    {
        Metrics().Validate();
        const auto read = _data->boundaryLayout->faces.Read();
        const auto view = read.View<Runtime::FixedRows<1>>();
        std::vector<BoundaryPoint> result(view.RowCount());
        for (index iRow = 0; iRow < view.RowCount(); ++iRow)
            result[iRow] = view(iRow, 0);
        return result;
    }
    Runtime::Array<BoundaryPoint>::ReadLease FiniteVolume::ReadBoundaryPoints() const
    {
        Metrics().Validate();
        return _data->boundaryLayout->points.Read();
    }
    Runtime::Array<BoundaryPoint>::ReadLease FiniteVolume::ReadBoundaryFaces() const
    {
        Metrics().Validate();
        return _data->boundaryLayout->faces.Read();
    }
    bool FiniteVolume::HasInterfaces() const { return _data->hasInterfaces; }
    BoundaryInputs FiniteVolume::BuildBoundaryInputs(
        std::optional<Physics::Chemistry::ChemicalWorkerPool> workers) const
    {
        Metrics().Validate();
        DNDS_check_throw_info(_data->equation.HasMixture() == bool(workers),
                              "interface packet needs exactly its equation chemistry backend");
        auto thermal = std::make_shared<const EquationThermalPolicy>(
            _data->equation, _data->settings.numerics.densityFloor,
            _data->settings.numerics.temperatureFloor,
            _data->settings.numerics.sensibleEnergyFloor,
            _data->settings.numerics.pressureFloor,
            workers ? workers->Acquire()
                    : Physics::Chemistry::ChemicalWorkerPool::Lease{});
        BoundaryInputs input(
            BuildPacketRows(_data->equation.Schema(),
                            _data->boundaryLayout->points.RowCount(),
                            "boundary quadrature"),
            BuildPacketRows(_data->equation.Schema(),
                            _data->boundaryLayout->faces.RowCount(),
                            "boundary face centers"),
            _data->equation.PropertyKey(), std::move(thermal));
        input._layout = _data->boundaryLayout;
        BeginBoundaryUpdate(input);
        return input;
    }
    BoundaryInputs FiniteVolume::FreezeBoundaryInputs(
        const BoundaryInputs &input) const
    {
        ValidateBoundaryInputs(input, input.Time());
        BoundaryInputs frozen(
            EquationDofOperations::NewVector(input._quadrature),
            EquationDofOperations::NewVector(input._faceCenters),
            input._property, input._thermal);
        frozen._layout = _data->boundaryLayout;
        BeginBoundaryUpdate(frozen);
        EquationDofOperations::Copy(frozen._quadrature, input._quadrature);
        EquationDofOperations::Copy(frozen._faceCenters, input._faceCenters);
        PublishBoundaryInputs(frozen, input.Time(), input.Epoch(),
                              input.InterfaceSources());
        return frozen;
    }
    void FiniteVolume::ValidateBoundaryLayout(const BoundaryInputs &input) const
    {
        Metrics().Validate();
        DNDS_check_throw_info(input._layout == _data->boundaryLayout &&
                                  input._layout->stamp.Valid(),
                              "boundary packet point order or geometry differs");
        DNDS_check_throw_info(input._property == _data->equation.PropertyKey() &&
                                  input._thermal,
                              "boundary packet physical model differs");
        const auto validate = [&](const EquationDofs &values, index rows)
        {
            _data->equation.ValidateStateSchema(values);
            DNDS_check_throw_info(values.Device() == Runtime::Device::Host() &&
                                      values.Space().OwnedRows() == size_t(rows) &&
                                      values.Space().GhostRows() == 0,
                                  "boundary packet requires compatible CPU equation rows");
            for (index iField = 0; iField < values.PhysicalFieldNumber(); ++iField)
                DNDS_check_throw_info(
                    !values.PhysicalField(iField).Owned().Layout().IsVariable(),
                    "boundary packet field needs uniform rows");
        };
        validate(input._quadrature, input._layout->points.RowCount());
        validate(input._faceCenters, input._layout->faces.RowCount());
    }
    void FiniteVolume::BeginBoundaryUpdate(BoundaryInputs &input) const
    {
        ValidateBoundaryLayout(input);
        input._publication.reset();
        input._updating = true;
        for (auto *values : {&input._quadrature, &input._faceCenters})
        {
            auto write = values->WriteOwned();
            const auto layout = write.ComponentLayout();
            const auto rows = layout.View();
            CPU::FillRows(rows, rows.ComponentNumber(), std::numeric_limits<real>::quiet_NaN());
        }
    }
    void FiniteVolume::PublishBoundaryInputs(
        BoundaryInputs &input, real time, StateEpoch epoch,
        std::vector<BoundaryInterfaceSource> sources) const
    {
        ValidateBoundaryLayout(input);
        DNDS_check_throw_info(input._updating && std::isfinite(time), "boundary publication needs a begun update and finite physical time");
        // PP POLICY -- an interface packet is a frozen complete mean-like point
        // state, not a raw reconstruction polynomial. Flow/species therefore
        // satisfy the stored-state simplex and clipped EOS/T invariant, while
        // RANS uses its independent model bounds. Sampling never writes a
        // temporary EOS composition back into these transported packet values.
        const auto validate = [&](const EquationDofs &values,
                                  const auto &points)
        {
            const auto read = values.ReadOwned();
            const auto layout = read.ComponentLayout();
            const auto rows = layout.View();
            const auto pointRead = points.Read();
            const auto pointView = pointRead.template View<Runtime::FixedRows<1>>();
            const auto thermal = input._thermal->View();
            const auto rans = _data->equation.HasRANS()
                                  ? RANSStateView{_data->equation.RANS().Kind(),
                                                  _data->equation.RANS().ComponentNumber()}
                                  : RANSStateView{};
            const auto ransBlock = rans.components ? read.Block("rans")
                                                   : EquationBlockReadView{};
            RequireNumericalSuccess(CPU::ValidateInterfacePacketRows(rows, pointView, thermal, rans, ransBlock), "ValidateInterfacePacketRows");
        };
        validate(input._quadrature, input._layout->points);
        validate(input._faceCenters, input._layout->faces);
        for (const auto &source : sources)
        {
            DNDS_check_throw_info(
                source.donorDomain >= 0 && source.receiverDomain >= 0 &&
                    source.routeGeneration > 0 && source.routeOwner,
                "boundary interface source is incomplete");
            source.donorPose.Validate();
            source.receiverPose.Validate();
        }
        BoundaryInputs::Publication publication{time, epoch};
        publication.sources = std::move(sources);
        Snapshot(input._quadrature, publication.versions, publication.shapes);
        Snapshot(input._faceCenters, publication.faceVersions,
                 publication.faceShapes);
        input._publication = std::move(publication);
        input._updating = false;
    }
    void FiniteVolume::ValidateBoundaryInputs(const BoundaryInputs &input, real time) const
    {
        ValidateBoundaryLayout(input);
        const auto read = input._quadrature.ReadOwned();
        const auto faceRead = input._faceCenters.ReadOwned();
        // Active writers have not advanced their generations yet.
        DNDS_check_throw_info(input._publication && !input._updating && input._publication->time == time &&
                                  Current(input._quadrature, input._publication->versions,
                                          input._publication->shapes) &&
                                  Current(input._faceCenters,
                                          input._publication->faceVersions,
                                          input._publication->faceShapes),
                              "boundary packet is unpublished, stale, modified or from another physical stage");
    }
    BoundaryInputs FiniteVolume::AliasBoundaryInputs(const BoundaryInputs &input) const
    {
        Metrics().Validate();
        DNDS_check_throw_info(input._layout && input._layout->stamp.Valid() && input._layout->mesh.lock() == _data->mesh,
                              "boundary alias requires the same live reference mesh");
        const auto read = input._quadrature.ReadOwned();
        const auto faceRead = input._faceCenters.ReadOwned();
        DNDS_check_throw_info(input._publication && !input._updating &&
                                  Current(input._quadrature,
                                          input._publication->versions,
                                          input._publication->shapes) &&
                                  Current(input._faceCenters,
                                          input._publication->faceVersions,
                                          input._publication->faceShapes),
                              "cannot alias an unpublished or modified boundary packet");
        if (input._layout != _data->boundaryLayout)
        {
            const auto validate = [&](const auto &a, const auto &b,
                                      const EquationDofs &values)
            {
                DNDS_check_throw_info(a.RowCount() == b.RowCount(), "boundary alias point counts differ");
                const auto aRead = a.Read(), bRead = b.Read();
                const auto av = aRead.template View<Runtime::FixedRows<1>>(), bv = bRead.template View<Runtime::FixedRows<1>>();
                const auto valueRead = values.ReadOwned();
                const auto layout = valueRead.ComponentLayout();
                const auto rows = layout.View();
                const auto thermal = input._thermal->View();
                RequireNumericalSuccess(CPU::ValidateInterfaceAliasRows(rows, av, bv, thermal), "ValidateInterfaceAliasRows");
            };
            validate(input._layout->points, _data->boundaryLayout->points,
                     input._quadrature);
            validate(input._layout->faces, _data->boundaryLayout->faces,
                     input._faceCenters);
        }
        auto alias = input;
        alias._layout = _data->boundaryLayout;
        ValidateBoundaryInputs(alias, alias.Time());
        return alias;
    }
}
