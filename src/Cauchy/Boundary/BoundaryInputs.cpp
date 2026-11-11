#include "Cauchy/Discretization/FiniteVolumeInternal.hpp"
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
    const std::vector<BoundaryPoint> &FiniteVolume::BoundaryPoints() const
    {
        Metrics().Validate();
        return _data->boundaryLayout->points;
    }
    const std::vector<BoundaryPoint> &FiniteVolume::BoundaryFaces() const
    {
        Metrics().Validate();
        return _data->boundaryLayout->faces;
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
                            index(_data->boundaryLayout->points.size()),
                            "boundary quadrature"),
            BuildPacketRows(_data->equation.Schema(),
                            index(_data->boundaryLayout->faces.size()),
                            "boundary face centers"),
            _data->equation.PropertyKey(), std::move(thermal));
        input._layout = _data->boundaryLayout;
        BeginBoundaryUpdate(input);
        return input;
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
        validate(input._quadrature, index(input._layout->points.size()));
        validate(input._faceCenters, index(input._layout->faces.size()));
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
            for (index iPoint = 0; iPoint < rows.RowCount(); ++iPoint)
                for (index j = 0; j < rows.ComponentNumber(); ++j)
                    rows(iPoint, j) = std::numeric_limits<real>::quiet_NaN();
        }
    }
    void FiniteVolume::PublishBoundaryInputs(BoundaryInputs &input, real time, StateEpoch epoch) const
    {
        ValidateBoundaryLayout(input);
        DNDS_check_throw_info(input._updating && std::isfinite(time), "boundary publication needs a begun update and finite physical time");
        const auto validate = [&](const EquationDofs &values,
                                  const auto &points)
        {
            const auto read = values.ReadOwned();
            const auto layout = read.ComponentLayout();
            const auto rows = layout.View();
            const auto thermal = input._thermal->View();
            const auto rans = _data->equation.HasRANS()
                                  ? RANSStateView{_data->equation.RANS().Kind(),
                                                  _data->equation.RANS().ComponentNumber()}
                                  : RANSStateView{};
            const auto ransBlock = rans.components ? read.Block("rans")
                                                   : EquationBlockReadView{};
            for (index iPoint = 0; iPoint < rows.RowCount(); ++iPoint)
                if (points[iPoint].interface)
                {
                    DNDS_check_throw_info(
                        thermal.Admissible(
                            EquationPositivityDetail::PointRow<decltype(rows)>{
                                rows, iPoint}),
                        "interface update has missing, nonfinite or thermally inadmissible equation rows");
                    if (rans.components)
                        DNDS_check_throw_info(rans.Admissible(ransBlock[iPoint]),
                                              "interface update has inadmissible RANS rows");
                }
        };
        validate(input._quadrature, input._layout->points);
        validate(input._faceCenters, input._layout->faces);
        BoundaryInputs::Publication publication{time, epoch};
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
                DNDS_check_throw_info(a.size() == b.size(), "boundary alias point counts differ");
                const auto valueRead = values.ReadOwned();
                const auto layout = valueRead.ComponentLayout();
                const auto rows = layout.View();
                const auto thermal = input._thermal->View();
                for (index iPoint = 0; iPoint < index(a.size()); ++iPoint)
                {
                    DNDS_check_throw_info(a[iPoint].cell == b[iPoint].cell && a[iPoint].face == b[iPoint].face && a[iPoint].quadrature == b[iPoint].quadrature &&
                                              a[iPoint].zone == b[iPoint].zone && a[iPoint].point == b[iPoint].point && a[iPoint].normal == b[iPoint].normal && a[iPoint].interface == b[iPoint].interface,
                                          "boundary alias point order, geometry or semantics differ");
                    if (b[iPoint].interface)
                        DNDS_check_throw_info(
                            thermal.Admissible(
                                EquationPositivityDetail::PointRow<decltype(rows)>{
                                    rows, iPoint}),
                            "aliased boundary state violates the target level's admissibility bounds");
                }
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
