#include "Cauchy/Discretization/FiniteVolumeInternal.hpp"

namespace DNDS::Cauchy
{
    const std::vector<BoundaryPoint> &FiniteVolume::BoundaryPoints() const
    {
        Metrics().Validate();
        return _data->boundaryLayout->points;
    }
    bool FiniteVolume::HasInterfaces() const { return _data->hasInterfaces; }
    BoundaryInputs FiniteVolume::BuildBoundaryInputs() const
    {
        Metrics().Validate();
        BoundaryInputs input;
        input._layout = _data->boundaryLayout;
        input.flow = Runtime::Array<real>(Runtime::RowLayout::Dense(input._layout->points.size(), 5));
        BeginBoundaryUpdate(input);
        return input;
    }
    void FiniteVolume::ValidateBoundaryLayout(const BoundaryInputs &input) const
    {
        Metrics().Validate();
        DNDS_check_throw_info(input._layout == _data->boundaryLayout && input._layout->stamp.Valid(), "boundary packet point order or geometry differs");
        DNDS_check_throw_info(input.flow.GetDevice() == Runtime::Device::Host() && !input.flow.Layout().IsVariable() &&
                                  input.flow.RowCount() == index(input._layout->points.size()) && input.flow.Layout().Width() == 5,
                              "boundary packet requires compatible CPU flow5 rows");
    }
    void FiniteVolume::BeginBoundaryUpdate(BoundaryInputs &input) const
    {
        ValidateBoundaryLayout(input);
        auto write = input.flow.Write();
        input._publication.reset();
        input._updating = true;
        const auto values = write.View<Runtime::UniformTableRows>();
        for (index iPoint = 0; iPoint < values.RowCount(); ++iPoint)
            for (int j = 0; j < 5; ++j)
                values(iPoint, j) = std::numeric_limits<real>::quiet_NaN();
    }
    void FiniteVolume::PublishBoundaryInputs(BoundaryInputs &input, real time, StateEpoch epoch) const
    {
        ValidateBoundaryLayout(input);
        DNDS_check_throw_info(input._updating && std::isfinite(time), "boundary publication needs a begun update and finite physical time");
        const auto read = input.flow.Read();
        const auto values = read.View<Runtime::UniformTableRows>();
        for (index iPoint = 0; iPoint < values.RowCount(); ++iPoint)
            if (input._layout->points[iPoint].interface)
            {
                FlowState state;
                for (int j = 0; j < 5; ++j)
                    state[j] = values(iPoint, j);
                DNDS_check_throw_info(ThermallyAdmissible(state, _data->thermalBounds), "interface update has missing, nonfinite or inadmissible flow rows");
            }
        input._publication = BoundaryInputs::Publication{time, epoch, input.flow.Version(), input.flow.ShapeVersion()};
        input._updating = false;
    }
    void FiniteVolume::ValidateBoundaryInputs(const BoundaryInputs &input, real time) const
    {
        ValidateBoundaryLayout(input);
        const auto read = input.flow.Read(); // An in-flight writer has not advanced its generation yet.
        DNDS_check_throw_info(input._publication && !input._updating && input._publication->time == time &&
                                  input._publication->version == input.flow.Version() && input._publication->shape == input.flow.ShapeVersion(),
                              "boundary packet is unpublished, stale, modified or from another physical stage");
    }
    BoundaryInputs FiniteVolume::AliasBoundaryInputs(const BoundaryInputs &input) const
    {
        Metrics().Validate();
        DNDS_check_throw_info(input._layout && input._layout->stamp.Valid() && input._layout->mesh.lock() == _data->mesh,
                              "boundary alias requires the same live reference mesh");
        const auto read = input.flow.Read();
        DNDS_check_throw_info(input._publication && !input._updating && input._publication->version == input.flow.Version() &&
                                  input._publication->shape == input.flow.ShapeVersion(),
                              "cannot alias an unpublished or modified boundary packet");
        const auto &a = input._layout->points, &b = _data->boundaryLayout->points;
        DNDS_check_throw_info(a.size() == b.size(), "boundary alias point counts differ");
        if (input._layout != _data->boundaryLayout)
        {
            const auto values = read.View<Runtime::UniformTableRows>();
            for (index iPoint = 0; iPoint < index(a.size()); ++iPoint)
            {
                DNDS_check_throw_info(a[iPoint].cell == b[iPoint].cell && a[iPoint].face == b[iPoint].face && a[iPoint].quadrature == b[iPoint].quadrature &&
                                          a[iPoint].zone == b[iPoint].zone && a[iPoint].point == b[iPoint].point && a[iPoint].normal == b[iPoint].normal && a[iPoint].interface == b[iPoint].interface,
                                      "boundary alias point order, geometry or semantics differ");
                if (b[iPoint].interface)
                {
                    FlowState state;
                    for (int j = 0; j < 5; ++j)
                        state[j] = values(iPoint, j);
                    DNDS_check_throw_info(ThermallyAdmissible(state, _data->thermalBounds), "aliased boundary state violates the target level's admissibility bounds");
                }
            }
        }
        auto alias = input;
        alias._layout = _data->boundaryLayout;
        ValidateBoundaryInputs(alias, alias.Time());
        return alias;
    }
}
