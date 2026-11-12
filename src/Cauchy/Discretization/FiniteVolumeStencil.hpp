#pragma once

#include "DNDS/Runtime/Array/Array.hpp"
#include "DNDS/Core/Span.hpp"
#include "Cauchy/Boundary/BoundarySystem.hpp"
#include "Geom/Geometry/Geometric.hpp"

namespace DNDS::Cauchy::FiniteVolumeDetail
{
    struct SampleValue
    {
        index cell, other, point;
        int side;
        int zone;
        Geom::tPoint normal, ownPoint, otherPoint;
        Geom::tGPoint rotation;
        const BoundaryCondition &boundary;
        index auxiliaryFace = UnInitIndex;
        real viscousDistance = 0;
        index face = UnInitIndex;
        /// Dense column slot in this cell's canonical Jacobian row. Filled
        /// once after JacobianColumns() is constructed; UnInitIndex for
        /// diagonal/boundary samples and non-Jacobian quadrature samples.
        index jacobianColumn = UnInitIndex;
        /// Row in the incoming packet used by this sample: RHS flow for
        /// quadrature samples, faceFlow for centered Jacobian samples.
        index boundaryPoint = UnInitIndex;
        /// Dense RHSLocalDiagnostics force slot, resolved once during setup.
        /// Non-wall and interior samples keep UnInitIndex.
        index forceSlot = UnInitIndex;
        /// Exact setup-time frame test. Ordinary and translated-periodic faces
        /// keep components unchanged; even a small nonidentity rotation remains
        /// active. Default true keeps hand-built samples conservative.
        bool changesFrame = true;
    };

    /// Backend-neutral per-incidence geometry. Boundary policies are held once
    /// in the small zone registry, never copied into every quadrature record.
    struct SampleRecord
    {
        index cell, other, point, condition;
        int side, zone;
        std::array<real, 3> normal, ownPoint, otherPoint;
        std::array<real, 9> rotation;
        index auxiliaryFace, face, jacobianColumn, boundaryPoint, forceSlot;
        real viscousDistance;
        bool changesFrame;
    };
    static_assert(std::is_trivially_copyable_v<SampleRecord>);

    /// CPU point expansion of plain field data; no owner, allocation or lease.
    inline SampleValue ExpandSample(const SampleRecord &record, SpanView<const BoundaryCondition> conditions)
    {
        const auto point = [](const auto &v)
        { return Geom::tPoint{v[0], v[1], v[2]}; };
        SampleValue sample{record.cell, record.other, record.point, record.side, record.zone,
                           point(record.normal), point(record.ownPoint), point(record.otherPoint),
                           Eigen::Map<const Geom::tGPoint>(record.rotation.data()), conditions[record.condition]};
        sample.auxiliaryFace = record.auxiliaryFace;
        sample.viscousDistance = record.viscousDistance;
        sample.face = record.face;
        sample.jacobianColumn = record.jacobianColumn;
        sample.boundaryPoint = record.boundaryPoint;
        sample.forceSlot = record.forceSlot;
        sample.changesFrame = record.changesFrame;
        return sample;
    }
    template <class Sample>
    inline SampleRecord PackSample(const Sample &sample, index condition)
    {
        SampleRecord record{sample.cell, sample.other, sample.point, condition, sample.side, sample.zone, {sample.normal[0], sample.normal[1], sample.normal[2]}, {sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]}, {sample.otherPoint[0], sample.otherPoint[1], sample.otherPoint[2]}, {}, sample.auxiliaryFace, sample.face, sample.jacobianColumn, sample.boundaryPoint, sample.forceSlot, sample.viscousDistance, sample.changesFrame};
        for (int j = 0; j < 9; ++j)
            record.rotation[j] = sample.rotation.data()[j];
        return record;
    }
}

namespace DNDS::Cauchy
{
    /// Pure numeric view of FV incidence, quadrature, region and PP fields.
    /// Acquired once by the operator; device kernels need no host vectors.
    struct FiniteVolumeStencilView
    {
        using Indices = Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
        Runtime::ArrayView<const FiniteVolumeDetail::SampleRecord, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> samples, jacobianSamples;
        Indices offsets, jacobianOffsets, stencilFaces, faceEntries, auxiliaryCoreFaces, auxiliaryDependentFaces, rhsCoreFaces, rhsDependentFaces, auxiliaryCellOffsets, auxiliaryCellFaces, positivityOffsets, boundaryEntries, coreCells, dependentCells;
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> auxiliaryFaceArea;
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::FixedRows<3>> positivityDisplacements;
    };

    /** @brief Mesh-sized numerical fields, independent of small physics metadata.
     * @details Setup vectors are published into Arrays before execution. Read
     * scopes retain both the allocation and the immutable layout through every
     * synchronous call, or are captured by the asynchronous operation record.
     */
    struct FiniteVolumeStencilStorage
    {
        Runtime::Array<FiniteVolumeDetail::SampleRecord> samples{Runtime::RowLayout::Dense(0, 1)}, jacobianSamples{Runtime::RowLayout::Dense(0, 1)};
        Runtime::Array<index> offsets{Runtime::RowLayout::Dense(0, 1)};
        Runtime::Array<index> jacobianOffsets{Runtime::RowLayout::Dense(0, 1)};
        Runtime::Array<index> stencilFaces{Runtime::RowLayout::Dense(0, 1)};
        Runtime::Array<index> faceEntries{Runtime::RowLayout::Dense(0, 1)};
        Runtime::Array<index> auxiliaryCoreFaces{Runtime::RowLayout::Dense(0, 1)};
        Runtime::Array<index> auxiliaryDependentFaces{Runtime::RowLayout::Dense(0, 1)};
        Runtime::Array<index> rhsCoreFaces{Runtime::RowLayout::Dense(0, 1)};
        Runtime::Array<index> rhsDependentFaces{Runtime::RowLayout::Dense(0, 1)};
        Runtime::Array<index> auxiliaryCellOffsets{Runtime::RowLayout::Dense(0, 1)};
        Runtime::Array<index> auxiliaryCellFaces{Runtime::RowLayout::Dense(0, 1)};
        Runtime::Array<index> positivityOffsets{Runtime::RowLayout::Dense(0, 1)};
        Runtime::Array<index> boundaryEntries{Runtime::RowLayout::Dense(0, 1)};
        Runtime::Array<index> coreCells{Runtime::RowLayout::Dense(0, 1)};
        Runtime::Array<index> dependentCells{Runtime::RowLayout::Dense(0, 1)};
        Runtime::Array<real> auxiliaryFaceArea{Runtime::RowLayout::Dense(0, 1)};
        Runtime::Array<real> positivityDisplacements{Runtime::RowLayout::Dense(0, 3)};
        struct ReadAccess
        {
            Runtime::Array<FiniteVolumeDetail::SampleRecord>::ReadLease samples, jacobianSamples;
            Runtime::Array<index>::ReadLease offsets, jacobianOffsets, stencilFaces, faceEntries, auxiliaryCoreFaces, auxiliaryDependentFaces, rhsCoreFaces, rhsDependentFaces, auxiliaryCellOffsets, auxiliaryCellFaces, positivityOffsets, boundaryEntries, coreCells, dependentCells;
            Runtime::Array<real>::ReadLease auxiliaryFaceArea, positivityDisplacements;
            explicit ReadAccess(const FiniteVolumeStencilStorage &storage)
                : samples(storage.samples.Read()), jacobianSamples(storage.jacobianSamples.Read()),
                  offsets(storage.offsets.Read()),
                  jacobianOffsets(storage.jacobianOffsets.Read()),
                  stencilFaces(storage.stencilFaces.Read()),
                  faceEntries(storage.faceEntries.Read()),
                  auxiliaryCoreFaces(storage.auxiliaryCoreFaces.Read()),
                  auxiliaryDependentFaces(storage.auxiliaryDependentFaces.Read()),
                  rhsCoreFaces(storage.rhsCoreFaces.Read()),
                  rhsDependentFaces(storage.rhsDependentFaces.Read()),
                  auxiliaryCellOffsets(storage.auxiliaryCellOffsets.Read()),
                  auxiliaryCellFaces(storage.auxiliaryCellFaces.Read()),
                  positivityOffsets(storage.positivityOffsets.Read()),
                  boundaryEntries(storage.boundaryEntries.Read()),
                  coreCells(storage.coreCells.Read()),
                  dependentCells(storage.dependentCells.Read()),
                  auxiliaryFaceArea(storage.auxiliaryFaceArea.Read()), positivityDisplacements(storage.positivityDisplacements.Read()) {}
            FiniteVolumeStencilView View() const
            {
                return {samples.View<Runtime::FixedRows<1>>(), jacobianSamples.View<Runtime::FixedRows<1>>(),
                        offsets.View<Runtime::FixedRows<1>>(),
                        jacobianOffsets.View<Runtime::FixedRows<1>>(),
                        stencilFaces.View<Runtime::FixedRows<1>>(),
                        faceEntries.View<Runtime::FixedRows<1>>(),
                        auxiliaryCoreFaces.View<Runtime::FixedRows<1>>(),
                        auxiliaryDependentFaces.View<Runtime::FixedRows<1>>(),
                        rhsCoreFaces.View<Runtime::FixedRows<1>>(),
                        rhsDependentFaces.View<Runtime::FixedRows<1>>(),
                        auxiliaryCellOffsets.View<Runtime::FixedRows<1>>(),
                        auxiliaryCellFaces.View<Runtime::FixedRows<1>>(),
                        positivityOffsets.View<Runtime::FixedRows<1>>(),
                        boundaryEntries.View<Runtime::FixedRows<1>>(),
                        coreCells.View<Runtime::FixedRows<1>>(),
                        dependentCells.View<Runtime::FixedRows<1>>(),
                        auxiliaryFaceArea.View<Runtime::FixedRows<1>>(), positivityDisplacements.View<Runtime::FixedRows<3>>()};
            }
        };
        ReadAccess Read() const { return ReadAccess(*this); }
    };
}
