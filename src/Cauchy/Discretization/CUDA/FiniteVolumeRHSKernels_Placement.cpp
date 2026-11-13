#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels.hpp"
#include <optional>

namespace DNDS::Cauchy::CUDA
{
    namespace R = Runtime;
    void RHSGeometry::ValidateDevice(R::Device device) const
    {
        R::ValidateViewDevices(device, samples, offsets, faceEntries, faceOffsets, cells, quadrature, conditions);
        if (hasBasis)
            basis.ValidateDevice(device);
        if (hasWall)
            R::ValidateViewDevices(device, wallMean, wallFaces);
    }
    struct RHSGeometryPublication::Data
    {
        RHSGeometry host;
        R::Array<RHSGeometry> storage;
        std::optional<R::Array<RHSGeometry>::ReadLease> read;
        Data(RHSGeometry geometry, R::ExecutionContext context) : host(geometry)
        {
            host.ValidateDevice(context.device);
            DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && host.offsets.RowCount() > 0 &&
                                      host.offsets.RowCount() - 1 <= host.cells.RowCount(),
                                  "invalid CUDA RHS geometry publication");
            R::Array<RHSGeometry> staging(R::RowLayout::Dense(1, 1));
            {
                auto write = staging.Write();
                write.View<R::FixedRows<1>>()(0, 0) = host;
            }
            storage = staging.To(context.device, context);
            storage.SetName("Cauchy/RHSGeometryArguments", DNDS_ALLOCATION_SITE());
            read.emplace(storage.Read(context));
        }
    };
    RHSGeometryPublication::RHSGeometryPublication(RHSGeometry geometry, R::ExecutionContext context)
        : _data(std::make_shared<Data>(geometry, context)) {}
    const RHSGeometry &RHSGeometryPublication::HostDescriptor() const { return _data->host; }
    R::ArrayView<const RHSGeometry, R::AccessBoundsChecks, R::FixedRows<1>> RHSGeometryPublication::View() const
    {
        return _data->read->View<R::FixedRows<1>>();
    }
    R::Device RHSGeometryPublication::Device() const { return _data->storage.GetDevice(); }
    void RHSKernelParameters::Validate(int components) const
    {
        physics.Validate(components);
        DNDS_check_throw_info(thermal.flowFirst == 0 && thermal.components == components && thermal.independent == 0 &&
                                  thermal.speciesEnergies[0].empty() && thermal.speciesEnergies[1].empty(),
                              "invalid perfect-gas RHS policy");
    }
    void RHSKernelControls::Validate() const
    {
        DNDS_check_throw_info(std::isfinite(time) && components >= 5 && components <= 7 && (dimension == 2 || dimension == 3), "invalid CUDA RHS shape/profile");
        DNDS_check_throw_info(solver >= Riemann::Solver::Rusanov && solver <= Riemann::Solver::AUSMPlusUp2, "unknown CUDA Riemann solver");
        DNDS_check_throw_info(direct || firstOrder || (modes > 0 && modes <= 19), "RHS polynomial degree/shape is missing");
        DNDS_check_throw_info(!primitiveGradient || components == 5, "primitive viscous gradients require pure flow5");
        DNDS_check_throw_info(!produceLow || components == 5, "low-order flux blending applies to flow5 only");
    }
    void RHSFields::ValidateDevice(R::Device device) const
    {
        R::ValidateViewDevices(device, mean, reconstruction, viscousData);
        if (hasPreparedMean)
            preparedMean.ValidateDevice(device);
        if (hasIncoming)
            incoming.ValidateDevice(device);
        if (hasWave)
            waveJump.ValidateDevice(device);
        if (hasPressure)
            pressureMinimum.ValidateDevice(device);
    }
}
