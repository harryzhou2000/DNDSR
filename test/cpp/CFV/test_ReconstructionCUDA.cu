#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include "DNDS/Runtime/Communication/NCCLTransfer.hpp"
#include "Geom/Unstructured/UnstructuredMeshSnapshot.hpp"
#include "../Geom/NativeMeshOverlapFixture.hpp"
#include <cuda_runtime.h>
#include <limits>
#include <cstdlib>
#include "RotatedSectorFixture.hpp"
#include "ReconstructionFixture.hpp"

namespace C = DNDS::CFV;
namespace R = DNDS::Runtime;
namespace G = DNDS::Geom;
using I = DNDS::index;
static DNDS::MPIInfo mpi;
static C::ReconstructionDeviceKernel jacobiKernel = C::ReconstructionDeviceKernel::FP64;
static std::shared_ptr<R::NCCLTransferProvider> nccl;
static R::TransferBinding Halo(const C::VariationalReconstructionPlan &plan, C::FVField &field, R::TransferTransport transport = R::TransferTransport::Direct)
{
    R::TransferOptions options;
    options.provider = nccl;
    return plan.BindHalo(field, nccl ? R::TransferTransport::DevicePacked : transport, options);
}

static void Fill(C::FVField &field, double offset)
{
    auto host = field.Owned().To(R::Device::Host());
    {
        auto write = host.Write();
        for (I iRow = 0; iRow < host.RowCount(); ++iRow)
            for (I iComp = 0; iComp < I(field.Width()); ++iComp)
                write.View()(iRow, iComp) = offset + .01 * (iRow + 7 * mpi.rank + iComp);
    }
    field.Owned().CopyFrom(host, field.Owned().GetDevice() == R::Device::Host() ? R::ExecutionContext::Host() : R::ExecutionContext::CUDA(0)).Wait();
}
static void Equal(const C::FVField &a, const C::FVField &b, double tolerance = 5e-11)
{
    auto host = b.Owned().To(R::Device::Host());
    const auto left = a.Owned().Read(), right = host.Read();
    for (I iRow = 0; iRow < host.RowCount(); ++iRow)
        for (I iComp = 0; iComp < I(a.Width()); ++iComp)
            CHECK(right.View()(iRow, iComp) == doctest::Approx(left.View()(iRow, iComp)).epsilon(tolerance).scale(1));
}
__global__ void MetricProbe(C::FiniteVolumeHostView metrics, C::ReconstructionBasisHostView basis,
                            G::UnstructuredMeshView mesh, I iPoint, int side, R::ArrayView<double> result)
{
    if (blockIdx.x || threadIdx.x)
        return;
    result(0, 0) = metrics.GetCellVol(0);
    result(0, 1) = metrics.GetCellCent(0)[0];
    result(0, 2) = basis.FaceMatrix(iPoint, side)[1][1];
    result(0, 3) = mesh.CellNodes(0).size();
    result(0, 4) = mesh.Coord(mesh.CellNodes(0)[0])[0];
}
TEST_CASE("Device publications retain actual mesh metrics and basis with explicit optional capabilities")
{
    auto fixture = G::OverlapTest::BuildNativeFixture(mpi, false);
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(fixture.mesh);
    auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics);
    auto resident = plan.Place(R::Device::CUDA(0), R::ExecutionContext::CUDA(0), {true, true, true});
    auto read = resident.AcquireResidentRead();
    auto mesh = resident.MeshSnapshot().AcquireRead(R::ExecutionContext::CUDA(0));
    auto source = fixture.mesh->AcquireRead(G::MeshReadRequirements::CellFlux());
    const auto view = source.HostView();
    const auto face = view.CellFaces(0)[0];
    const auto parents = view.FaceCells(face);
    const int side = parents[0] == 0 ? 0 : 1;
    auto metricRead = metrics.AcquireRead();
    const auto point = metricRead.HostView().FaceQuadratureBegin(face);
    R::Array<double> result(R::RowLayout::Dense(1, 5), R::Device::CUDA(0));
    {
        auto write = result.Write(R::ExecutionContext::CUDA(0));
        MetricProbe<<<1, 1>>>(read.MetricView(), read.BasisView(), mesh.DeviceView(), point, side, write.View());
        REQUIRE(cudaGetLastError() == cudaSuccess);
        write.Complete(R::Completion::Record(R::ExecutionContext::CUDA(0)));
        write.Wait();
    }
    const auto host = result.To(R::Device::Host());
    const auto values = host.Read();
    CHECK(values.View()(0, 0) == metrics.GetCellVol(0));
    CHECK(values.View()(0, 1) == metrics.GetCellCent(0)[0]);
    CHECK(values.View()(0, 2) == plan.GetCachedFaceBasis(face, side, 0)(1, 1));
    CHECK(values.View()(0, 3) == view.CellNodes(0).size());
    CHECK(values.View()(0, 4) == view.Coord(view.CellNodes(0)[0])[0]);
    CHECK(resident.ResidentBuffers().size() >= 14);
    auto minimal = plan.Place(R::Device::CUDA(0), R::ExecutionContext::CUDA(0));
    CHECK_THROWS(minimal.AcquireResidentRead().MetricView());
    CHECK_THROWS(minimal.MeshSnapshot());
}
TEST_CASE("CUDA Jacobi and WBAP agree with CPU for physical DOF layouts and direct MPI split work")
{
    auto fixture = G::OverlapTest::BuildNativeFixture(mpi, false);
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(fixture.mesh);
    if (mpi.size > 1 && !nccl)
        REQUIRE(DNDS::MPI::isCudaAware());
    for (int degree : {1, 2, 3})
    {
        C::ReconstructionSettings settings;
        settings.maxOrder = degree;
        auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
        auto resident = plan.Place(R::Device::CUDA(0), R::ExecutionContext::CUDA(0));
        const I components = degree == 3 ? 7 : 5, width = components * plan.NRecDOF();
        auto mean = metrics.BuildField(components);
        Fill(mean, 1.0);
        auto reference = plan.BuildWorkspace(components);
        auto referenceCoefficients = plan.BuildCoefficients(components);
        Fill(referenceCoefficients, .02);
        auto bc = plan.BuildBoundaryValues(components);
        {
            auto write = bc.values.Write();
            for (I row = 0; row < bc.values.RowCount(); ++row)
                for (I j = 0; j < components; ++j)
                    write.View()(row, j) = 1 + .1 * j;
        }
        auto deviceBC = resident.PlaceBoundaryValues(bc);
        const auto formats = std::vector<R::RowLayout>{R::RowLayout::Dense(0, width), R::RowLayout::ColMajor(0, width, {32, 1}), R::RowLayout::TiledColMajor(0, width, 8)};
        for (const auto &shape : formats)
        {
            CAPTURE(degree);
            CAPTURE(shape.Kind());
            Fill(referenceCoefficients, .02);
            auto deviceMean = metrics.BuildField(R::RowLayout::ColMajor(0, components, {32, 1}), resident.Device());
            Fill(deviceMean, 1.0);
            auto work = resident.BuildWorkspace(components, shape);
            auto coefficients = resident.BuildCoefficients(components, shape);
            Fill(coefficients, .02);
            auto meanHalo = metrics.BindHalo(mean), referenceHalo = metrics.BindHalo(referenceCoefficients);
            auto deviceMeanHalo = Halo(resident, deviceMean), deviceHalo = Halo(resident, coefficients, shape.Kind() == R::RowLayoutKind::Table ? R::TransferTransport::Direct : R::TransferTransport::DevicePacked);
            meanHalo.Begin().Wait();
            deviceMeanHalo.Begin(R::ExecutionContext::CUDA(0)).Wait();
            R::EagerEngine engine;
            const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
            const auto comm = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
            CHECK(resident.SupportsKernel(C::ReconstructionDeviceKernel::FP64));
            CHECK_FALSE(resident.SupportsKernel(static_cast<C::ReconstructionDeviceKernel>(-1)));
            CHECK_THROWS(resident.DoReconstructionIter(deviceMean, coefficients, work, deviceBC, static_cast<C::ReconstructionDeviceKernel>(-1)));
            for (int step = 0; step < 4; ++step)
            {
                referenceHalo.Begin().Wait();
                plan.DoReconstructionIter(mean, referenceCoefficients, reference, bc);
                R::TransferOperation pending;
                if (nccl)
                {
                    comm.WaitFor(stream.Record());
                    deviceHalo.Enqueue(engine, comm);
                }
                else
                    pending = deviceHalo.Begin(R::ExecutionContext::CUDA(0));
                resident.SubmitReconstructionIterTo(engine, stream, deviceMean, coefficients, work, deviceBC, C::ReconstructionRegion::Core, jacobiKernel);
                if (nccl)
                    stream.WaitFor(comm.Record());
                else
                    pending.Wait();
                resident.SubmitReconstructionIterTo(engine, stream, deviceMean, coefficients, work, deviceBC, C::ReconstructionRegion::HaloDependent, jacobiKernel);
                resident.SubmitCommitReconstruction(engine, stream, coefficients, work);
                engine.Synchronize();
                resident.CheckStatus(work);
                Equal(referenceCoefficients, coefficients);
            }
            auto transform = plan.BuildFaceTransform(0, 5);
            {
                auto forward = transform.forward.Write(), inverse = transform.inverse.Write();
                for (I iFace = 0; iFace < metrics.NumFaceProc(); ++iFace)
                {
                    forward.View()(iFace, 5) = .2;
                    inverse.View()(iFace, 5) = -.2;
                }
            }
            plan.FinalizeFaceTransform(transform);
            auto deviceTransform = resident.PlaceFaceTransform(transform);
            for (int order = degree; order >= 1; --order)
            {
                referenceHalo.Begin().Wait();
                plan.DoLimiterWBAP_3Order(referenceCoefficients, reference, order, std::cref(transform));
                R::TransferOperation pending;
                if (nccl)
                {
                    comm.WaitFor(stream.Record());
                    deviceHalo.Enqueue(engine, comm);
                }
                else
                    pending = deviceHalo.Begin(R::ExecutionContext::CUDA(0));
                resident.SubmitLimiterWBAP_3OrderTo(engine, stream, coefficients, work, order, std::cref(deviceTransform), C::ReconstructionRegion::Core);
                if (nccl)
                    stream.WaitFor(comm.Record());
                else
                    pending.Wait();
                resident.SubmitLimiterWBAP_3OrderTo(engine, stream, coefficients, work, order, std::cref(deviceTransform), C::ReconstructionRegion::HaloDependent);
                resident.SubmitCommitReconstruction(engine, stream, coefficients, work);
                engine.Synchronize();
                resident.CheckStatus(work);
                Equal(referenceCoefficients, coefficients, 2e-9);
            }
        }
    }
}
TEST_CASE("CUDA preserves rotated periodic vectors and axisymmetric prepared operators")
{
    for (bool sector : {true, false})
    {
        auto mesh = sector ? CFVTest::Sector(mpi) : G::OverlapTest::BuildNativeFixture(mpi, false).mesh;
        auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh, 5, !sector);
        C::ReconstructionSettings settings;
        settings.maxOrder = 2;
        auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
        auto resident = plan.Place(R::Device::CUDA(0), R::ExecutionContext::CUDA(0));
        const I components = 21, width = components * plan.NRecDOF();
        const auto vectors = std::vector<I>{1, 7};
        auto mean = metrics.BuildField(components);
        Fill(mean, 1.0);
        auto cpu = plan.BuildWorkspace(components, vectors);
        auto cpuCoefficients = plan.BuildCoefficients(components);
        Fill(cpuCoefficients, .02);
        auto gpu = resident.BuildWorkspace(components, R::RowLayout::TiledColMajor(0, width, 16), vectors);
        auto gpuCoefficients = resident.BuildCoefficients(components, R::RowLayout::TiledColMajor(0, width, 16));
        Fill(gpuCoefficients, .02);
        auto gpuMean = metrics.BuildField(R::RowLayout::ColMajor(0, components, {32, 1}), resident.Device());
        Fill(gpuMean, 1.0);
        auto boundary = plan.BuildBoundaryValues(components);
        {
            auto write = boundary.values.Write();
            for (I iRow = 0; iRow < boundary.values.RowCount(); ++iRow)
                for (I j = 0; j < components; ++j)
                    write.View()(iRow, j) = 1 + .1 * j;
        }
        auto gpuBoundary = resident.PlaceBoundaryValues(boundary);
        auto meanHalo = metrics.BindHalo(mean), gpuMeanHalo = Halo(resident, gpuMean);
        auto cpuHalo = metrics.BindHalo(cpuCoefficients), gpuHalo = Halo(resident, gpuCoefficients);
        meanHalo.Begin().Wait();
        gpuMeanHalo.Begin(R::ExecutionContext::CUDA(0)).Wait();
        for (int iteration = 0; iteration < 3; ++iteration)
        {
            cpuHalo.Begin().Wait();
            gpuHalo.Begin(R::ExecutionContext::CUDA(0)).Wait();
            plan.DoReconstructionIter(mean, cpuCoefficients, cpu, boundary);
            resident.DoReconstructionIter(gpuMean, gpuCoefficients, gpu, gpuBoundary, jacobiKernel);
            Equal(cpuCoefficients, gpuCoefficients, 2e-9);
        }
        for (int order = 2; order >= 1; --order)
        {
            cpuHalo.Begin().Wait();
            gpuHalo.Begin(R::ExecutionContext::CUDA(0)).Wait();
            plan.DoLimiterWBAP_3Order(cpuCoefficients, cpu, order);
            resident.DoLimiterWBAP_3Order(gpuCoefficients, gpu, order);
            Equal(cpuCoefficients, gpuCoefficients, 2e-8);
        }
    }
}
TEST_CASE("CUDA degree three 3D reconstruction supports 37 coupled components and empty peers")
{
    for (I gridWidth : {I(1), I(3)})
    {
        CAPTURE(gridWidth);
        auto mesh = CFVTest::CartesianHexes(mpi, gridWidth);
        auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh);
        C::ReconstructionSettings settings;
        settings.maxOrder = 3;
        auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
        REQUIRE(plan.NRecDOF() == 19);
        auto resident = plan.Place(R::Device::CUDA(0), R::ExecutionContext::CUDA(0));
        const I components = 37, width = components * plan.NRecDOF();
        auto mean = metrics.BuildField(components);
        Fill(mean, 1.0);
        auto cpu = plan.BuildWorkspace(components);
        auto cpuCoefficients = plan.BuildCoefficients(components);
        Fill(cpuCoefficients, .02);
        auto gpu = resident.BuildWorkspace(components, R::RowLayout::ColMajor(0, width, {32, 1}));
        auto gpuCoefficients = resident.BuildCoefficients(components, R::RowLayout::ColMajor(0, width, {32, 1}));
        Fill(gpuCoefficients, .02);
        auto gpuMean = metrics.BuildField(R::RowLayout::TiledColMajor(0, components, 8), resident.Device());
        Fill(gpuMean, 1.0);
        auto boundary = plan.BuildBoundaryValues(components);
        {
            auto write = boundary.values.Write();
            for (I iRow = 0; iRow < boundary.values.RowCount(); ++iRow)
                for (I j = 0; j < components; ++j)
                    write.View()(iRow, j) = 1 + .1 * j;
        }
        auto gpuBoundary = resident.PlaceBoundaryValues(boundary);
        auto meanHalo = metrics.BindHalo(mean), gpuMeanHalo = Halo(resident, gpuMean);
        auto cpuHalo = metrics.BindHalo(cpuCoefficients), gpuHalo = Halo(resident, gpuCoefficients);
        meanHalo.Begin().Wait();
        gpuMeanHalo.Begin(R::ExecutionContext::CUDA(0)).Wait();
        for (int iteration = 0; iteration < 2; ++iteration)
        {
            cpuHalo.Begin().Wait();
            gpuHalo.Begin(R::ExecutionContext::CUDA(0)).Wait();
            plan.DoReconstructionIter(mean, cpuCoefficients, cpu, boundary);
            resident.DoReconstructionIter(gpuMean, gpuCoefficients, gpu, gpuBoundary, jacobiKernel);
            Equal(cpuCoefficients, gpuCoefficients, 2e-9);
        }
        for (int order = 3; order >= 1; --order)
        {
            cpuHalo.Begin().Wait();
            gpuHalo.Begin(R::ExecutionContext::CUDA(0)).Wait();
            plan.DoLimiterWBAP_3Order(cpuCoefficients, cpu, order);
            resident.DoLimiterWBAP_3Order(gpuCoefficients, gpu, order);
            Equal(cpuCoefficients, gpuCoefficients, 2e-8);
        }
    }
}

TEST_CASE("Device errors reject unsupported SOR stale publications and invalid indicators without silent output")
{
    auto fixture = G::OverlapTest::BuildNativeFixture(mpi, false);
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(fixture.mesh);
    C::ReconstructionSettings settings;
    settings.SORInstead = true;
    auto sor = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
    CHECK_THROWS(sor.Place(R::Device::CUDA(0), R::ExecutionContext::CUDA(0)));
    settings.SORInstead = false;
    auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
    auto resident = plan.Place(R::Device::CUDA(0), R::ExecutionContext::CUDA(0));
    auto work = resident.BuildWorkspace(5, R::RowLayout::TiledColMajor(0, 5 * plan.NRecDOF(), 8));
    auto coefficients = resident.BuildCoefficients(5, R::RowLayout::TiledColMajor(0, 5 * plan.NRecDOF(), 8));
    Fill(coefficients, 1.0);
    Fill(work.temporary, 9.0);
    // A real mixed active/inactive pass, followed by recovery/error checks.
    auto cpu = plan.BuildWorkspace(5);
    auto cpuCoefficients = plan.BuildCoefficients(5);
    Fill(cpuCoefficients, 1.0);
    R::Array<double> activation(R::RowLayout::Dense(metrics.NumCell(), 1));
    {
        auto write = activation.Write();
        for (I iRow = 0; iRow < metrics.NumCell(); ++iRow)
            write.View()(iRow, 0) = iRow % 2;
    }
    auto gpuActivation = activation.To(R::Device::CUDA(0));
    metrics.BindHalo(cpuCoefficients).Begin().Wait();
    Halo(resident, coefficients).Begin(R::ExecutionContext::CUDA(0)).Wait();
    plan.DoLimiterWBAP_3OrderTo(cpuCoefficients, cpu, 1, std::nullopt, C::ReconstructionRegion::All, &activation, .5);
    plan.CommitReconstruction(cpuCoefficients, cpu);
    R::EagerEngine activeEngine;
    auto activeStream = activeEngine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    resident.SubmitLimiterWBAP_3OrderTo(activeEngine, activeStream, coefficients, work, 1, std::nullopt, C::ReconstructionRegion::All, &gpuActivation, .5);
    resident.SubmitCommitReconstruction(activeEngine, activeStream, coefficients, work);
    activeEngine.Synchronize();
    resident.CheckStatus(work);
    Equal(cpuCoefficients, coefficients, 2e-9);
    auto bad = work;
    bad.temporary = coefficients;
    CHECK_THROWS(resident.SubmitCommitReconstruction(activeEngine, activeStream, coefficients, bad));
    bad = work;
    bad.components = std::numeric_limits<I>::max();
    CHECK_THROWS(resident.SubmitCommitReconstruction(activeEngine, activeStream, coefficients, bad));
    bad = work;
    bad.scratch = R::Array<double>(R::RowLayout::Padded(std::vector<size_t>(size_t(work.deviceBlocks * 6), 0), work.scratch.Layout().Width()), R::Device::CUDA(0));
    CHECK_THROWS(resident.SubmitCommitReconstruction(activeEngine, activeStream, coefficients, bad));
    Fill(coefficients, 1.0);
    Fill(work.temporary, 9.0);
    auto before = work.temporary.Owned().To(R::Device::Host());
    R::Array<double> indicator(R::RowLayout::Dense(metrics.NumCell(), 1));
    indicator.Write().At(0, 0) = std::numeric_limits<double>::quiet_NaN();
    auto deviceIndicator = indicator.To(R::Device::CUDA(0));
    R::EagerEngine engine;
    auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    resident.SubmitLimiterWBAP_3OrderTo(engine, stream, coefficients, work, 1, std::nullopt, C::ReconstructionRegion::All, &deviceIndicator);
    resident.SubmitCommitReconstruction(engine, stream, coefficients, work);
    engine.Synchronize();
    CHECK_THROWS(resident.CheckStatus(work));
    const auto actual = work.temporary.Owned().To(R::Device::Host());
    const auto a = actual.Read(), b = before.Read();
    for (I row = 0; row < actual.RowCount(); ++row)
        for (I j = 0; j < actual.RowSize(row); ++j)
            CHECK(a.View()(row, j) == b.View()(row, j));
    {
        auto mutation = fixture.mesh->versions.BeginMutation(R::Change::Geometry);
        mutation.Commit();
    }
    CHECK_THROWS(resident.Validate());
}
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    mpi = DNDS::MPIInfo(MPI_COMM_WORLD);
    if (const auto *value = std::getenv("DNDS_TEST_RECONSTRUCTION_KERNEL"))
    {
        DNDS_check_throw_info(std::string(value) == "tensor" || std::string(value) == "shared" || std::string(value) == "async", "unknown test reconstruction kernel");
        jacobiKernel = std::string(value) == "shared" ? C::ReconstructionDeviceKernel::FP64TensorCoreShared : std::string(value) == "async" ? C::ReconstructionDeviceKernel::FP64TensorCoreAsync
                                                                                                                                            : C::ReconstructionDeviceKernel::FP64TensorCore;
    }
    if (const auto *value = std::getenv("DNDS_TEST_TRANSFER_PROVIDER"))
    {
        DNDS_check_throw_info(std::string(value) == "nccl", "unknown test transfer provider");
        nccl = R::NCCLTransferProvider::Create(mpi.comm, R::Device::CUDA(0));
    }
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    const auto result = context.run();
    nccl.reset();
    MPI_Finalize();
    return result;
}
