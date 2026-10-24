#define DOCTEST_CONFIG_IMPLEMENT
#define DNDS_MESH_FACADE_TEST_EMBEDDED
#include "test_MeshFacade.cpp"
#include "Geom/Mesh/NativeMeshSnapshot.hpp"
#include <cuda_runtime.h>
#include <cstdlib>
#include <atomic>
#include <thread>

static int FacadeGPUOrdinal()
{
    const auto mpi = FacadeMPI();
    int devices = 0;
    REQUIRE(cudaGetDeviceCount(&devices) == cudaSuccess);
    REQUIRE(devices > 0);
    const char *visible = std::getenv("CUDA_VISIBLE_DEVICES");
    // The external one-rank-per-GPU wrapper verifies unique physical UUIDs.
    // Without visibility isolation, every rank must select its own GPU.
    const bool isolated = devices == 1 && visible && *visible && std::string(visible).find(',') == std::string::npos;
    REQUIRE((isolated || devices >= mpi.size));
    const int ordinal = isolated ? 0 : mpi.rank;
    REQUIRE(cudaSetDevice(ordinal) == cudaSuccess);
    return ordinal;
}

struct FacadeStreamGate
{
    std::atomic<bool> release{false};
    cudaStream_t stream;
    explicit FacadeStreamGate(cudaStream_t value) : stream(value) {}
    static void CUDART_CB Wait(void *opaque)
    {
        auto &gate = *static_cast<FacadeStreamGate *>(opaque);
        while (!gate.release.load(std::memory_order_acquire))
            std::this_thread::yield();
    }
    ~FacadeStreamGate()
    {
        release.store(true, std::memory_order_release);
        // Exception-safe lifetime: the callback must stop touching this object
        // before its atomic storage is destroyed. No CUDA calls in the callback.
        cudaStreamSynchronize(stream);
    }
};

TEST_CASE("Mesh facade CUDA kernels traverse retained snapshots with host parity")
{
    const int ordinal = FacadeGPUOrdinal();
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    const auto context = R::ExecutionContext::CUDA(ordinal, stream);
    G::Test::MeshViewProbeResult *output = nullptr;
    REQUIRE(cudaMalloc(reinterpret_cast<void **>(&output), sizeof(*output)) == cudaSuccess);
    G::Test::MeshViewProbeResult *actual = nullptr;
    REQUIRE(cudaMallocHost(reinterpret_cast<void **>(&actual), sizeof(*actual)) == cudaSuccess);
    for (int fixture = 0; fixture < 3; ++fixture)
    {
        const bool periodic = fixture == 1, empty = fixture == 2;
        auto mesh = FacadeFixture(periodic, empty);
        G::Test::MeshViewProbeResult expected;
        {
            auto host = mesh->AcquireRead(AllFacadeRequirements());
            expected = G::Test::MeshViewProbe(host.HostView());
        }
        auto versions = mesh->versions;
        std::optional<G::MeshReadAccess> access;
        access.emplace(mesh->AcquireRead(AllFacadeRequirements(), context));
        FacadeStreamGate gate(stream);
        CHECK_THROWS(access->HostView());
        CHECK_FALSE(versions.CanMutate());
        CHECK_THROWS(mesh->versions.BeginMutation(R::Change::Geometry));
        // CUDA lazy module loading can synchronize on the first kernel launch.
        // Load the probe before a host callback deliberately blocks this stream.
        REQUIRE(G::Test::LaunchMeshViewProbe(access->DeviceView(), output, stream) == cudaSuccess);
        REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
        REQUIRE(cudaLaunchHostFunc(stream, FacadeStreamGate::Wait, &gate) == cudaSuccess);
        REQUIRE(G::Test::LaunchMeshViewProbe(access->DeviceView(), output, stream) == cudaSuccess);
        REQUIRE(cudaMemcpyAsync(actual, output, sizeof(*actual), cudaMemcpyDeviceToHost, stream) == cudaSuccess);
        access->Complete(R::Completion::Record(context));
        CHECK_THROWS(access->DeviceView());
        // Drop every legacy allocation owner before stream completion. The
        // scope retains CUDA values/row metadata and the mesh version lease.
        CHECK(cudaStreamQuery(stream) == cudaErrorNotReady);
        mesh.reset();
        CHECK_FALSE(versions.CanMutate());
        gate.release.store(true, std::memory_order_release);
        access->Wait();
        CHECK_THROWS(access->Complete(R::Completion::Record(context)));
        CHECK_THROWS(access->DeviceView());
        CheckProbeEqual(*actual, expected);
        CHECK_FALSE(versions.CanMutate());
        access.reset();
        CHECK(versions.CanMutate());
    }
    REQUIRE(cudaFreeHost(actual) == cudaSuccess);
    REQUIRE(cudaFree(output) == cudaSuccess);
    REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
}

TEST_CASE("Moved mesh CUDA access can outlive its destroyed acquisition stream")
{
    const int ordinal = FacadeGPUOrdinal();
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    auto mesh = FacadeFixture();
    std::optional<G::MeshReadAccess> movedFrom;
    movedFrom.emplace(mesh->AcquireRead(G::MeshReadRequirements::CellFlux(), R::ExecutionContext::CUDA(ordinal, stream)));
    {
        G::MeshReadAccess destination(std::move(*movedFrom));
        CHECK_THROWS(movedFrom->DeviceView());
        destination.Complete(R::Completion::Record(R::ExecutionContext::CUDA(ordinal, stream)));
        CHECK_THROWS(destination.DeviceView());
        destination.Wait();
        CHECK_THROWS(destination.DeviceView());
    }
    CHECK(mesh->versions.CanMutate());
    REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
    movedFrom.reset(); // Must not record an event on the destroyed stream.
    CHECK(cudaGetLastError() == cudaSuccess);
}

TEST_CASE("Mesh facade CUDA completion validates the acquisition stream")
{
    const int ordinal = FacadeGPUOrdinal();
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    auto mesh = FacadeFixture(true);
    {
        auto access = mesh->AcquireRead(AllFacadeRequirements(), R::ExecutionContext::CUDA(ordinal, stream));
        CHECK_THROWS(access.Complete(R::Completion::Record(R::ExecutionContext::CUDA(ordinal))));
        access.Complete(R::Completion::Record(R::ExecutionContext::CUDA(ordinal, stream)));
        access.Wait();
    }
    CHECK(mesh->versions.CanMutate());
    REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
}

TEST_CASE("Native mesh CUDA queries match existing const legacy pair views")
{
    const int ordinal = FacadeGPUOrdinal();
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    int *deviceErrors = nullptr;
    REQUIRE(cudaMalloc(reinterpret_cast<void **>(&deviceErrors), sizeof(int)) == cudaSuccess);
    for (bool periodic : {false, true})
    {
        auto native = FacadeFixture(periodic), legacy = FacadeFixture(periodic);
        if (!periodic)
        {
            auto zeros = [](const auto &rows)
            {
                std::vector<std::vector<G::NodePeriodicBits>> result;
                for (I i = 0; i < rows.Size(); ++i)
                    result.emplace_back(size_t(rows.RowSize(i)), G::NodePeriodicBits{});
                return result;
            };
            PublishRows(legacy->cell2nodePbi, "legacy.reference.cellBits", legacy->mpi, zeros(legacy->cell2node));
            PublishRows(legacy->face2nodePbi, "legacy.reference.faceBits", legacy->mpi, zeros(legacy->face2node));
            PublishRows(legacy->cell2facePbi, "legacy.reference.slotBits", legacy->mpi, zeros(legacy->cell2face));
        }
        // Independent legacy allocations use their established device copies.
        // Reference acquisition is const; no legacy implementation is altered.
        legacy->coords.to_device(DNDS::DeviceBackend::CUDA);
        legacy->cell2node.to_device(DNDS::DeviceBackend::CUDA);
        legacy->face2node.to_device(DNDS::DeviceBackend::CUDA);
        legacy->cell2face.to_device(DNDS::DeviceBackend::CUDA);
        legacy->face2cell.to_device(DNDS::DeviceBackend::CUDA);
        legacy->cellElemInfo.to_device(DNDS::DeviceBackend::CUDA);
        legacy->faceElemInfo.to_device(DNDS::DeviceBackend::CUDA);
        legacy->cell2nodePbi.to_device(DNDS::DeviceBackend::CUDA);
        legacy->face2nodePbi.to_device(DNDS::DeviceBackend::CUDA);
        legacy->cell2facePbi.to_device(DNDS::DeviceBackend::CUDA);
        const auto &reference = std::as_const(*legacy);
        auto raggedReference = [](const auto &pair)
        {
            using Base = DNDS::ParArray<I, DNDS::NonUniformSize>;
            using View = DNDS::ArrayPair<Base>::t_deviceViewConst<DNDS::DeviceBackend::CUDA>;
            return View{static_cast<const Base &>(*pair.father).template deviceView<DNDS::DeviceBackend::CUDA>(),
                        static_cast<const Base &>(*pair.son).template deviceView<DNDS::DeviceBackend::CUDA>()};
        };
        using FixedBase = DNDS::ParArray<I, 2>;
        using FixedView = DNDS::ArrayPair<FixedBase>::t_deviceViewConst<DNDS::DeviceBackend::CUDA>;
        G::Test::LegacyConstMeshReference old{
            reference.coords.template deviceView<DNDS::DeviceBackend::CUDA>(),
            raggedReference(reference.cell2node),
            raggedReference(reference.face2node),
            raggedReference(reference.cell2face),
            FixedView{static_cast<const FixedBase &>(*reference.face2cell.father).template deviceView<DNDS::DeviceBackend::CUDA>(),
                      static_cast<const FixedBase &>(*reference.face2cell.son).template deviceView<DNDS::DeviceBackend::CUDA>()},
            reference.cellElemInfo.template deviceView<DNDS::DeviceBackend::CUDA>(),
            reference.faceElemInfo.template deviceView<DNDS::DeviceBackend::CUDA>(),
            reference.cell2nodePbi.template deviceView<DNDS::DeviceBackend::CUDA>(),
            reference.face2nodePbi.template deviceView<DNDS::DeviceBackend::CUDA>(),
            reference.cell2facePbi.template deviceView<DNDS::DeviceBackend::CUDA>(),
            reference.periodicInfo, periodic};
        auto access = native->AcquireRead(G::MeshReadRequirements::CellFlux(), R::ExecutionContext::CUDA(ordinal, stream));
        REQUIRE(G::Test::LaunchLegacyMeshParity(access.DeviceView(), old, deviceErrors, stream) == cudaSuccess);
        access.Complete(R::Completion::Record(R::ExecutionContext::CUDA(ordinal, stream)));
        access.Wait();
        int errors = -1;
        REQUIRE(cudaMemcpy(&errors, deviceErrors, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
        CHECK(errors == 0);
    }
    REQUIRE(cudaFree(deviceErrors) == cudaSuccess);
    REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
}

TEST_CASE("Detached native CUDA mesh backing is placed once and reused by reads")
{
    const int ordinal = FacadeGPUOrdinal();
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    const auto context = R::ExecutionContext::CUDA(ordinal, stream);
    G::Test::MeshViewProbeResult *output = nullptr;
    REQUIRE(cudaMalloc(reinterpret_cast<void **>(&output), sizeof(*output)) == cudaSuccess);
    auto mesh = FacadeFixture(true);
    auto host = G::NativeMeshSnapshot::Capture(*mesh, AllFacadeRequirements());
    const auto expected = G::Test::MeshViewProbe(host.AcquireRead().HostView());
    std::optional<G::NativeMeshSnapshot> placed;
    placed.emplace(host.To(R::Device::CUDA(ordinal), context));
    mesh.reset();
    const DNDS::real *firstCoordinateAllocation = nullptr;
    for (int repeat = 0; repeat < 3; ++repeat)
    {
        auto access = placed->AcquireRead(context);
        auto view = access.DeviceView();
        // Coordinates have a fixed table layout: this obtains only its device
        // pointer; no device values or CSR metadata are read by the host.
        const auto allocation = view.Coord(0).data();
        if (!repeat)
            firstCoordinateAllocation = allocation;
        CHECK(allocation == firstCoordinateAllocation);
        if (repeat == 2)
            placed.reset(); // Scope alone retains persistent backing.
        REQUIRE(G::Test::LaunchMeshViewProbe(view, output, stream) == cudaSuccess);
        access.Complete(R::Completion::Record(context));
        access.Wait();
        G::Test::MeshViewProbeResult actual;
        REQUIRE(cudaMemcpy(&actual, output, sizeof(actual), cudaMemcpyDeviceToHost) == cudaSuccess);
        CheckProbeEqual(actual, expected);
    }
    REQUIRE(cudaFree(output) == cudaSuccess);
    REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const int result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
