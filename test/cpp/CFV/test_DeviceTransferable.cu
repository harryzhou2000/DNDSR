/**
 * @file test_DeviceTransferable.cu
 * @brief Unit tests for DeviceTransferable CRTP mixin on FiniteVolume.
 *
 * Tests CUDA and Host device transfers using the DeviceTransferable
 * mixin that FiniteVolume inherits. Verifies that data survives the
 * host->device->host round-trip by reading metric values through
 * FiniteVolumeDeviceView inside a CUDA kernel and comparing against
 * host-side values.
 *
 * This is a standalone MPI test (not doctest) following the existing
 * CUDA test pattern in app/DNDS/.
 *
 * NOTE: Do NOT "using namespace DNDS;" in .cu files -- DNDS::index
 * conflicts with POSIX index() from <strings.h> under nvcc.
 */

#include "CFV/FiniteVolume.hpp"
#include "Geom/Mesh/Mesh.hpp"
#include "Geom/Mesh/NativeMeshCGNS.hpp"
#include "Geom/Mesh/NativeMeshTopology.hpp"
#include "Geom/Mesh/Mesh_Helpers.hpp"
#include "../Geom/LegacyMeshFixtureCodec.hpp"
#include "DNDS/Device/CUDA_Utils.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace // anonymous -- avoid POSIX index() ambiguity
{
    using DNDS::DeviceBackend;
    using DNDS::MPIInfo;
    using DNDS::ssp;

    // ===================================================================
    // Mesh path helper (same convention as test_Reconstruction.cpp)
    // ===================================================================
    std::string meshPath(const std::string &name)
    {
        std::string f(__FILE__);
        for (int i = 0; i < 4; i++)
        {
            auto pos = f.rfind('/');
            if (pos == std::string::npos)
                pos = f.rfind('\\');
            if (pos != std::string::npos)
                f = f.substr(0, pos);
        }
        return f + "/data/mesh/" + name;
    }

    // ===================================================================
    // Build a simple mesh + FiniteVolume with constructed metrics
    // ===================================================================
    std::pair<ssp<DNDS::Geom::UnstructuredMesh>, ssp<DNDS::CFV::FiniteVolume>>
    buildMeshAndFV(MPIInfo &mpi, bool native)
    {
        constexpr int dim = 2;
        DNDS::Geom::PartitionOptions partition;
        partition.metisSeed = 42;
        auto mesh = DNDS::Geom::ReadNativeMeshFromCGNS(mpi, dim,
                                                       meshPath("Uniform_3x3_wall.cgns"), partition, DNDS::Geom::FBC_Name_2_ID_Default);
        if (!native)
        {
            // The reference traverses the historical primary payload codec
            // into independent Runtime ownership, then rebuilds native faces.
            auto reference = DNDS::Geom::TestFixtureCodec::DecodePrimaryReference(*mesh);
            {
                auto sourceRead = mesh->AcquireRead(DNDS::Geom::MeshReadRequirements::CellFlux());
                auto referenceRead = reference->AcquireRead(DNDS::Geom::MeshReadRequirements::CellFlux());
                const auto source = sourceRead.HostView(), decoded = referenceRead.HostView();
                DNDS_assert(source.NumCell() == decoded.NumCell() && source.NumNode() == decoded.NumNode());
                if (source.NumNode())
                    DNDS_assert(source.Coord(0).data() != decoded.Coord(0).data());
                for (DNDS::index cell = 0; cell < source.NumCell(); ++cell)
                {
                    DNDS_assert(mesh->OriginalEntityId(DNDS::Geom::EntityKind::Cell, cell) == reference->OriginalEntityId(DNDS::Geom::EntityKind::Cell, cell));
                    DNDS::Geom::MeshQueryCoords expected, actual;
                    source.GetCoordsOnCell(cell, expected);
                    decoded.GetCoordsOnCell(cell, actual);
                    DNDS_assert(expected.isApprox(actual, 1e-14));
                    DNDS_assert(source.GetCellInfo(cell).type == decoded.GetCellInfo(cell).type);
                }
            }
            mesh = std::move(reference);
        }
        DNDS_assert_info(mesh->HasNativeState(), "device fixture requires native mesh authority");

        auto fv = std::make_shared<DNDS::CFV::FiniteVolume>(mpi, mesh);

        DNDS::CFV::FiniteVolumeSettings defaultSettings(dim);
        nlohmann::ordered_json j;
        defaultSettings.WriteIntoJson(j);
        j["maxOrder"] = 1;
        j["intOrder"] = 5;
        fv->parseSettings(j);

        fv->SetCellAtrBasic();
        fv->ConstructCellVolume();
        fv->ConstructCellBary();
        fv->ConstructCellCent();
        fv->ConstructCellIntJacobiDet();
        fv->ConstructCellIntPPhysics();
        fv->ConstructCellAlignedHBox();
        fv->ConstructCellMajorHBoxCoordInertia();
        fv->SetFaceAtrBasic();
        fv->ConstructFaceArea();
        fv->ConstructFaceCent();
        fv->ConstructFaceIntJacobiDet();
        fv->ConstructFaceIntPPhysics();
        fv->ConstructFaceUnitNorm();
        fv->ConstructFaceMeanNorm();
        fv->ConstructCellSmoothScale();
        fv->FinalizeMetrics();

        return {mesh, fv};
    }

} // anonymous namespace

// ===================================================================
// CUDA kernel: read cell volumes and barycenters from device view,
// write into output buffers for host-side verification.
// ===================================================================
DNDS_GLOBAL void kernel_read_fv_metrics(
    DNDS::CFV::FiniteVolume::t_deviceView<DeviceBackend::CUDA> *fv_view,
    DNDS::real *out_volumes,
    DNDS::real *out_bary_x,
    DNDS::real *out_bary_y,
    DNDS::real *out_face_areas,
    DNDS::index numCells,
    DNDS::index numFaces)
{
    DNDS::index tid = (DNDS::index)blockIdx.x * (DNDS::index)blockDim.x + (DNDS::index)threadIdx.x;

    if (tid < numCells)
    {
        out_volumes[tid] = fv_view->GetCellVol(tid);
        auto bary = fv_view->GetCellBary(tid);
        out_bary_x[tid] = bary(0);
        out_bary_y[tid] = bary(1);
    }

    if (tid < numFaces)
    {
        out_face_areas[tid] = fv_view->GetFaceArea(tid);
    }
}

// This fixture contains Quad4 cells. Exercise the embedded mesh view itself,
// including random node access, rather than only the independent FV arrays.
DNDS_GLOBAL void kernel_read_fv_mesh(
    DNDS::CFV::FiniteVolume::t_deviceView<DeviceBackend::CUDA> *fv,
    DNDS::index *nodes, DNDS::index *faces, DNDS::index *neighbors,
    DNDS::real *coordinates, DNDS::index cells, DNDS::index ownedCells)
{
    const DNDS::index cell = DNDS::index(blockIdx.x) * blockDim.x + threadIdx.x;
    if (cell >= cells)
        return;
    const auto &mesh = fv->mesh;
    for (DNDS::rowsize slot = 0; slot < 4; ++slot)
    {
        const auto node = mesh.CellNodes(cell)[slot];
        nodes[cell * 4 + slot] = node;
        // Outer ghost cell faces may be unavailable. Geometry cones are
        // complete for every proc row; face-side traversal is owned-only.
        if (cell < ownedCells)
        {
            const auto face = mesh.CellFaces(cell)[slot];
            faces[cell * 4 + slot] = face;
            neighbors[cell * 4 + slot] = mesh.CellFaceOther(cell, face, slot);
        }
        const auto point = mesh.Coord(node);
        for (int d = 0; d < 3; ++d)
            coordinates[(cell * 4 + slot) * 3 + d] = point[d];
    }
}

namespace // anonymous for test functions
{
    // ===================================================================
    // Test: Device state tracking via DeviceTransferable mixin
    // ===================================================================
    void test_device_state_tracking(DNDS::CFV::FiniteVolume &fv, MPIInfo &mpi)
    {
        if (mpi.rank == 0)
            std::cout << "  test_device_state_tracking... " << std::flush;

        auto B0 = fv.device();
        DNDS_assert_info(B0 == DeviceBackend::Unknown,
                         "Expected Unknown before any transfer");

        DNDS::index bytes0 = fv.getDeviceArrayBytes();
        DNDS_assert_info(bytes0 > 0, "Expected positive array bytes");

        fv.to_device(DeviceBackend::CUDA);
        auto B1 = fv.device();
        DNDS_assert_info(B1 == DeviceBackend::CUDA,
                         "Expected CUDA after to_device(CUDA)");

        DNDS::index bytes1 = fv.getDeviceArrayBytes();
        DNDS_assert_info(bytes1 > 0, "Expected positive array bytes on CUDA");

        fv.to_host();
        auto B2 = fv.device();
        DNDS_assert_info(B2 == DeviceBackend::Unknown,
                         "Expected Unknown after to_host()");

        if (mpi.rank == 0)
            std::cout << "PASSED (bytes=" << bytes0 << ")" << std::endl;
    }

    // ===================================================================
    // Test: Host device transfer preserves data access
    // ===================================================================
    void test_host_device_transfer(DNDS::CFV::FiniteVolume &fv, MPIInfo &mpi)
    {
        if (mpi.rank == 0)
            std::cout << "  test_host_device_transfer... " << std::flush;

        DNDS::index nCells = fv.mesh->NumCellProc();
        std::vector<DNDS::real> vol_before(nCells);
        for (DNDS::index i = 0; i < nCells; i++)
            vol_before[i] = fv.GetCellVol(i);

        fv.to_device(DeviceBackend::Host);
        auto B = fv.device();
        DNDS_assert_info(B == DeviceBackend::Host,
                         "Expected Host after to_device(Host)");

        for (DNDS::index i = 0; i < nCells; i++)
        {
            DNDS_assert_info(fv.GetCellVol(i) == vol_before[i],
                             "Cell volume changed after Host transfer");
        }

        fv.to_host();

        for (DNDS::index i = 0; i < nCells; i++)
        {
            DNDS_assert_info(fv.GetCellVol(i) == vol_before[i],
                             "Cell volume changed after to_host()");
        }

        if (mpi.rank == 0)
            std::cout << "PASSED" << std::endl;
    }

    // ===================================================================
    // Test: CUDA round-trip -- transfer FV to device, read in kernel,
    // transfer back, verify values match.
    // ===================================================================
    void test_cuda_round_trip(DNDS::CFV::FiniteVolume &fv, MPIInfo &mpi)
    {
        if (mpi.rank == 0)
            std::cout << "  test_cuda_round_trip... " << std::flush;

        DNDS::index nCells = fv.mesh->NumCellProc();
        DNDS::index nFaces = fv.mesh->NumFaceProc();
        const DNDS::index ownedCells = fv.mesh->NumCell();

        // 1) Record host-side metric values
        std::vector<DNDS::real> host_volumes(nCells);
        std::vector<DNDS::real> host_bary_x(nCells);
        std::vector<DNDS::real> host_bary_y(nCells);
        std::vector<DNDS::real> host_face_areas(nFaces);
        std::vector<DNDS::index> host_nodes(nCells * 4), host_faces(ownedCells * 4), host_neighbors(ownedCells * 4);
        std::vector<DNDS::real> host_coordinates(nCells * 12);
        {
            auto read = fv.mesh->AcquireRead(DNDS::Geom::MeshReadRequirements::CellFlux());
            const auto view = read.HostView();
            for (DNDS::index cell = 0; cell < nCells; ++cell)
            {
                DNDS_assert(view.CellNodes(cell).size() == 4);
                if (cell < ownedCells)
                    DNDS_assert(view.CellFaces(cell).size() == 4);
                for (DNDS::rowsize slot = 0; slot < 4; ++slot)
                {
                    const auto node = view.CellNodes(cell)[slot];
                    host_nodes[cell * 4 + slot] = node;
                    if (cell < ownedCells)
                    {
                        const auto face = view.CellFaces(cell)[slot];
                        DNDS_assert_info(face >= 0 && face < view.NumFaceProc(), "owned cell face must be available");
                        host_faces[cell * 4 + slot] = face;
                        host_neighbors[cell * 4 + slot] = view.CellFaceOther(cell, face, slot);
                    }
                    for (int d = 0; d < 3; ++d)
                        host_coordinates[(cell * 4 + slot) * 3 + d] = view.Coord(node)[d];
                }
            }
        }

        for (DNDS::index i = 0; i < nCells; i++)
        {
            host_volumes[i] = fv.GetCellVol(i);
            auto bary = fv.GetCellBary(i);
            host_bary_x[i] = bary(0);
            host_bary_y[i] = bary(1);
        }
        for (DNDS::index i = 0; i < nFaces; i++)
            host_face_areas[i] = fv.GetFaceArea(i);

        // 2) Transfer metric arrays. AcquireDeviceRead retains the mesh placement.
        if (fv.mesh->HasNativeState())
        {
            int ordinal = -1;
            DNDS_CUDA_CHECKED(cudaGetDevice(&ordinal));
            fv.mesh->PlaceNative(DNDS::Runtime::Device::CUDA(ordinal),
                                 DNDS::Runtime::ExecutionContext::CUDA(ordinal));
        }
        fv.to_device(DeviceBackend::CUDA);

        DNDS_assert_info(fv.device() == DeviceBackend::CUDA,
                         "Expected CUDA device after to_device");

        // 3) Build device view and copy to device memory
        auto fv_read = fv.AcquireDeviceRead<DeviceBackend::CUDA>();
        auto fv_view_host = fv_read.View();
        DNDS::CUDA::DeviceObject<DNDS::CFV::FiniteVolume::t_deviceView<DeviceBackend::CUDA>> fv_dev(fv_view_host);

        // 4) Allocate device output buffers
        DNDS::index maxN = std::max(nCells, nFaces);
        thrust::device_vector<DNDS::real> d_volumes(nCells);
        thrust::device_vector<DNDS::real> d_bary_x(nCells);
        thrust::device_vector<DNDS::real> d_bary_y(nCells);
        thrust::device_vector<DNDS::real> d_face_areas(nFaces);
        thrust::device_vector<DNDS::index> d_nodes(nCells * 4), d_faces(ownedCells * 4), d_neighbors(ownedCells * 4);
        thrust::device_vector<DNDS::real> d_coordinates(nCells * 12);

        // 5) Launch kernel
        uint32_t threadsPerBlock = 256;
        if (maxN > 0)
        {
            auto [blocksPerGrid, unused] = DNDS::CUDA::calckernelSizeSimple(maxN, threadsPerBlock);
            kernel_read_fv_metrics<<<blocksPerGrid, threadsPerBlock>>>(
                fv_dev.get(),
                thrust::raw_pointer_cast(d_volumes.data()),
                thrust::raw_pointer_cast(d_bary_x.data()),
                thrust::raw_pointer_cast(d_bary_y.data()),
                thrust::raw_pointer_cast(d_face_areas.data()),
                nCells, nFaces);
            kernel_read_fv_mesh<<<blocksPerGrid, threadsPerBlock>>>(
                fv_dev.get(), thrust::raw_pointer_cast(d_nodes.data()),
                thrust::raw_pointer_cast(d_faces.data()), thrust::raw_pointer_cast(d_neighbors.data()),
                thrust::raw_pointer_cast(d_coordinates.data()), nCells, ownedCells);
        }
        DNDS_CUDA_CHECKED(cudaDeviceSynchronize());

        // 6) Copy kernel results back to host
        thrust::host_vector<DNDS::real> h_volumes = d_volumes;
        thrust::host_vector<DNDS::real> h_bary_x = d_bary_x;
        thrust::host_vector<DNDS::real> h_bary_y = d_bary_y;
        thrust::host_vector<DNDS::real> h_face_areas = d_face_areas;
        thrust::host_vector<DNDS::index> h_nodes = d_nodes, h_faces = d_faces, h_neighbors = d_neighbors;
        thrust::host_vector<DNDS::real> h_coordinates = d_coordinates;
        for (size_t i = 0; i < host_nodes.size(); ++i)
        {
            DNDS_assert(h_nodes[i] == host_nodes[i]);
        }
        for (size_t i = 0; i < host_faces.size(); ++i)
        {
            DNDS_assert(h_faces[i] == host_faces[i]);
            DNDS_assert(h_neighbors[i] == host_neighbors[i]);
        }
        for (size_t i = 0; i < host_coordinates.size(); ++i)
            DNDS_assert(std::abs(h_coordinates[i] - host_coordinates[i]) < 1e-14);

        // 7) Verify: values read on GPU must match host-side values
        for (DNDS::index i = 0; i < nCells; i++)
        {
            DNDS_assert_infof(
                std::abs(h_volumes[i] - host_volumes[i]) < 1e-14,
                "Cell %lld volume mismatch: GPU=%.16e host=%.16e",
                (long long)i, h_volumes[i], host_volumes[i]);
            DNDS_assert_infof(
                std::abs(h_bary_x[i] - host_bary_x[i]) < 1e-14,
                "Cell %lld bary_x mismatch: GPU=%.16e host=%.16e",
                (long long)i, h_bary_x[i], host_bary_x[i]);
            DNDS_assert_infof(
                std::abs(h_bary_y[i] - host_bary_y[i]) < 1e-14,
                "Cell %lld bary_y mismatch: GPU=%.16e host=%.16e",
                (long long)i, h_bary_y[i], host_bary_y[i]);
        }
        for (DNDS::index i = 0; i < nFaces; i++)
        {
            DNDS_assert_infof(
                std::abs(h_face_areas[i] - host_face_areas[i]) < 1e-14,
                "Face %lld area mismatch: GPU=%.16e host=%.16e",
                (long long)i, h_face_areas[i], host_face_areas[i]);
        }

        // 8) Transfer back to host and verify again
        fv_read.Wait();
        fv.to_host();

        DNDS_assert_info(fv.device() == DeviceBackend::Unknown,
                         "Expected Unknown after to_host");

        for (DNDS::index i = 0; i < nCells; i++)
        {
            DNDS_assert_infof(
                fv.GetCellVol(i) == host_volumes[i],
                "Cell %lld volume changed after round-trip: now=%.16e was=%.16e",
                (long long)i, fv.GetCellVol(i), host_volumes[i]);
        }

        if (mpi.rank == 0)
            std::cout << "PASSED (cells=" << nCells << ", faces=" << nFaces << ")" << std::endl;
    }

    // ===================================================================
    // Test: FiniteVolumeDeviceView is trivially copyable and host view
    // returns correct values without any device transfer.
    // ===================================================================
    void test_device_view_trivially_copyable(DNDS::CFV::FiniteVolume &fv, MPIInfo &mpi)
    {
        if (mpi.rank == 0)
            std::cout << "  test_device_view_trivially_copyable... " << std::flush;

        static_assert(
            std::is_trivially_copyable_v<DNDS::CFV::FiniteVolume::t_deviceView<DeviceBackend::CUDA>>,
            "FiniteVolumeDeviceView<CUDA> must be trivially copyable");
        static_assert(
            std::is_trivially_copyable_v<DNDS::CFV::FiniteVolume::t_deviceView<DeviceBackend::Host>>,
            "FiniteVolumeDeviceView<Host> must be trivially copyable");

        auto host_read = fv.AcquireDeviceRead<DeviceBackend::Host>();
        auto host_view = host_read.View();
        DNDS::index nCells = fv.mesh->NumCellProc();
        for (DNDS::index i = 0; i < nCells; i++)
        {
            DNDS_assert_info(
                host_view.GetCellVol(i) == fv.GetCellVol(i),
                "Host deviceView value mismatch");
        }

        if (mpi.rank == 0)
            std::cout << "PASSED" << std::endl;
    }

    // ===================================================================
    // Test: Multiple CUDA round-trips (stability check)
    // ===================================================================
    void test_cuda_multiple_round_trips(DNDS::CFV::FiniteVolume &fv, MPIInfo &mpi)
    {
        if (mpi.rank == 0)
            std::cout << "  test_cuda_multiple_round_trips... " << std::flush;

        DNDS::index nCells = fv.mesh->NumCellProc();
        std::vector<DNDS::real> expected_vols(nCells);
        for (DNDS::index i = 0; i < nCells; i++)
            expected_vols[i] = fv.GetCellVol(i);

        for (int trip = 0; trip < 3; trip++)
        {
            fv.to_device(DeviceBackend::CUDA);
            DNDS_assert_info(fv.device() == DeviceBackend::CUDA, "Expected CUDA");

            fv.to_host();
            DNDS_assert_info(fv.device() == DeviceBackend::Unknown, "Expected Unknown");

            for (DNDS::index i = 0; i < nCells; i++)
            {
                DNDS_assert_infof(
                    fv.GetCellVol(i) == expected_vols[i],
                    "Trip %d cell %lld volume drift: %.16e vs %.16e",
                    trip, (long long)i, fv.GetCellVol(i), expected_vols[i]);
            }
        }

        if (mpi.rank == 0)
            std::cout << "PASSED (3 round-trips)" << std::endl;
    }

} // anonymous namespace

// ===================================================================
// main
// ===================================================================
int main(int argc, char *argv[])
{
    DNDS::MPI::Init_thread(&argc, &argv);
    DNDS::MPIInfo mpi;
    mpi.setWorld();

    if (mpi.rank == 0)
        std::cout << "=== DeviceTransferable unit tests (CUDA + Host) ===" << std::endl;

    for (bool native : {false, true})
    {
        auto [mesh, fv] = buildMeshAndFV(mpi, native);
        DNDS_assert_info(mesh->HasNativeState(), "device fixture must retain native authority");
        if (mpi.rank == 0)
            std::cout << "Mesh built (path=" << (native ? "direct native" : "primary wire reference") << "): " << mesh->NumCell()
                      << " cells, " << mesh->NumFace() << " faces" << std::endl;
        test_device_view_trivially_copyable(*fv, mpi);
        test_device_state_tracking(*fv, mpi);
        test_host_device_transfer(*fv, mpi);
        test_cuda_round_trip(*fv, mpi);
        test_cuda_multiple_round_trips(*fv, mpi);
    }

    if (mpi.rank == 0)
        std::cout << "=== All DeviceTransferable tests PASSED ===" << std::endl;

    DNDS::MPI::Finalize();
    return 0;
}
