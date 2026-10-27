# CMakeLists.txt for C++ unit tests (doctest)
#
# Usage:
#   cmake -DDNDS_BUILD_TESTS=ON ..
#   cmake --build . -t all_unit_tests -j8
#   ctest --test-dir . --output-on-failure
#
# For aggregated doctest summary (test cases + assertions):
#   python scripts/ctest_summary.py [ctest args...]
#
# Aggregate targets:
#   dnds_unit_tests  — only DNDS/ tests
#   geom_unit_tests  — only Geom/ tests
#   all_unit_tests   — everything
#
# MPI tests are registered for each np in DNDS_TEST_NP_LIST.
# Set via environment variable: DNDS_TEST_NP_LIST="1;2;4" cmake ..
# Default: 1;2;4;8
#
# OpenMP threads can be configured via DNDS_TEST_OMP_THREADS (default: 2).

# --- Read NP list from env, default to 1;2;4;8 ---
if(DEFINED ENV{DNDS_TEST_NP_LIST})
    set(DNDS_TEST_NP_LIST "$ENV{DNDS_TEST_NP_LIST}")
else()
    set(DNDS_TEST_NP_LIST "1;2;4;8")
endif()
message(STATUS "DNDS_TEST_NP_LIST: ${DNDS_TEST_NP_LIST}")

# --- Read OMP threads from env, default to 2 ---
if(DEFINED ENV{DNDS_TEST_OMP_THREADS})
    set(DNDS_TEST_OMP_THREADS "$ENV{DNDS_TEST_OMP_THREADS}")
else()
    set(DNDS_TEST_OMP_THREADS "2")
endif()
message(STATUS "DNDS_TEST_OMP_THREADS: ${DNDS_TEST_OMP_THREADS}")

# --- Detect OpenMPI and add --oversubscribe if needed ---
# OpenMPI requires --oversubscribe when np > physical cores (common in CI/testing).
# MPICH/Intel MPI do not need (or recognize) this flag.
#
# Also support DNDS_TEST_MPIEXTRA: space-separated flags appended to every
# add_test MPI command (e.g. "--oversubscribe").  On GitHub runners set
# OMPI_MCA_rmaps_base_oversubscribe=1 as a belt-and-suspenders fallback.
if(NOT "$ENV{DNDS_TEST_MPIEXTRA}" STREQUAL "")
    separate_arguments(_DNDS_MPIEXTRA UNIX_COMMAND "$ENV{DNDS_TEST_MPIEXTRA}")
    list(APPEND MPIEXEC_PREFLAGS ${_DNDS_MPIEXTRA})
    message(STATUS "DNDS_TEST_MPIEXTRA: appending ${_DNDS_MPIEXTRA} to MPIEXEC_PREFLAGS")
elseif(MPIEXEC_EXECUTABLE)
    execute_process(
        COMMAND ${MPIEXEC_EXECUTABLE} --version
        OUTPUT_VARIABLE _MPIEXEC_VERSION_OUTPUT
        ERROR_VARIABLE _MPIEXEC_VERSION_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_MPIEXEC_VERSION_OUTPUT MATCHES "Open MPI")
        message(STATUS "Detected OpenMPI: adding --oversubscribe to MPIEXEC_PREFLAGS for tests")
        list(APPEND MPIEXEC_PREFLAGS "--oversubscribe")
    endif()
endif()

# --- Timeout per np ---
function(_dnds_np_timeout NP OUT_VAR)
    if(NP LESS_EQUAL 2)
        set(${OUT_VAR} ${DNDS_TEST_TIMEOUT} PARENT_SCOPE)
    elseif(NP LESS_EQUAL 4)
        math(EXPR _t "${DNDS_TEST_TIMEOUT} * 3 / 2")
        set(${OUT_VAR} ${_t} PARENT_SCOPE)
    else()
        math(EXPR _t "${DNDS_TEST_TIMEOUT} * 2")
        set(${OUT_VAR} ${_t} PARENT_SCOPE)
    endif()
endfunction()

find_path(DNDS_EXTERNAL_INCLUDE_DOCTEST doctest.h PATHS
    "${PROJECT_SOURCE_DIR}/external/doctest"
    NO_DEFAULT_PATH REQUIRED)

# --- helper: add a serial unit test executable (DNDS) ---
function(dnds_add_unit_test TEST_NAME SRC_FILE LIBS)
    set(TARGET_NAME "dnds_test_${TEST_NAME}")
    add_executable(${TARGET_NAME} ${SRC_FILE})
    target_include_directories(${TARGET_NAME} PRIVATE
        ${DNDS_EXTERNAL_INCLUDE_DOCTEST}
        ${DNDS_EXTERNAL_INCLUDES}
        ${DNDS_INCLUDES})
    target_link_libraries(${TARGET_NAME} PRIVATE ${LIBS} ${DNDS_EXTERNAL_LIBS})
    if(DNDS_USE_CCACHE)
        set_property(TARGET ${TARGET_NAME} PROPERTY CXX_COMPILER_LAUNCHER ${DNDS_CCACHE_EXEC})
    endif()
    set_target_properties(${TARGET_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        EXCLUDE_FROM_ALL ON)
    add_test(NAME dnds_${TEST_NAME} COMMAND ${TARGET_NAME})
    _dnds_maybe_set_timeout(dnds_${TEST_NAME} ${DNDS_TEST_TIMEOUT})
    set_tests_properties(dnds_${TEST_NAME} PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=${DNDS_TEST_OMP_THREADS}")
endfunction()

# --- helper: add an MPI unit test executable (DNDS) ---
# MPI tests run all TEST_CASEs together per NP value (avoids mpirun overhead per test case).
function(dnds_add_mpi_test TEST_NAME SRC_FILE LIBS)
    set(TARGET_NAME "dnds_test_${TEST_NAME}")
    add_executable(${TARGET_NAME} ${SRC_FILE})
    target_include_directories(${TARGET_NAME} PRIVATE
        ${DNDS_EXTERNAL_INCLUDE_DOCTEST}
        ${DNDS_EXTERNAL_INCLUDES}
        ${DNDS_INCLUDES})
    target_link_libraries(${TARGET_NAME} PRIVATE ${LIBS} ${DNDS_EXTERNAL_LIBS})
    if(DNDS_USE_CCACHE)
        set_property(TARGET ${TARGET_NAME} PROPERTY CXX_COMPILER_LAUNCHER ${DNDS_CCACHE_EXEC})
    endif()
    set_target_properties(${TARGET_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        EXCLUDE_FROM_ALL ON)
    foreach(NP ${DNDS_TEST_NP_LIST})
        _dnds_np_timeout(${NP} _TIMEOUT)
        add_test(NAME dnds_${TEST_NAME}_np${NP}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${NP}
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${TARGET_NAME}> ${MPIEXEC_POSTFLAGS})
        _dnds_maybe_set_timeout(dnds_${TEST_NAME}_np${NP} ${_TIMEOUT})
        set_tests_properties(dnds_${TEST_NAME}_np${NP} PROPERTIES
            PROCESSORS ${NP}
            ENVIRONMENT "OMP_NUM_THREADS=${DNDS_TEST_OMP_THREADS}")
    endforeach()
endfunction()

# --- helper: add a serial unit test executable (Geom) ---
function(geom_add_unit_test TEST_NAME SRC_FILE LIBS)
    set(TARGET_NAME "geom_test_${TEST_NAME}")
    add_executable(${TARGET_NAME} ${SRC_FILE})
    target_include_directories(${TARGET_NAME} PRIVATE
        ${DNDS_EXTERNAL_INCLUDE_DOCTEST}
        ${DNDS_EXTERNAL_INCLUDES}
        ${DNDS_INCLUDES})
    target_link_libraries(${TARGET_NAME} PRIVATE ${LIBS} ${DNDS_EXTERNAL_LIBS})
    if(DNDS_USE_CCACHE)
        set_property(TARGET ${TARGET_NAME} PROPERTY CXX_COMPILER_LAUNCHER ${DNDS_CCACHE_EXEC})
    endif()
    set_target_properties(${TARGET_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        EXCLUDE_FROM_ALL ON)
    add_test(NAME geom_${TEST_NAME} COMMAND ${TARGET_NAME})
    _dnds_maybe_set_timeout(geom_${TEST_NAME} ${DNDS_TEST_TIMEOUT})
    set_tests_properties(geom_${TEST_NAME} PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=${DNDS_TEST_OMP_THREADS}")
endfunction()

# --- helper: add an MPI unit test executable (Geom) ---
# MPI tests run all TEST_CASEs together per NP value (avoids mpirun overhead per test case).
function(geom_add_mpi_test TEST_NAME SRC_FILE LIBS)
    set(TARGET_NAME "geom_test_${TEST_NAME}")
    add_executable(${TARGET_NAME} ${SRC_FILE})
    target_include_directories(${TARGET_NAME} PRIVATE
        ${DNDS_EXTERNAL_INCLUDE_DOCTEST}
        ${DNDS_EXTERNAL_INCLUDES}
        ${DNDS_INCLUDES})
    target_link_libraries(${TARGET_NAME} PRIVATE ${LIBS} ${DNDS_EXTERNAL_LIBS})
    if(DNDS_USE_CCACHE)
        set_property(TARGET ${TARGET_NAME} PROPERTY CXX_COMPILER_LAUNCHER ${DNDS_CCACHE_EXEC})
    endif()
    set_target_properties(${TARGET_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        EXCLUDE_FROM_ALL ON)
    foreach(NP ${DNDS_TEST_NP_LIST})
        _dnds_np_timeout(${NP} _TIMEOUT)
        add_test(NAME geom_${TEST_NAME}_np${NP}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${NP}
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${TARGET_NAME}> ${MPIEXEC_POSTFLAGS})
        _dnds_maybe_set_timeout(geom_${TEST_NAME}_np${NP} ${_TIMEOUT})
        set_tests_properties(geom_${TEST_NAME}_np${NP} PROPERTIES
            PROCESSORS ${NP}
            ENVIRONMENT "OMP_NUM_THREADS=${DNDS_TEST_OMP_THREADS}")
    endforeach()
endfunction()

# ===================================================================
# Aggregate targets
# ===================================================================
add_custom_target(dnds_unit_tests)   # DNDS/ tests only
add_custom_target(geom_unit_tests)   # Geom/ tests only
add_custom_target(all_unit_tests)    # everything
add_dependencies(all_unit_tests dnds_unit_tests geom_unit_tests)

# ===================================================================
# DNDS core tests
# ===================================================================
dnds_add_unit_test(array "DNDS/test_Array.cpp" "dnds")
dnds_add_unit_test(runtime "DNDS/test_Runtime.cpp" "dnds")
dnds_add_unit_test(runtime_array "DNDS/test_RuntimeArray.cpp" "dnds")
dnds_add_unit_test(runtime_element_schema "DNDS/test_RuntimeElementSchema.cpp" "dnds")
add_dependencies(dnds_unit_tests dnds_test_runtime_element_schema)
dnds_add_unit_test(legacy_array_bridge "DNDS/test_LegacyArrayBridge.cpp" "dnds")
add_dependencies(dnds_unit_tests dnds_test_legacy_array_bridge)
dnds_add_unit_test(runtime_row_builder "DNDS/test_RuntimeRowBuilder.cpp" "dnds")
dnds_add_unit_test(runtime_engine "DNDS/test_EagerEngine.cpp" "dnds")
add_dependencies(dnds_unit_tests dnds_test_runtime_engine)
add_dependencies(dnds_unit_tests dnds_test_runtime_row_builder)
add_dependencies(dnds_unit_tests dnds_test_runtime_array)
# Separate extension exercises sharing across pybind modules, not test-only
# methods embedded in the production Runtime binding. Build/install the normal
# Python modules first, then set PYTHONPATH to include this target's directory.
# Do not apply pybind11's host LTO/strip extras to the CUDA device-link object.
pybind11_add_module(runtime_array_bridge_test MODULE EXCLUDE_FROM_ALL NO_EXTRAS DNDS/RuntimeArrayBridge_bind.cpp)
target_include_directories(runtime_array_bridge_test PRIVATE ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
target_link_libraries(runtime_array_bridge_test PRIVATE dnds_shared ${DNDS_EXTERNAL_LIBS})
if(DNDS_USE_CUDA)
    target_sources(runtime_array_bridge_test PRIVATE DNDS/RuntimeArrayBridge_CUDA.cu ../demos/MockStencil.cu)
endif()
set_target_properties(runtime_array_bridge_test PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
dnds_add_unit_test(config "DNDS/test_Config.cpp" "dnds")
dnds_add_unit_test(hard_eigen "DNDS/test_HardEigen.cpp" "dnds")
dnds_add_mpi_test(object_contracts "DNDS/test_ObjectContracts.cpp" "dnds")
target_sources(dnds_test_object_contracts PRIVATE DNDS/test_LocalChecks.cpp)

dnds_add_mpi_test(serializer "DNDS/test_Serializer.cpp" "dnds")
dnds_add_mpi_test(mpi "DNDS/test_MPI.cpp" "dnds")
dnds_add_mpi_test(runtime_exchange "DNDS/test_RuntimeExchange.cpp" "dnds")
dnds_add_mpi_test(runtime_row_transfer "DNDS/test_RuntimeRowTransfer.cpp" "dnds")
dnds_add_mpi_test(runtime_transient "DNDS/test_RuntimeTransient.cpp" "dnds")
add_dependencies(dnds_unit_tests dnds_test_runtime_transient)
dnds_add_mpi_test(runtime_array_io "DNDS/test_RuntimeArrayIO.cpp" "dnds")
dnds_add_mpi_test(runtime_native_io "DNDS/test_RuntimeNativeIO.cpp" "dnds")
add_dependencies(dnds_unit_tests dnds_test_runtime_native_io)
add_dependencies(dnds_unit_tests dnds_test_runtime_row_transfer dnds_test_runtime_array_io)
dnds_add_mpi_test(array_transformer "DNDS/test_ArrayTransformer.cpp" "dnds")
dnds_add_mpi_test(array_derived "DNDS/test_ArrayDerived.cpp" "dnds")
dnds_add_mpi_test(array_dof "DNDS/test_ArrayDOF.cpp" "dnds")
dnds_add_mpi_test(index_mapping "DNDS/test_IndexMapping.cpp" "dnds")
dnds_add_mpi_test(permutation_transfer "DNDS/test_PermutationTransfer.cpp" "dnds")

if(DNDS_USE_CUDA)
    dnds_add_unit_test(runtime_cuda "DNDS/test_Runtime_CUDA.cu" "dnds")
    dnds_add_unit_test(runtime_cuda_engine "DNDS/test_CUDAExecution.cu" "dnds")
    set_tests_properties(dnds_runtime_cuda_engine PROPERTIES TIMEOUT 60)
    add_dependencies(dnds_unit_tests dnds_test_runtime_cuda_engine)
    add_dependencies(dnds_unit_tests dnds_test_runtime_cuda)
    dnds_add_mpi_test(array_dof_cuda "DNDS/test_ArrayDOF_CUDA.cu" "dnds")
    add_dependencies(dnds_unit_tests dnds_test_array_dof_cuda)
endif()

set(DNDS_TEST_TARGETS
    dnds_test_runtime
    dnds_test_runtime_exchange
    dnds_test_array
    dnds_test_config
    dnds_test_object_contracts
    dnds_test_serializer
    dnds_test_mpi
    dnds_test_array_transformer
    dnds_test_array_derived
    dnds_test_array_dof
    dnds_test_index_mapping
    dnds_test_permutation_transfer
)
foreach(T ${DNDS_TEST_TARGETS})
    add_dependencies(dnds_unit_tests ${T})
endforeach()

# ===================================================================
# Geom tests
# ===================================================================
geom_add_unit_test(elements "Geom/test_Elements.cpp" "geom;dnds")
geom_add_unit_test(quadrature "Geom/test_Quadrature.cpp" "geom;dnds")
target_sources(geom_test_quadrature PRIVATE Geom/test_QuadratureCachePeer.cpp Geom/test_LocalPerformance.cpp)
geom_add_mpi_test(mesh_index_conversion "Geom/test_MeshIndexConversion.cpp" "geom;dnds")
geom_add_mpi_test(mesh_pipeline "Geom/test_MeshPipeline.cpp" "geom;dnds")
geom_add_mpi_test(mesh_distributed_read "Geom/test_MeshDistributedRead.cpp" "geom;dnds")
geom_add_mpi_test(mesh_connectivity "Geom/test_MeshConnectivity.cpp" "geom;dnds")
geom_add_mpi_test(mesh_connectivity_ghost "Geom/test_MeshConnectivity_Ghost.cpp" "geom;dnds")
geom_add_mpi_test(mesh_connectivity_interpolate "Geom/test_MeshConnectivity_Interpolate.cpp" "geom;dnds")
geom_add_mpi_test(mesh_reorder "Geom/test_MeshReorder.cpp" "geom;dnds")
geom_add_unit_test(mesh_cgns_multizone "Geom/test_MeshCGNSMultiZone.cpp" "geom;dnds")
geom_add_mpi_test(mesh_audit "Geom/test_MeshAudit.cpp" "geom;dnds")
geom_add_mpi_test(mesh_facade "Geom/test_MeshFacade.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_mesh_facade)
geom_add_mpi_test(mesh_facade_pipeline "Geom/test_MeshFacadePipeline.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_mesh_facade_pipeline)
geom_add_mpi_test(mesh_facade_overlap "Geom/test_MeshFacadeOverlap.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_mesh_facade_overlap)
geom_add_mpi_test(relation_access "Geom/test_RelationAccess.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_relation_access)
geom_add_mpi_test(native_relation_conversion "Geom/test_NativeRelationConversion.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_relation_conversion)
geom_add_mpi_test(native_connectivity "Geom/test_NativeConnectivity.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_connectivity)
geom_add_mpi_test(native_mesh_scenarios "Geom/test_NativeMeshScenarios.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_scenarios)
geom_add_mpi_test(native_mesh_snapshot "Geom/test_NativeMeshSnapshot.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_snapshot)
geom_add_mpi_test(native_mesh_publication "Geom/test_NativeMeshPublication.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_publication)
geom_add_mpi_test(native_mesh_construction "Geom/test_NativeMeshConstruction.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_construction)
geom_add_mpi_test(native_mesh_setup "Geom/test_NativeMeshSetup.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_setup)
geom_add_mpi_test(native_mesh_builder "Geom/test_NativeMeshBuilder.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_builder)
geom_add_mpi_test(native_relation_registry "Geom/test_NativeRelationRegistry.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_relation_registry)

geom_add_mpi_test(native_mesh_transition "Geom/test_NativeMeshTransition.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_transition)
geom_add_mpi_test(native_transition_rebuild "Geom/test_NativeTransitionRebuild.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_transition_rebuild)
geom_add_mpi_test(native_mesh_archive "Geom/test_NativeMeshArchive.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_archive)
geom_add_mpi_test(native_mesh_cgns "Geom/test_NativeMeshCGNS.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_cgns)
geom_add_mpi_test(native_mesh_topology "Geom/test_NativeMeshTopology.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_topology)
geom_add_mpi_test(native_mesh_openfoam "Geom/test_NativeMeshOpenFOAM.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_openfoam)
geom_add_mpi_test(native_mesh_legacy_archive "Geom/test_NativeMeshLegacyArchive.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_legacy_archive)
geom_add_mpi_test(native_mesh_preparation "Geom/test_NativeMeshPreparation.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_preparation)
geom_add_mpi_test(native_mesh_partition_graph "Geom/test_NativeMeshPartitionGraph.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_partition_graph)
geom_add_mpi_test(native_mesh_wall_dist "Geom/test_NativeMeshWallDist.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_wall_dist)
geom_add_mpi_test(native_mesh_caches "Geom/test_NativeMeshCaches.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_caches)
geom_add_mpi_test(native_mesh_local_reorder "Geom/test_NativeMeshLocalReorder.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_local_reorder)
geom_add_mpi_test(native_mesh_serial_output "Geom/test_NativeMeshSerialOutput.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_serial_output)
geom_add_mpi_test(native_mesh_refinement "Geom/test_NativeMeshRefinement.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_refinement)
geom_add_mpi_test(native_mesh_extraction "Geom/test_NativeMeshExtraction.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_extraction)
geom_add_mpi_test(native_mesh_smoothing "Geom/test_NativeMeshSmoothing.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_smoothing)
geom_add_mpi_test(native_mesh_periodic "Geom/test_NativeMeshPeriodic.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_periodic)
geom_add_mpi_test(native_mesh_periodic_import "Geom/test_NativeMeshPeriodicImport.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_periodic_import)
geom_add_mpi_test(mesh_edit "Geom/test_MeshEdit.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_mesh_edit)
geom_add_mpi_test(mesh_transition "Geom/test_MeshTransition.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_mesh_transition)
geom_add_mpi_test(mesh_mutation "Geom/test_MeshMutation.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_mesh_mutation)
geom_add_mpi_test(mesh_geometry_edit "Geom/test_MeshGeometryEdit.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_mesh_geometry_edit)
geom_add_mpi_test(mesh_current_output "Geom/test_MeshCurrentOutput.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_mesh_current_output)
geom_add_mpi_test(mesh_field_checkpoint "Geom/test_MeshFieldCheckpoint.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_mesh_field_checkpoint)
geom_add_mpi_test(mesh_entity_field "Geom/test_MeshEntityField.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_mesh_entity_field)
if(DNDS_USE_CUDA)
    geom_add_unit_test(mesh_facade_cuda "Geom/test_MeshFacade_CUDA.cu" "geom;dnds")
    target_sources(geom_test_mesh_facade_cuda PRIVATE Geom/MeshViewKernel.cu)
    add_dependencies(geom_unit_tests geom_test_mesh_facade_cuda)
    add_executable(geom_test_mesh_facade_overlap_cuda EXCLUDE_FROM_ALL Geom/test_MeshFacadeOverlap_CUDA.cu)
    target_include_directories(geom_test_mesh_facade_overlap_cuda PRIVATE
        ${DNDS_EXTERNAL_INCLUDE_DOCTEST} ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
    target_link_libraries(geom_test_mesh_facade_overlap_cuda PRIVATE geom dnds ${DNDS_EXTERNAL_LIBS})
    target_sources(geom_test_mesh_facade_overlap_cuda PRIVATE Geom/MeshFacadeOverlapKernel.cu)
    add_test(NAME geom_mesh_facade_overlap_cuda_np2
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
            ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/test/demos/one_rank_per_gpu.py
            $<TARGET_FILE:geom_test_mesh_facade_overlap_cuda> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(geom_mesh_facade_overlap_cuda_np2 PROPERTIES
        PROCESSORS 2 TIMEOUT 180 ENVIRONMENT "OMP_NUM_THREADS=${DNDS_TEST_OMP_THREADS}")
    add_dependencies(geom_unit_tests geom_test_mesh_facade_overlap_cuda)
    add_executable(geom_test_mesh_entity_field_cuda EXCLUDE_FROM_ALL Geom/test_MeshEntityField_CUDA.cu)
    target_include_directories(geom_test_mesh_entity_field_cuda PRIVATE
        ${DNDS_EXTERNAL_INCLUDE_DOCTEST} ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
    target_link_libraries(geom_test_mesh_entity_field_cuda PRIVATE geom dnds ${DNDS_EXTERNAL_LIBS})
    add_test(NAME geom_mesh_entity_field_cuda_np2
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
            ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/test/demos/one_rank_per_gpu.py
            $<TARGET_FILE:geom_test_mesh_entity_field_cuda> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(geom_mesh_entity_field_cuda_np2 PROPERTIES
        PROCESSORS 2 TIMEOUT 180 ENVIRONMENT "OMP_NUM_THREADS=${DNDS_TEST_OMP_THREADS}")
    add_dependencies(geom_unit_tests geom_test_mesh_entity_field_cuda)
endif()

set(GEOM_TEST_TARGETS
    geom_test_elements
    geom_test_quadrature
    geom_test_mesh_index_conversion
    geom_test_mesh_pipeline
    geom_test_mesh_distributed_read
    geom_test_mesh_connectivity
    geom_test_mesh_connectivity_ghost
    geom_test_mesh_connectivity_interpolate
    geom_test_mesh_reorder
    geom_test_mesh_cgns_multizone
    geom_test_mesh_audit
)
foreach(T ${GEOM_TEST_TARGETS})
    add_dependencies(geom_unit_tests ${T})
endforeach()

# ===================================================================
# CFV tests
# ===================================================================

# --- helper: add a serial unit test executable (CFV) ---
function(cfv_add_unit_test TEST_NAME SRC_FILE LIBS)
    set(TARGET_NAME "cfv_test_${TEST_NAME}")
    add_executable(${TARGET_NAME} ${SRC_FILE})
    target_include_directories(${TARGET_NAME} PRIVATE
        ${DNDS_EXTERNAL_INCLUDE_DOCTEST}
        ${DNDS_EXTERNAL_INCLUDES}
        ${DNDS_INCLUDES})
    target_link_libraries(${TARGET_NAME} PRIVATE ${LIBS} ${DNDS_EXTERNAL_LIBS})
    if(DNDS_USE_CCACHE)
        set_property(TARGET ${TARGET_NAME} PROPERTY CXX_COMPILER_LAUNCHER ${DNDS_CCACHE_EXEC})
    endif()
    set_target_properties(${TARGET_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        EXCLUDE_FROM_ALL ON)
    add_test(NAME cfv_${TEST_NAME} COMMAND ${TARGET_NAME})
    _dnds_maybe_set_timeout(cfv_${TEST_NAME} ${DNDS_TEST_TIMEOUT})
    set_tests_properties(cfv_${TEST_NAME} PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=${DNDS_TEST_OMP_THREADS}")
endfunction()

# --- helper: add an MPI unit test executable (CFV) ---
# MPI tests run all TEST_CASEs together per NP value (avoids mpirun overhead per test case).
function(cfv_add_mpi_test TEST_NAME SRC_FILE LIBS)
    set(TARGET_NAME "cfv_test_${TEST_NAME}")
    add_executable(${TARGET_NAME} ${SRC_FILE})
    target_include_directories(${TARGET_NAME} PRIVATE
        ${DNDS_EXTERNAL_INCLUDE_DOCTEST}
        ${DNDS_EXTERNAL_INCLUDES}
        ${DNDS_INCLUDES})
    target_link_libraries(${TARGET_NAME} PRIVATE ${LIBS} ${DNDS_EXTERNAL_LIBS})
    if(DNDS_USE_CCACHE)
        set_property(TARGET ${TARGET_NAME} PROPERTY CXX_COMPILER_LAUNCHER ${DNDS_CCACHE_EXEC})
    endif()
    set_target_properties(${TARGET_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        EXCLUDE_FROM_ALL ON)
    foreach(NP ${DNDS_TEST_NP_LIST})
        _dnds_np_timeout(${NP} _TIMEOUT)
        add_test(NAME cfv_${TEST_NAME}_np${NP}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${NP}
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${TARGET_NAME}> ${MPIEXEC_POSTFLAGS})
        _dnds_maybe_set_timeout(cfv_${TEST_NAME}_np${NP} ${_TIMEOUT})
        set_tests_properties(cfv_${TEST_NAME}_np${NP} PROPERTIES
            PROCESSORS ${NP}
            ENVIRONMENT "OMP_NUM_THREADS=${DNDS_TEST_OMP_THREADS}")
    endforeach()
endfunction()

add_custom_target(cfv_unit_tests)
add_dependencies(all_unit_tests cfv_unit_tests)

cfv_add_mpi_test(reconstruction "CFV/test_Reconstruction.cpp" "cfv;geom;dnds")
cfv_add_unit_test(limiters "CFV/test_Limiters.cpp" "cfv;geom;dnds")
cfv_add_mpi_test(reconstruction3d "CFV/test_Reconstruction3D.cpp" "cfv;geom;dnds")

set(CFV_TEST_TARGETS
    cfv_test_reconstruction
    cfv_test_limiters
    cfv_test_reconstruction3d
)

# --- CUDA CFV tests ---
if(DNDS_USE_CUDA)
    # helper: add a CUDA MPI test executable for CFV
    function(cfv_add_cuda_mpi_test TEST_NAME SRC_FILE LIBS)
        set(TARGET_NAME "cfv_test_${TEST_NAME}")
        add_executable(${TARGET_NAME} ${SRC_FILE})
        target_include_directories(${TARGET_NAME} PRIVATE
            ${DNDS_EXTERNAL_INCLUDE_DOCTEST}
            ${DNDS_EXTERNAL_INCLUDES}
            ${DNDS_INCLUDES})
        target_link_libraries(${TARGET_NAME} PRIVATE ${LIBS} ${DNDS_EXTERNAL_LIBS})
        if(DNDS_USE_CCACHE)
            set_property(TARGET ${TARGET_NAME} PROPERTY CXX_COMPILER_LAUNCHER ${DNDS_CCACHE_EXEC})
        endif()
        set_target_properties(${TARGET_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            EXCLUDE_FROM_ALL ON)
        # np=1 only for CUDA tests (single GPU)
        add_test(NAME cfv_${TEST_NAME}_np1
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${TARGET_NAME}> ${MPIEXEC_POSTFLAGS})
        _dnds_maybe_set_timeout(cfv_${TEST_NAME}_np1 ${DNDS_TEST_TIMEOUT})
        set_tests_properties(cfv_${TEST_NAME}_np1 PROPERTIES
            PROCESSORS 1
            ENVIRONMENT "OMP_NUM_THREADS=${DNDS_TEST_OMP_THREADS}")
    endfunction()

    cfv_add_cuda_mpi_test(device_transferable "CFV/test_DeviceTransferable.cu" "cfv;geom;dnds")
    list(APPEND CFV_TEST_TARGETS cfv_test_device_transferable)
endif()

foreach(T ${CFV_TEST_TARGETS})
    add_dependencies(cfv_unit_tests ${T})
endforeach()

# ===================================================================
# Euler tests
# ===================================================================

# --- helper: add a serial unit test executable (Euler) ---
# Gas.hpp functions are header-only; link cfv;geom;dnds for types.
function(euler_add_unit_test TEST_NAME SRC_FILE LIBS)
    set(TARGET_NAME "euler_test_${TEST_NAME}")
    add_executable(${TARGET_NAME} ${SRC_FILE})
    target_include_directories(${TARGET_NAME} PRIVATE
        ${DNDS_EXTERNAL_INCLUDE_DOCTEST}
        ${DNDS_EXTERNAL_INCLUDES}
        ${DNDS_INCLUDES})
    target_link_libraries(${TARGET_NAME} PRIVATE ${LIBS} ${DNDS_EXTERNAL_LIBS})
    if(DNDS_USE_CCACHE)
        set_property(TARGET ${TARGET_NAME} PROPERTY CXX_COMPILER_LAUNCHER ${DNDS_CCACHE_EXEC})
    endif()
    set_target_properties(${TARGET_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        EXCLUDE_FROM_ALL ON)
    add_test(NAME euler_${TEST_NAME} COMMAND ${TARGET_NAME})
    _dnds_maybe_set_timeout(euler_${TEST_NAME} ${DNDS_TEST_TIMEOUT})
    set_tests_properties(euler_${TEST_NAME} PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=${DNDS_TEST_OMP_THREADS};DNDS_MECH_PATH=${DNDS_CANTERA_DATA_DIR};CANTERA_DATA=${DNDS_CANTERA_DATA_DIR}")
endfunction()

# --- helper: add an MPI unit test executable (Euler) ---
# MPI tests run all TEST_CASEs together per NP value (avoids mpirun overhead per test case).
function(euler_add_mpi_test TEST_NAME SRC_FILE LIBS)
    set(TARGET_NAME "euler_test_${TEST_NAME}")
    add_executable(${TARGET_NAME} ${SRC_FILE})
    target_include_directories(${TARGET_NAME} PRIVATE
        ${DNDS_EXTERNAL_INCLUDE_DOCTEST}
        ${DNDS_EXTERNAL_INCLUDES}
        ${DNDS_INCLUDES})
    target_link_libraries(${TARGET_NAME} PRIVATE ${LIBS} ${DNDS_EXTERNAL_LIBS})
    if(DNDS_USE_CCACHE)
        set_property(TARGET ${TARGET_NAME} PROPERTY CXX_COMPILER_LAUNCHER ${DNDS_CCACHE_EXEC})
    endif()
    set_target_properties(${TARGET_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        EXCLUDE_FROM_ALL ON)
    foreach(NP ${DNDS_TEST_NP_LIST})
        _dnds_np_timeout(${NP} _TIMEOUT)
        add_test(NAME euler_${TEST_NAME}_np${NP}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${NP}
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${TARGET_NAME}> ${MPIEXEC_POSTFLAGS})
        _dnds_maybe_set_timeout(euler_${TEST_NAME}_np${NP} ${_TIMEOUT})
        set_tests_properties(euler_${TEST_NAME}_np${NP} PROPERTIES
            PROCESSORS ${NP}
            ENVIRONMENT "OMP_NUM_THREADS=${DNDS_TEST_OMP_THREADS};DNDS_MECH_PATH=${DNDS_CANTERA_DATA_DIR};CANTERA_DATA=${DNDS_CANTERA_DATA_DIR}")
    endforeach()
endfunction()

add_custom_target(euler_unit_tests)
add_dependencies(all_unit_tests euler_unit_tests)

euler_add_unit_test(gas_thermo "Euler/test_GasThermo.cpp" "cfv;geom;dnds")
euler_add_unit_test(riemann_solvers "Euler/test_RiemannSolvers.cpp" "cfv;geom;dnds")
euler_add_unit_test(rans "Euler/test_RANS.cpp" "cfv;geom;dnds")
euler_add_unit_test(limited_vr "Euler/test_LimitedVR.cpp" "cfv;geom;dnds")
if(DNDS_USE_CANTERA)
    euler_add_unit_test(source_chemical "Euler/test_SourceChemical.cpp" "euler_library_NS_EX;euler_library_fast_NS_EX;cfv;geom;dnds")
    euler_add_unit_test(reactive_split_indicator "Euler/test_ReactiveSplitIndicator.cpp" "euler_library_NS_EX;euler_library_fast_NS_EX;cfv;geom;dnds")
    euler_add_unit_test(uv "Euler/test_UV.cpp" "euler_library_NS_EX;euler_library_fast_NS_EX;cfv;geom;dnds")
    euler_add_unit_test(physics_properties "Euler/test_PhysicsProperties.cpp" "euler_library_NS_EX;euler_library_fast_NS_EX;cfv;geom;dnds")
    euler_add_unit_test(chem_ode "Euler/test_ChemODE.cpp" "euler_library_NS_EX;euler_library_fast_NS_EX;cfv;geom;dnds")
    target_compile_definitions(euler_test_chem_ode PRIVATE CT_USE_SYSTEM_FMT=1)
    euler_add_mpi_test(evaluator_reactive "Euler/test_EulerEvaluatorReactive.cpp"
        "euler_library_NS_EX;euler_library_fast_NS_EX;cfv;geom;dnds")
endif()

# Common standalone migration probe, also built against pinned legacy sources.
# It is opt-in: regular unit tests do not require a second solver source tree.
add_executable(euler_migration_probe EXCLUDE_FROM_ALL Euler/MigrationParityProbe.cpp)
target_include_directories(euler_migration_probe PRIVATE ${DNDS_EXTERNAL_INCLUDES} ${DNDS_INCLUDES})
target_link_libraries(euler_migration_probe PRIVATE
    euler_library_NS euler_library_fast_NS euler_library_NS_SA euler_library_fast_NS_SA
    euler_library_NS_3D euler_library_fast_NS_3D cfv geom dnds ${DNDS_EXTERNAL_LIBS})

# EulerEvaluator pipeline test: needs full Euler libraries for NS, NS_SA, NS_3D
euler_add_mpi_test(evaluator_pipeline "Euler/test_EulerEvaluator.cpp"
    "euler_library_NS;euler_library_fast_NS;euler_library_NS_SA;euler_library_fast_NS_SA;euler_library_NS_3D;euler_library_fast_NS_3D;cfv;geom;dnds")
foreach(NP ${DNDS_TEST_NP_LIST})
    if(TEST euler_evaluator_pipeline_np${NP})
        _dnds_maybe_set_timeout(euler_evaluator_pipeline_np${NP} ${DNDS_TEST_TIMEOUT_PIPELINE})
    endif()
endforeach()

set(EULER_TEST_TARGETS
    euler_test_gas_thermo
    euler_test_riemann_solvers
    euler_test_rans
    euler_test_limited_vr
    euler_test_evaluator_pipeline
)
if(DNDS_USE_CANTERA)
    list(APPEND EULER_TEST_TARGETS
        euler_test_source_chemical
        euler_test_reactive_split_indicator
        euler_test_uv
        euler_test_physics_properties
        euler_test_chem_ode
        euler_test_evaluator_reactive)
endif()
foreach(T ${EULER_TEST_TARGETS})
    add_dependencies(euler_unit_tests ${T})
endforeach()

# ===================================================================
# Solver tests (ODE, Linear, Direct, Scalar)
# ===================================================================

add_custom_target(solver_unit_tests)
add_dependencies(all_unit_tests solver_unit_tests)

# All Solver tests are serial (header-only, no MPI needed at runtime)
function(solver_add_unit_test TEST_NAME SRC_FILE LIBS)
    set(TARGET_NAME "solver_test_${TEST_NAME}")
    add_executable(${TARGET_NAME} ${SRC_FILE})
    target_include_directories(${TARGET_NAME} PRIVATE
        ${DNDS_EXTERNAL_INCLUDE_DOCTEST}
        ${DNDS_EXTERNAL_INCLUDES}
        ${DNDS_INCLUDES})
    target_link_libraries(${TARGET_NAME} PRIVATE ${LIBS} ${DNDS_EXTERNAL_LIBS})
    if(DNDS_USE_CCACHE)
        set_property(TARGET ${TARGET_NAME} PROPERTY CXX_COMPILER_LAUNCHER ${DNDS_CCACHE_EXEC})
    endif()
    set_target_properties(${TARGET_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        EXCLUDE_FROM_ALL ON)
    add_test(NAME solver_${TEST_NAME} COMMAND ${TARGET_NAME})
    _dnds_maybe_set_timeout(solver_${TEST_NAME} ${DNDS_TEST_TIMEOUT_SOLVER})
    set_tests_properties(solver_${TEST_NAME} PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=${DNDS_TEST_OMP_THREADS}")
endfunction()

solver_add_unit_test(ode "Solver/test_ODE.cpp" "dnds")
solver_add_unit_test(linear "Solver/test_Linear.cpp" "dnds")
solver_add_unit_test(direct "Solver/test_Direct.cpp" "dnds")
solver_add_unit_test(scalar "Solver/test_Scalar.cpp" "dnds")

set(SOLVER_TEST_TARGETS
    solver_test_ode
    solver_test_linear
    solver_test_direct
    solver_test_scalar
)
foreach(T ${SOLVER_TEST_TARGETS})
    add_dependencies(solver_unit_tests ${T})
endforeach()

geom_add_mpi_test(native_mesh_assertions "Geom/test_NativeMeshAssertions.cpp" "geom;dnds")
add_dependencies(geom_unit_tests geom_test_native_mesh_assertions)
