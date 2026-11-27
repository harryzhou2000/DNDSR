# message("source dir is ${PROJECT_SOURCE_DIR}")

# project(CFV LANGUAGES CXX)

set(LIBNAME cfv)

set(LINKS geom)

set(LINKS_SHARED
geom_shared
dnds_shared
${DNDS_EXTERNAL_LIBS}
)

set(CPPS
Legacy/FiniteVolume.cpp
Legacy/VariationalReconstruction.cpp
Legacy/ModelEvaluator.cpp
Prepare/FiniteVolumePlan.cpp
Prepare/FiniteVolumePlan_Reconstruction.cpp
Prepare/ReconstructionAMGPlan.cpp
Detail/ReconstructionData_FaceBasis.cpp
Prepare/FiniteVolumePlan_Placement.cpp
Prepare/GradientPlan.cpp
Prepare/GradientPlan_Placement.cpp
Prepare/LimitedVRPlan.cpp
Prepare/LimitedVRPlan_Placement.cpp
Operators/GradientPlan_Operators.cpp
Operators/GradientPlan_Queued.cpp
Operators/GradientPlan_Conversion.cpp
Operators/GradientPlan_Limiter.cpp
CPU/GradientKernels.cpp
CPU/LimitedVRKernels.cpp
Operators/PolynomialSamplingPlan.cpp
CPU/PolynomialSamplingKernels.cpp
CPU/PolynomialSamplingKernels_Values.cpp
CPU/PolynomialSamplingKernels_ValueGradient.cpp
Operators/PolynomialSamplingPlan_Values.cpp
Operators/PolynomialSamplingPlan_ValueGradient.cpp
Operators/LimitedVRPlan_Operators.cpp
Operators/LimitedVRPlan_Queued.cpp
Prepare/PolynomialSamplingPlan.cpp
Prepare/PolynomialSamplingPlan_Sharing.cpp
Plans/DistributedInterfaceSamplingPlan.cpp
CPU/DistributedInterfaceSamplingKernels.cpp
CPU/ReconstructionBasisKernels.cpp
CPU/ReconstructionKernels.cpp
CPU/ReconstructionAMGKernels.cpp
CPU/ReconstructionKernels_Linear.cpp
Operators/FiniteVolumePlan_Reconstruction.cpp
Operators/FiniteVolumePlan_Linear.cpp
)
if (DNDS_USE_CUDA)
list(APPEND CPPS Operators/CUDA/CharacteristicKernels_CUDA.cu Operators/CUDA/GradientKernels_CUDA.cu Operators/CUDA/LimitedVRKernels_CUDA.cu Operators/CUDA/PolynomialSamplingKernels_CUDA.cu Operators/CUDA/ReconstructionDevice.cu Operators/CUDA/ReconstructionTensorCore.cu)
option(DNDS_CFV_CUDA_ENABLE_V1_KERNELS "Build historical CFV CUDA sweep comparison kernels" OFF)
list(APPEND CPPS Operators/CUDA/ReconstructionDeviceKernels_Staged.cu Operators/CUDA/ReconstructionDevice_General_FP64.cu
    Operators/CUDA/ReconstructionSweepV2.cu)
set(_catalog "")
foreach(M IN ITEMS 2 3 5 9 19)
    foreach(C IN ITEMS 4 5 6 7)
        set(_tu "${CMAKE_CURRENT_BINARY_DIR}/sweep_v2/M${M}C${C}.cu")
        configure_file(Operators/CUDA/ReconstructionSweepV2_Shape.cu.in "${_tu}" @ONLY)
        list(APPEND CPPS "${_tu}")
        string(APPEND _catalog "CFV_V2_SHAPE(${M}, ${C})\n")
        dnds_cuda_kernel_assertions("${_tu}")
    endforeach()
endforeach()
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/sweep_v2/catalog.inc" CONTENT "${_catalog}")
if(DNDS_CFV_CUDA_ENABLE_V1_KERNELS)
list(APPEND CPPS
    Operators/CUDA/ReconstructionDeviceKernels_Staged_FP64.cu
    Operators/CUDA/ReconstructionDeviceKernels_Staged_FP64_M2.cu
    Operators/CUDA/ReconstructionDeviceKernels_Staged_FP64_M3.cu
    Operators/CUDA/ReconstructionDeviceKernels_Staged_FP64_M5.cu
    Operators/CUDA/ReconstructionDeviceKernels_Staged_FP64_M9.cu
    Operators/CUDA/ReconstructionDeviceKernels_Staged_FP64_M19.cu
    Operators/CUDA/ReconstructionDeviceKernels_Staged_FloatFloat.cu
    Operators/CUDA/ReconstructionDeviceKernels_Staged_FP32.cu)
endif()
list(APPEND CPPS Operators/CUDA/DistributedInterfaceSamplingKernels.cu)
list(APPEND CPPS Operators/CUDA/ReconstructionAMGKernels_CUDA.cu)
# Register only this explicitly enumerated numerical CUDA catalog. Legacy CUDA
# and standalone test/benchmark sources retain their device checks.
set(_cfv_checked_kernel_sources ${CPPS})
list(FILTER _cfv_checked_kernel_sources INCLUDE REGEX "^Operators/CUDA/.*\\.cu$")
dnds_cuda_kernel_assertions(${_cfv_checked_kernel_sources})
endif()

file(GLOB EXPLICIT_INST_CPP_FILES CONFIGURE_DEPENDS "Legacy/ExplicitInstantiations/*.cpp")

list(APPEND CPPS ${EXPLICIT_INST_CPP_FILES})

dnds_add_lib_pair(${LIBNAME} "${CPPS}" "${LINKS}" "${LINKS_SHARED}" "${DNDS_PCH_TARGETS_FAST}"
    ON ON) # FAST, EXCLUDE
set(_cfv_compile_targets cfv cfv_shared)
if(TARGET cfv_objects)
    list(APPEND _cfv_compile_targets cfv_objects)
    target_link_libraries(cfv_objects PRIVATE DNDS::ComputeScalar)
endif()
foreach(_target IN LISTS _cfv_compile_targets)
    target_include_directories(${_target} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    if(DNDS_USE_CUDA AND DNDS_CFV_CUDA_ENABLE_V1_KERNELS)
        target_compile_definitions(${_target} PUBLIC DNDS_CFV_CUDA_ENABLE_V1_KERNELS)
    endif()
endforeach()
# Public headers and inline accessors expose the underlying native module.
target_link_libraries(${LIBNAME}_shared INTERFACE geom_shared)

target_link_libraries(${LIBNAME} PUBLIC DNDS::ComputeScalar)
target_link_libraries(${LIBNAME}_shared PUBLIC DNDS::ComputeScalar)

install(TARGETS ${LIBNAME}_shared
    LIBRARY DESTINATION "DNDSR/lib"
    ARCHIVE DESTINATION "DNDSR/lib"
    COMPONENT py OPTIONAL
)

# Also install into the python package tree for editable installs
install(TARGETS ${LIBNAME}_shared
    LIBRARY DESTINATION "${PROJECT_SOURCE_DIR}/python/DNDSR/_lib"
    COMPONENT py OPTIONAL
)

set(CPPS_BIND
Legacy/Benchmarks/BenchmarkFiniteVolume.cpp
Plans/FiniteVolumePlan_bind.cpp
Bindings/cfv_pybind11.cpp
Legacy/FiniteVolume_bind.cpp
Legacy/ModelEvaluator_bind.cpp
)

if(DNDS_USE_CUDA)
    list(APPEND CPPS_BIND Legacy/Benchmarks/BenchmarkFiniteVolume.cu)
endif()

dnds_add_py_module(${LIBNAME}_pybind11 "${CPPS_BIND}" "${LIBNAME}_shared;${LINKS_SHARED}" "${DNDS_PCH_TARGETS_FAST}"
    ON ON ON)  # SHARED,FAST, EXCLUDE

install(
    TARGETS ${LIBNAME}_pybind11
    DESTINATION "DNDSR/CFV/_ext"
    COMPONENT py OPTIONAL
)
install(
    TARGETS ${LIBNAME}_pybind11
    DESTINATION "${PROJECT_SOURCE_DIR}/python/DNDSR/CFV/_ext"
    COMPONENT py OPTIONAL
)
