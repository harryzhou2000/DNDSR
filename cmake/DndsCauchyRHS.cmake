include_guard(GLOBAL)

# Accept the original misspelling as an input alias. Keep one authoritative
# cache entry so a later reconfigure cannot silently restore an old setting.
if(DEFINED DNDS_CAUCHY_ENBALE_3D)
    message(STATUS "Mapping DNDS_CAUCHY_ENBALE_3D to DNDS_CAUCHY_ENABLE_3D")
    set(DNDS_CAUCHY_ENABLE_3D "${DNDS_CAUCHY_ENBALE_3D}" CACHE BOOL "Build Cauchy for 3D meshes" FORCE)
    unset(DNDS_CAUCHY_ENBALE_3D CACHE)
endif()
option(DNDS_CAUCHY_ENABLE_COMPRESSIBLE "Build Cauchy for compressible equations" ON)
option(DNDS_CAUCHY_ENABLE_ACM "Build Cauchy for artificial compressibility" ON)
option(DNDS_CAUCHY_ENABLE_2D "Build Cauchy for 2D meshes" ON)
option(DNDS_CAUCHY_ENABLE_3D "Build Cauchy for 3D meshes" ON)
set(_cauchy_families)
if(DNDS_CAUCHY_ENABLE_COMPRESSIBLE)
    list(APPEND _cauchy_families PerfectGas)
endif()
if(DNDS_CAUCHY_ENABLE_ACM)
    list(APPEND _cauchy_families ACM)
endif()
set(_cauchy_dimensions)
foreach(_dim IN ITEMS 2 3)
    if(DNDS_CAUCHY_ENABLE_${_dim}D)
        list(APPEND _cauchy_dimensions ${_dim})
    endif()
endforeach()
if(NOT _cauchy_families OR NOT _cauchy_dimensions)
    message(FATAL_ERROR "Cauchy requires at least one equation family and one mesh dimension")
endif()

# Generate the manifest's polynomial RHS source catalog once, before targets are
# created. Both static/shared libraries must depend on the returned target, and
# add the returned include directory privately. Generated files stay build-local.
#
#   dnds_generate_cauchy_rhs(_rhs_sources _rhs_include _rhs_codegen)
#   target_sources(cauchy PRIVATE ${_rhs_sources})
#   target_include_directories(cauchy PRIVATE "${_rhs_include}")
#   add_dependencies(cauchy ${_rhs_codegen})
#
# Reuse those three values for cauchy_shared rather than invoking this twice.
function(dnds_generate_cauchy_rhs out_sources out_include_dir out_target family)
    set(_codegen_target dnds_${family}_codegen)
    if(TARGET ${_codegen_target})
        message(FATAL_ERROR "dnds_generate_cauchy_rhs must be called once and its outputs shared by consumers")
    endif()

    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    get_filename_component(_dnds_source_root "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.." ABSOLUTE)
    set(_dnds_generator "${_dnds_source_root}/scripts/generate_${family}.py")
    set(_dnds_manifest "${_dnds_source_root}/scripts/manifests/cauchy_rhs_shapes.json")
    set(_dnds_output_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/${family}")
    set(_dnds_index "${_dnds_output_dir}/FiniteVolumeRHSKernels_Sources.cmake")
    set(_dnds_command "${Python3_EXECUTABLE}" "${_dnds_generator}"
        --manifest "${_dnds_manifest}" --output-dir "${_dnds_output_dir}"
        --families ${_cauchy_families} --dimensions ${_cauchy_dimensions})

    # Input changes rerun configuration, where the full catalog is needed before
    # CMake constructs either Make or Ninja's compilation graph. Included .cmake
    # fragments (this file and the generated index) are also tracked by CMake.
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${_dnds_generator}" "${_dnds_source_root}/scripts/generate_cauchy_rhs.py" "${_dnds_manifest}")
    execute_process(COMMAND ${_dnds_command}
        RESULT_VARIABLE _dnds_result
        OUTPUT_VARIABLE _dnds_stdout
        ERROR_VARIABLE _dnds_stderr)
    if(NOT "${_dnds_result}" STREQUAL "0")
        message(FATAL_ERROR "Cauchy RHS generation failed (${_dnds_result}):\n${_dnds_stdout}${_dnds_stderr}")
    endif()
    include("${_dnds_index}")
    foreach(_dnds_output IN LISTS DNDS_CAUCHY_RHS_GENERATED_OUTPUTS)
        if(NOT EXISTS "${_dnds_output}")
            message(FATAL_ERROR "Cauchy RHS generator omitted required output: ${_dnds_output}")
        endif()
    endforeach()
    # Treat generated files as configuration inputs too. This covers the Make
    # multi-output-rule case where a secondary output is deleted while the first
    # output still exists: the pre-build configuration check restores the whole
    # catalog before it can use an incomplete source graph.
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        ${DNDS_CAUCHY_RHS_GENERATED_OUTPUTS})

    # Configuration restores missing outputs before either build tool schedules
    # consumers. Do not also attach a multi-output custom command: Make adds
    # touch rules for its secondary outputs, which would change generated source
    # timestamps on a no-op build and trigger another configuration. The private
    # CUDA implementation remains a normal compiler include dependency.
    add_custom_target(${_codegen_target}
        DEPENDS ${DNDS_CAUCHY_RHS_GENERATED_OUTPUTS})
    set_source_files_properties(${DNDS_CAUCHY_RHS_GENERATED_SOURCES}
        "${DNDS_CAUCHY_RHS_GENERATED_DECLARATIONS}" PROPERTIES GENERATED TRUE)

    set(${out_sources} "${DNDS_CAUCHY_RHS_GENERATED_SOURCES}" PARENT_SCOPE)
    set(${out_include_dir} "${DNDS_CAUCHY_RHS_GENERATED_INCLUDE_DIR}" PARENT_SCOPE)
    set(${out_target} ${_codegen_target} PARENT_SCOPE)
endfunction()
