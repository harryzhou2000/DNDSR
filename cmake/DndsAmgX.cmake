# Optional single-device linear preconditioner. DNDS owns all MPI exchange;
# this target supplies only the local AmgX provider and its CUDA dependencies.
option(DNDS_USE_AMGX "Enable the single-device AmgX preconditioner" OFF)
set(DNDS_AMGX_ROOT "" CACHE PATH "Install prefix from scripts/build_amgx.py")

if(DNDS_USE_AMGX)
    if(NOT DNDS_USE_CUDA)
        message(FATAL_ERROR "DNDS_USE_AMGX requires DNDS_USE_CUDA")
    endif()
    find_path(DNDS_AMGX_INCLUDE_DIR amgx_c.h
        HINTS "${DNDS_AMGX_ROOT}" PATH_SUFFIXES include REQUIRED)
    find_library(DNDS_AMGX_LIBRARY NAMES amgxsh amgx
        HINTS "${DNDS_AMGX_ROOT}" PATH_SUFFIXES lib lib64 REQUIRED)
    find_package(Threads REQUIRED)
    add_library(DNDS::AmgX UNKNOWN IMPORTED)
    set_target_properties(DNDS::AmgX PROPERTIES
        IMPORTED_LOCATION "${DNDS_AMGX_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${DNDS_AMGX_INCLUDE_DIR}"
        INTERFACE_COMPILE_DEFINITIONS DNDS_USE_AMGX
        INTERFACE_LINK_LIBRARIES "CUDA::cudart;CUDA::cusparse;CUDA::cublas;CUDA::cusolver;Threads::Threads;${CMAKE_DL_LIBS}")
    message(STATUS "Single-device AmgX: ${DNDS_AMGX_LIBRARY}")
endif()
