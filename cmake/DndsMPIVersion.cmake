# Check the MPI headers selected by FindMPI, not an unrelated mpirun on PATH.
# MPI_CXX_VERSION is the MPI standard version, not the Open MPI release.
include(CheckCXXSourceCompiles)
include(CMakePushCheckState)

function(dnds_check_mpi_version)
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_LIBRARIES MPI::MPI_CXX)
    # Recheck when a build directory is reconfigured with another MPI install.
    unset(DNDS_MPI_VERSION_SUPPORTED CACHE)
    unset(DNDS_MPI_VERSION_SUPPORTED)
    check_cxx_source_compiles([=[
        #include <mpi.h>
        #if defined(OMPI_MAJOR_VERSION) && defined(OMPI_MINOR_VERSION) && defined(OMPI_RELEASE_VERSION)
        #if OMPI_MAJOR_VERSION == 4 && OMPI_MINOR_VERSION == 1 && OMPI_RELEASE_VERSION == 0
        #error "DNDSR does not support Open MPI 4.1.0"
        #endif
        #endif
        int main() { return 0; }
    ]=] DNDS_MPI_VERSION_SUPPORTED)
    cmake_pop_check_state()
    if(NOT DNDS_MPI_VERSION_SUPPORTED)
        message(FATAL_ERROR
            "MPI version compatibility check failed. Open MPI 4.1.0 is not supported "
            "because it can hang in transport at large scales. Select another MPI "
            "release and reconfigure with matching compiler wrappers and libraries. "
            "If using another release, inspect the CMake compile-check log.")
    endif()
endfunction()
