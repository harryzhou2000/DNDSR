
cmake_minimum_required(VERSION 3.1)

# need CUDA project support ?
# manual perferences
set(CMAKE_VERBOSE_MAKEFILE ON)
set(DNDS_SUPRESS_INSERT_CHECK ON CACHE BOOL "close the inserted outputs")
set(DNDS_USE_NDEBUG_MACRO OFF CACHE BOOL "use DEBUG macro for eigen output and cassert etc") # performance impact is small so off by default(without -DNDEBUG)

if(UNIX OR MINGW)
    set(CMAKE_CXX_COMPILER mpicxx CACHE FILEPATH "compiler with your MPI wrapping")
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a") # ! using static libs
else()
    message(FATAL_ERROR "NOT YET IMPLEMENTED HERE")
endif()

set(CMAKE_INSTALL_PREFIX build/bin CACHE FILEPATH "install place")

set(CMAKE_BUILD_TYPE Release CACHE STRING "Release or Debug")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Release" "Debug" "RelWithDebInfo" "MinSizeRel")
set(CMAKE_USE_RELATIVE_PATHS ON)

# ##########################################
project(DNDSR LANGUAGES CXX)

# ##########################################
if(UNIX)
    find_library(EXTERNAL_LIB_HDF5 hdf5 PATHS "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}" NO_DEFAULT_PATH)
    find_library(EXTERNAL_LIB_CGNS cgns PATHS "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}" NO_DEFAULT_PATH)
    find_library(EXTERNAL_LIB_METIS metis PATHS "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}" NO_DEFAULT_PATH)
    find_library(EXTERNAL_LIB_PARMETIS parmetis PATHS "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}" NO_DEFAULT_PATH)
    find_library(EXTERNAL_LIB_TECIO tecio PATHS "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}" NO_DEFAULT_PATH)
    find_path(EXTERNAL_INCLUDE_HDF5 hdf5.h PATHS "${PROJECT_SOURCE_DIR}/external/HDF5/include" NO_DEFAULT_PATH)
    find_path(EXTERNAL_INCLUDE_CGNS cgnslib.h PATHS "${PROJECT_SOURCE_DIR}/external/cgns/include" NO_DEFAULT_PATH)
    find_path(EXTERNAL_INCLUDE_METIS metis.h PATHS "${PROJECT_SOURCE_DIR}/external/metis/include" NO_DEFAULT_PATH)
    find_path(EXTERNAL_INCLUDE_PARMETIS parmetis.h PATHS "${PROJECT_SOURCE_DIR}/external/parmetis/include" NO_DEFAULT_PATH)
    find_path(EXTERNAL_INCLUDE_TECIO TECIO.h PATHS "${PROJECT_SOURCE_DIR}/external/tecio/include" NO_DEFAULT_PATH)
    find_path(EXTERNAL_INCLUDE_EIGEN eigen3.pc.in PATHS "${PROJECT_SOURCE_DIR}/external/eigen" NO_DEFAULT_PATH)
endif()

set (EXTERNAL_LIBS "")

if(UNIX OR MINGW)
    set(EXTERNAL_LIBS dl z)
else()
    message(FATAL_ERROR "NOT YET IMPLEMENTED HERE")
endif()

set(EXTERNAL_LIBS ${EXTERNAL_LIBS}\\
    ${EXTERNAL_LIB_CGNS}\\
    ${EXTERNAL_LIB_HDF5}\\
    ${EXTERNAL_LIB_METIS}\\
    ${EXTERNAL_LIB_PARMETIS}\\
    ${EXTERNAL_LIB_TECIO})

set(EXTERNAL_INCLUDES \\
    ${EXTERNAL_INCLUDE_HDF5}\\
    ${EXTERNAL_INCLUDE_CGNS}\\
    ${EXTERNAL_INCLUDE_METIS}\\
    ${EXTERNAL_INCLUDE_PARMETIS}\\
    ${EXTERNAL_INCLUDE_TECIO}\\
    ${EXTERNAL_INCLUDE_EIGEN})

message(STATUS "EXTERNAL_LIBS ${EXTERNAL_LIBS}")
message(STATUS "EXTERNAL_INCLUDES ${EXTERNAL_INCLUDES}")

add_executable(cgns_APITest test/external/cgns_APITest.cpp)
target_link_libraries(cgns_APITest PUBLIC ${EXTERNAL_LIBS})
target_include_directories(cgns_APITest PUBLIC ${EXTERNAL_INCLUDES})