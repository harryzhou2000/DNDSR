cmake_minimum_required(VERSION 3.1)

# message("source dir is ${PROJECT_SOURCE_DIR}")
include("${PROJECT_SOURCE_DIR}/cmakeCommonUtils.cmake") # from father

# project(DNDS LANGUAGES CXX)

set(LIBNAME dnds)

set(LINKS
# ${EXTERNAL_LIBS}
)

set(CPPS
MPI.cpp
Defines.cpp
Profiling.cpp
HardEigen.cpp
SerializerJSON.cpp
)

dnds_add_lib_fast(${LIBNAME} "${CPPS}" "${LINKS}")

