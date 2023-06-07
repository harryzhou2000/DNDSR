
cmake_minimum_required(VERSION 3.1)

# need CUDA project support ?
# manual perferences
set(CMAKE_VERBOSE_MAKEFILE ON)
# set(CMAKE_CXX_STANDARD_REQUIRED 17)
# set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)

set(DNDS_SUPRESS_INSERT_CHECK ON CACHE BOOL "close the inserted outputs")
set(DNDS_USE_NDEBUG_MACRO OFF CACHE BOOL "use NDEBUG macro for eigen output and cassert etc") # performance impact is small so off by default(without -DNDEBUG)
set(DNDS_USE_PARALLEL_MACRO OFF CACHE BOOL "for auto multi-thread?")
set(DNDS_USE_OMP ON CACHE BOOL "for openmp enabling")
set(DNDS_FAST_BUILD_FAST ON CACHE BOOL "don't -g on basic modules")


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
    find_library(EXTERNAL_LIB_HDF5 hdf5 PATHS 
        "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}" 
        "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/lib"
        NO_DEFAULT_PATH REQUIRED)
    find_library(EXTERNAL_LIB_CGNS cgns PATHS 
        "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}" 
        "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/lib"
        NO_DEFAULT_PATH REQUIRED)
    find_library(EXTERNAL_LIB_METIS metis PATHS 
        "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}"
        "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/lib"
         NO_DEFAULT_PATH REQUIRED)
    find_library(EXTERNAL_LIB_PARMETIS parmetis PATHS 
        "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}" 
        "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/lib"
        NO_DEFAULT_PATH REQUIRED)
    # find_library(EXTERNAL_LIB_TECIO tecio PATHS 
        # "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}" 
        # "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/lib"
        # NO_DEFAULT_PATH) #not needed for now
    
    find_path(EXTERNAL_INCLUDE_HDF5 hdf5.h PATHS 
        "${PROJECT_SOURCE_DIR}/external/HDF5/include"
        "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/include"
        NO_DEFAULT_PATH REQUIRED)
    find_path(EXTERNAL_INCLUDE_CGNS cgnslib.h PATHS 
        "${PROJECT_SOURCE_DIR}/external/cgns/include" 
        "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/include"
        NO_DEFAULT_PATH REQUIRED)
    find_path(EXTERNAL_INCLUDE_METIS metis.h PATHS 
        "${PROJECT_SOURCE_DIR}/external/metis/include" 
        "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/include"
        NO_DEFAULT_PATH REQUIRED)
    find_path(EXTERNAL_INCLUDE_PARMETIS parmetis.h PATHS 
        "${PROJECT_SOURCE_DIR}/external/parmetis/include" 
        "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/include"
        NO_DEFAULT_PATH REQUIRED)
    # find_path(EXTERNAL_INCLUDE_TECIO TECIO.h PATHS 
    #     "${PROJECT_SOURCE_DIR}/external/tecio/include" 
    #     "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/include"
    #     NO_DEFAULT_PATH) #!not needed for now
    find_path(EXTERNAL_INCLUDE_EIGEN eigen3.pc.in PATHS 
        "${PROJECT_SOURCE_DIR}/external/eigen-3.4.0" 
        "${PROJECT_SOURCE_DIR}/external/eigen" 
        NO_DEFAULT_PATH REQUIRED)
    find_path(EXTERNAL_INCLUDE_JSON json.hpp PATHS 
        "${PROJECT_SOURCE_DIR}/external/nlohmann" 
        NO_DEFAULT_PATH REQUIRED)
    # find_path(EXTERNAL_INCLUDE_RAPIDJSON rapidjson 
    #     PATHS 
    #     "${PROJECT_SOURCE_DIR}/external/rapidjson/include" 
    #     "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/include"
    #     NO_DEFAULT_PATH)
endif()

set (EXTERNAL_LIBS "")

set(EXTERNAL_LIBS ${EXTERNAL_LIBS}
    ${EXTERNAL_LIB_CGNS}
    ${EXTERNAL_LIB_HDF5}
    ${EXTERNAL_LIB_METIS}
    ${EXTERNAL_LIB_PARMETIS}
    # ${EXTERNAL_LIB_TECIO}
    )

if(UNIX OR MINGW)
    set(EXTERNAL_LIBS ${EXTERNAL_LIBS}
    dl 
    z)
    message(${CMAKE_CXX_COMPILER_ID})
    add_compile_options(-std=c++17)
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        # add_compile_options(-fopenmp)
        if (DNDS_USE_OMP)
            set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-fopenmp")
        endif()
        if (DNDS_USE_PARALLEL_MACRO)
            # add_compile_definitions(_GLIBCXX_PARALLEL)
            # doesn't seem available
        endif()
    endif()
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        # add_compile_options(-fopenmp)
        if (DNDS_USE_OMP)
            set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-fopenmp")
        endif()
        if (DNDS_USE_PARALLEL_MACRO)
            add_compile_definitions(_GLIBCXX_PARALLEL)
        endif()
    endif()
else()
    message(FATAL_ERROR "NOT YET IMPLEMENTED HERE")
endif()



set(EXTERNAL_INCLUDES 
    ${EXTERNAL_INCLUDE_HDF5}
    ${EXTERNAL_INCLUDE_CGNS}
    ${EXTERNAL_INCLUDE_METIS}
    ${EXTERNAL_INCLUDE_PARMETIS}
    # ${EXTERNAL_INCLUDE_TECIO}
    ${EXTERNAL_INCLUDE_EIGEN}
    ${EXTERNAL_INCLUDE_JSON}
    # ${EXTERNAL_INCLUDE_RAPIDJSON}
    )

set(DNDS_INCLUDES ${PROJECT_SOURCE_DIR}/src)

add_compile_options(-Wall -Wno-unused-but-set-variable -Wno-unused-variable -Wno-sign-compare)
if (DNDS_SUPRESS_INSERT_CHECK)
    add_compile_definitions(NINSERT)
endif()
if (DNDS_USE_NDEBUG_MACRO)
    add_compile_definitions(NDEBUG)
else()
    add_compile_options(-UNDEBUG)
endif()
if (DNDS_USE_OMP)
    add_compile_definitions(DNDS_USE_OMP)
endif()

# if (DNDS_USE_PARALLEL_MACRO)
#     add_compile_definitions(PARALLEL)
# endif()

message(STATUS "EXTERNAL_LIBS ${EXTERNAL_LIBS}")
message(STATUS "EXTERNAL_INCLUDES ${EXTERNAL_INCLUDES}")

set(INCLUDES \\
${DNDS_INCLUDES})



## test exes

set(TESTS_EXTERNAL
cgns_APITest
eigen_Test
STL_Test
)

set(TESTS_DNDS
array_Test
arrayTrans_test
arrayDerived_test
)

set(TESTS_Geom
elements_Test
meshSerial_Test
)

set(TESTS_CFV
vrStatic_Test
vrBasic_Test
)

set(TESTS_Euler
euler
euler3D
)

# set(OBJS_DNDS
# MPI
# )

# foreach(OBJ ${OBJS_DNDS})
#     add_library(${OBJ} OBJECT src/DNDS/${OBJ}.cpp)
#     target_link_libraries(${OBJ} PUBLIC ${EXTERNAL_LIBS})
#     target_include_directories(${OBJ} PUBLIC ${EXTERNAL_INCLUDES} PUBLIC ${INCLUDES})
# endforeach()

# set(OBJS
# ${OBJS_DNDS}
# )

set(CPPS_DNDS
src/DNDS/MPI.cpp
src/DNDS/Defines.cpp
src/DNDS/Profiling.cpp
src/DNDS/HardEigen.cpp
)

set(CPPS_Geom
src/Geom/Elements.cpp
src/Geom/Mesh.cpp
)

set(CPPS_CFV
src/CFV/VariationalReconstruction.cpp
)

set(CPPS_Euler
src/Euler/EulerEvaluator_EvaluateDt_NS_2D.cpp
src/Euler/EulerEvaluator_EvaluateDt_NS_3D.cpp
src/Euler/EulerEvaluator_EvaluateDt_NS.cpp
src/Euler/EulerEvaluator_EvaluateDt_NS_SA.cpp
src/Euler/EulerEvaluator_EvaluateRHS_NS_2D.cpp
src/Euler/EulerEvaluator_EvaluateRHS_NS_3D.cpp
src/Euler/EulerEvaluator_EvaluateRHS_NS.cpp
src/Euler/EulerEvaluator_EvaluateRHS_NS_SA.cpp
src/Euler/EulerEvaluator_NS_2D.cpp
src/Euler/EulerEvaluator_NS_3D.cpp
src/Euler/EulerEvaluator_NS.cpp
src/Euler/EulerEvaluator_NS_SA.cpp
)

set(CPPS_FAST
${CPPS_DNDS}
${CPPS_Geom}
)

set(CPPS
${CPPS_CFV}
${CPPS_Euler}
)

set(CPP_TARGETS
)
set(CPP_TARGETS_OBJ
)

foreach(CPP ${CPPS})
    string(REPLACE "/" "___" CPP_NAME ${CPP})
    # message(STATUS ${CPP_NAME})
    string(REPLACE ".cpp" "_" CPP_NAME_A ${CPP_NAME})
    add_library(${CPP_NAME_A} OBJECT ${CPP})
    target_include_directories(${CPP_NAME_A} PUBLIC ${EXTERNAL_INCLUDES} PUBLIC ${INCLUDES})
    list(APPEND CPP_TARGETS ${CPP_NAME_A})
    list(APPEND CPP_TARGETS_OBJ $<TARGET_OBJECTS:${CPP_NAME_A}>)
endforeach()

foreach(CPP ${CPPS_FAST})
    string(REPLACE "/" "___" CPP_NAME ${CPP})
    # message(STATUS ${CPP_NAME})
    string(REPLACE ".cpp" "_" CPP_NAME_A ${CPP_NAME})
    add_library(${CPP_NAME_A} OBJECT ${CPP})
    target_include_directories(${CPP_NAME_A} PUBLIC ${EXTERNAL_INCLUDES} PUBLIC ${INCLUDES})
    target_compile_options(${CPP_NAME_A} PUBLIC -O3)
    list(APPEND CPP_TARGETS ${CPP_NAME_A})
    list(APPEND CPP_TARGETS_OBJ $<TARGET_OBJECTS:${CPP_NAME_A}>)
endforeach()

macro(ADD_EXE_TEST EXES MAIN_DIR)
    message(STATUS ${EXES})
    foreach(EXE ${EXES})
        # add_executable(${EXE} test/external/${EXE}.cpp $<TARGET_OBJECTS:${OBJS}>)
        add_executable(${EXE} ${MAIN_DIR}/${EXE}.cpp ${CPP_TARGETS_OBJ})
        target_link_libraries(${EXE} PUBLIC ${EXTERNAL_LIBS})
        target_include_directories(${EXE} PUBLIC ${EXTERNAL_INCLUDES} PUBLIC ${INCLUDES})
        set_target_properties(${EXE} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test RUNTIME_OUTPUT_NAME ${EXE}.exe)
    endforeach()
endmacro(ADD_EXE_TEST)

ADD_EXE_TEST("${TESTS_EXTERNAL}" "test/external")
ADD_EXE_TEST("${TESTS_DNDS}" "test/DNDS")
ADD_EXE_TEST("${TESTS_Geom}" "test/Geom")
ADD_EXE_TEST("${TESTS_CFV}" "test/CFV")
ADD_EXE_TEST("${TESTS_Euler}" "test/Euler")




## doxygen
find_package(Doxygen)
if (DOXYGEN_FOUND)
    message(STATUS "Doxygen Found") 
    #! TODO: use automatic config Doxygen.in file
    ##### using custom DOXYGEN
    set(DOXYGEN_IN ${PROJECT_SOURCE_DIR}/docs/Doxyfile)
    add_custom_target(docs ALL 
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_IN}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT Doing Doxygen
        VERBATIM
    )
    ##### using CMAKE convenient DOXYGEN!only in 3.9 +!!!
    
else(DOXYGEN_FOUND)
    message("Doxygen Not Found")
endif(DOXYGEN_FOUND)



