if(UNIX OR MINGW)
    cmake_minimum_required(VERSION 3.1)
else()
    cmake_minimum_required(VERSION 3.20)
endif()

# ##########################################
project(DNDSR LANGUAGES CXX) #! need to be in front of find_package(MPI)



# need CUDA project support ?
# manual perferences
set(CMAKE_VERBOSE_MAKEFILE ON)
# set(CMAKE_CXX_STANDARD_REQUIRED 17)
# set(CMAKE_CXX_STANDARD 17)

#! why?
# find_package(MPI REQUIRED)

set(DNDS_SUPRESS_INSERT_CHECK ON CACHE BOOL "close the inserted outputs")
set(DNDS_USE_NDEBUG_MACRO OFF CACHE BOOL "use NDEBUG macro for eigen output and cassert etc") # performance impact is small so off by default(without -DNDEBUG)
set(DNDS_USE_PARALLEL_MACRO OFF CACHE BOOL "for auto multi-thread?")
set(DNDS_USE_OMP ON CACHE BOOL "for openmp enabling")
set(DNDS_FAST_BUILD_FAST ON CACHE BOOL "don't -g on basic modules")



if(UNIX )
    set(CMAKE_CXX_COMPILER mpicxx CACHE FILEPATH "compiler with your MPI wrapping")
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a") # ! using static libs
elseif(MSVC OR WIN32 OR MINGW)
    # nothing
    find_package(MPI REQUIRED)
    if(MPI_CXX_FOUND)
        set (CMAKE_EXE_LINKER_FLAGS "${MPI_C_LINK_FLAGS} ${CMAKE_EXE_LINKER_FLAGS}")
    else()
        message((FATAL_ERROR "MPI NOT FOUND"))
    endif()
else()
    message(FATAL_ERROR "NOT YET IMPLEMENTED HERE")
endif()

set(CMAKE_INSTALL_PREFIX build/bin CACHE FILEPATH "install place")

set(CMAKE_BUILD_TYPE Release CACHE STRING "Release or Debug")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Release" "Debug" "RelWithDebInfo" "MinSizeRel")
set(CMAKE_USE_RELATIVE_PATHS ON)

message("external install hard guess at: external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")

# ##########################################
find_library(EXTERNAL_LIB_ZLIB z z_D REQUIRED)
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

find_path(EXTERNAL_INCLUDE_ZLIB zlib.h REQUIRED)
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
find_path(EXTERNAL_INCLUDE_BOOST boost PATHS 
    "${PROJECT_SOURCE_DIR}/external/boost_1_82_0" 
    NO_DEFAULT_PATH REQUIRED)
find_path(EXTERNAL_INCLUDE_CGAL CGAL PATHS 
    "${PROJECT_SOURCE_DIR}/external/CGAL-5.6/include" 
    "${PROJECT_SOURCE_DIR}/external/CGAL-5.6" 
    NO_DEFAULT_PATH REQUIRED)
find_path(EXTERNAL_INCLUDE_JSON json.hpp PATHS 
    "${PROJECT_SOURCE_DIR}/external/nlohmann" 
    NO_DEFAULT_PATH REQUIRED)
find_path(EXTERNAL_INCLUDE_CPPCODEC base32_hex.hpp PATHS
    "${PROJECT_SOURCE_DIR}/external/cppcodec" 
    NO_DEFAULT_PATH REQUIRED)
# find_path(EXTERNAL_INCLUDE_RAPIDJSON rapidjson 
#     PATHS 
#     "${PROJECT_SOURCE_DIR}/external/rapidjson/include" 
#     "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/include"
#     NO_DEFAULT_PATH)
find_path(EXTERNAL_INCLUDE_NANOFLANN nanoflann.hpp PATHS
    "${PROJECT_SOURCE_DIR}/external/nanoflann-1.4.3" 
    NO_DEFAULT_PATH REQUIRED)

find_path(EXTERNAL_DOXYGEN_AWESOME doxygen-awesome.css PATHS 
    "${PROJECT_SOURCE_DIR}/external/doxygen-awesome-css-2.2.1" 
    NO_DEFAULT_PATH)

set (EXTERNAL_LIBS "")

if (MPI_CXX_FOUND)
    set(EXTERNAL_LIBS ${MPI_CXX_LIBRARIES})
    message("MPI_CXX_LIBRARIES" ${MPI_CXX_LIBRARIES})
endif()

set(EXTERNAL_LIBS ${EXTERNAL_LIBS}
    ${EXTERNAL_LIB_ZLIB}
    ${EXTERNAL_LIB_CGNS}
    ${EXTERNAL_LIB_HDF5}
    ${EXTERNAL_LIB_METIS}
    ${EXTERNAL_LIB_PARMETIS}
    # ${EXTERNAL_LIB_TECIO}
    )
message("EXTERNAL_LIBS  " ${EXTERNAL_LIBS})

if(UNIX OR MINGW)
    set(EXTERNAL_LIBS ${EXTERNAL_LIBS}
    stdc++fs)
    if(UNIX)
        set(EXTERNAL_LIBS ${EXTERNAL_LIBS}
        dl)
    endif()
    if(MINGW)
        add_compile_options(-Wa,-mbig-obj) # too many sections
    endif()
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
            # add_compile_definitions(_GLIBCXX_PARALLEL)
            add_definitions(-D_GLIBCXX_PARALLEL)
        endif()
    endif()
    ### set warnings
    add_compile_options(-Wall -Wno-unused-but-set-variable -Wno-unused-variable -Wno-sign-compare)

elseif(WIN32 OR MSVC)
    set(EXTERNAL_LIBS ${EXTERNAL_LIBS} # no dl
    )
    message(${CMAKE_CXX_COMPILER_ID})
    # add_compile_options(/std:c++17 /IGNORE:C2124) # cl is buggy in c++17 mode ...
    add_compile_options(-std=c++17 --target=x86_64-pc-windows-gnu)
    if (DNDS_USE_OMP)
        find_package(OpenMP REQUIRED)
        if(OpenMP_CXX_FLAGS)
            set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS})
        else()
            message(FATAL_ERROR "OMP not found")
        endif()
    endif()
    if (DNDS_USE_PARALLEL_MACRO)
        # ! to be tested
    endif()

    ### set warnings
    add_compile_options(-Wall)
else()
    message(FATAL_ERROR "NOT YET IMPLEMENTED HERE")
endif()



set(EXTERNAL_INCLUDES 
    ${EXTERNAL_INCLUDE_ZLIB}
    ${EXTERNAL_INCLUDE_HDF5}
    ${EXTERNAL_INCLUDE_CGNS}
    ${EXTERNAL_INCLUDE_METIS}
    ${EXTERNAL_INCLUDE_PARMETIS}
    # ${EXTERNAL_INCLUDE_TECIO}
    ${EXTERNAL_INCLUDE_EIGEN}
    ${EXTERNAL_INCLUDE_JSON}
    # ${EXTERNAL_INCLUDE_RAPIDJSON}
    ${EXTERNAL_INCLUDE_CPPCODEC}
    ${EXTERNAL_INCLUDE_NANOFLANN}
    ${EXTERNAL_INCLUDE_BOOST}
    ${EXTERNAL_INCLUDE_CGAL}
    )

if (MPI_CXX_FOUND)
    set(EXTERNAL_INCLUDES ${MPI_CXX_INCLUDE_DIRS} ${EXTERNAL_INCLUDES})
endif()

set(DNDS_INCLUDES ${PROJECT_SOURCE_DIR}/src)

if (UNIX)
add_compile_options(-Wall -Wno-unused-but-set-variable -Wno-unused-variable -Wno-sign-compare)
endif()

if (DNDS_SUPRESS_INSERT_CHECK)
    # add_compile_definitions(NINSERT)
    add_definitions(-DNINSERT)
endif()
if (DNDS_USE_NDEBUG_MACRO)
    # add_compile_definitions(NDEBUG)
    add_definitions(-DNDEBUG)
else()
    add_compile_options(-UNDEBUG)
endif()
if (DNDS_USE_OMP)
    # add_compile_definitions(DNDS_USE_OMP)
    add_definitions(-DDNDS_USE_OMP)
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
json_Test
cgal_AABBTest
)

set(TESTS_DNDS
array_Test
arrayTrans_test
arrayDerived_test
serializerJSON_Test
)

set(TESTS_Geom
elements_Test
meshSerial_Test
partitionMeshSerial
)

set(TESTS_CFV
vrStatic_Test
vrBasic_Test
)

set(TESTS_Euler
euler
euler3D
eulerSA
eulerSA3D
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
src/DNDS/SerializerJSON.cpp
)

set(CPPS_Geom
src/Geom/Elements.cpp
src/Geom/Mesh.cpp
src/Geom/Mesh_Plts.cpp
src/Geom/Mesh_Serial_BuildCell2Cell.cpp
src/Geom/Mesh_Serial_ReadFromCGNS.cpp
src/Geom/Mesh_Serial_Partition.cpp
)

set(CPPS_CFV
src/CFV/VariationalReconstruction.cpp
)

set(CPPS_Euler
src/Euler/EulerEvaluator_EvaluateDt_NS_2D.cpp
src/Euler/EulerEvaluator_EvaluateDt_NS_3D.cpp
src/Euler/EulerEvaluator_EvaluateDt_NS.cpp
src/Euler/EulerEvaluator_EvaluateDt_NS_SA.cpp
src/Euler/EulerEvaluator_EvaluateDt_NS_SA_3D.cpp
src/Euler/EulerEvaluator_EvaluateRHS_NS_2D.cpp
src/Euler/EulerEvaluator_EvaluateRHS_NS_3D.cpp
src/Euler/EulerEvaluator_EvaluateRHS_NS.cpp
src/Euler/EulerEvaluator_EvaluateRHS_NS_SA.cpp
src/Euler/EulerEvaluator_EvaluateRHS_NS_SA_3D.cpp
src/Euler/EulerEvaluator_NS_2D.cpp
src/Euler/EulerEvaluator_NS_3D.cpp
src/Euler/EulerEvaluator_NS.cpp
src/Euler/EulerEvaluator_NS_SA.cpp
src/Euler/EulerEvaluator_NS_SA_3D.cpp
src/Euler/EulerSolver_NS_2D.cpp
src/Euler/EulerSolver_NS_3D.cpp
src/Euler/EulerSolver_NS.cpp
src/Euler/EulerSolver_NS_SA.cpp
src/Euler/EulerSolver_NS_SA_3D.cpp
)

set(CPPS_FAST
${CPPS_DNDS}
${CPPS_Geom}
${CPPS_CFV}
)

set(CPPS
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
    if(DNDS_FAST_BUILD_FAST)
        if(LLVM OR MINGW OR UNIX)
            target_compile_options(${CPP_NAME_A} PUBLIC -O3)
        elseif(WIN32 OR MSVC)
            # target_compile_options(${CPP_NAME_A} PUBLIC /O2)
            target_compile_options(${CPP_NAME_A} PUBLIC -O3)
        else()
            message(FATAL_ERROR "NOT YET IMPLEMENTED HERE")
        endif()
    endif()
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



#######################################################
## doxygen
#######################################################

find_package(Doxygen)
if (DOXYGEN_FOUND)
    message(STATUS "Doxygen Found") 
    #! TODO: use automatic config Doxygen.in file
    ##### using custom DOXYGEN
    set(DOXYGEN_IN ${PROJECT_SOURCE_DIR}/docs/Doxyfile)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    add_custom_target(docs ALL 
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT Doing Doxygen
        VERBATIM
    )
    execute_process(COMMAND python ${CMAKE_SOURCE_DIR}/docs/getAllAttachForDox.py 
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/docs")

    # find_file(DOXYGEN_PDF_FILES docs/*.pdf NO_DEFAULT_PATH)
    # message(STATUS ${DOXYGEN_PDF_FILES})
    ##### using CMAKE convenient DOXYGEN!only in 3.9 +!!!
    
else(DOXYGEN_FOUND)
    message("Doxygen Not Found")
endif(DOXYGEN_FOUND)



