
function(add_fast_flags LIBNAME)

    list(FIND DNDS_NODEBUG_MODULES "${LIBNAME}" _found_index)
    set(DO_FAST ${DNDS_FAST_BUILD_FAST})
    if(NOT _found_index EQUAL -1)
        message(WARNING "${LIBNAME} found in no-debug modules, forcing -g0 and optimization")
        set(DO_FAST ON)
    endif()
    if(DO_FAST)
        if(LLVM OR MINGW OR UNIX)
            target_compile_options(${LIBNAME} PRIVATE -O3 -g0)
            if(${CMAKE_BUILD_TYPE} STREQUAL "Debug" AND ${DNDS_USE_NO_OMIT_FRAME_POINTER})
                target_compile_options(${LIBNAME} PRIVATE -fno-omit-frame-pointer)
            endif()
        elseif(WIN32 OR MSVC)
            # target_compile_options(${CPP_NAME_A} PRIVATE /O2)
            if(${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
                target_compile_options(${LIBNAME} PRIVATE /O2 /g0)
            else()
                target_compile_options(${LIBNAME} PRIVATE -O3 -g0)
            endif()
        else()
            message(FATAL_ERROR "NOT YET IMPLEMENTED HERE")
        endif()
    endif()
endfunction(add_fast_flags)

function(dnds_add_lib_fast LIBNAME CPPS LINKS PCH_TARGET)
    message(DEBUG "lib fast ${LIBNAME} CPPS is ${CPPS}")
    add_library(${LIBNAME} STATIC ${CPPS})
    target_include_directories(${LIBNAME} PRIVATE ${DNDS_EXTERNAL_INCLUDES} PRIVATE ${DNDS_INCLUDES})
    target_link_libraries(${LIBNAME} INTERFACE ${LINKS})
    # target_link_libraries(${LIBNAME} PRIVATE ${PCH_LINKS})
    if((CMAKE_VERSION VERSION_GREATER_EQUAL "3.16") AND DNDS_USE_PRECOMPILED_HEADER)
        target_precompile_headers(${LIBNAME} REUSE_FROM ${PCH_TARGET})
    endif()
    if( DNDS_USE_CCACHE )
        set_property(TARGET ${LIBNAME} PROPERTY CXX_COMPILER_LAUNCHER ccache)
    endif()
    add_fast_flags(${LIBNAME})
endfunction(dnds_add_lib_fast)

function(dnds_add_lib LIBNAME CPPS LINKS PCH_TARGET)
    message(DEBUG "lib ${LIBNAME} CPPS is ${CPPS}")
    add_library(${LIBNAME} STATIC ${CPPS})
    target_include_directories(${LIBNAME} PRIVATE ${DNDS_EXTERNAL_INCLUDES} PRIVATE ${DNDS_INCLUDES})
    target_link_libraries(${LIBNAME} INTERFACE ${LINKS})
    # target_link_libraries(${LIBNAME} PRIVATE ${PCH_LINKS})
    if((CMAKE_VERSION VERSION_GREATER_EQUAL "3.16") AND DNDS_USE_PRECOMPILED_HEADER)
        target_precompile_headers(${LIBNAME} REUSE_FROM ${PCH_TARGET})
    endif()
    if( DNDS_USE_CCACHE )
        set_property(TARGET ${LIBNAME} PROPERTY CXX_COMPILER_LAUNCHER ccache)
    endif()
endfunction(dnds_add_lib)

function(dnds_add_lib_pch_fast LIBNAME CPPS HPPS LINKS)
    message(DEBUG "lib pch fast ${LIBNAME} CPPS is ${CPPS}") 
    message(DEBUG "lib pch fast ${LIBNAME} HPPS is ${HPPS}")
    add_library(${LIBNAME} STATIC ${CPPS})
    if((CMAKE_VERSION VERSION_GREATER_EQUAL "3.16") AND DNDS_USE_PRECOMPILED_HEADER)
        target_precompile_headers(${LIBNAME} PRIVATE ${HPPS})
    endif()
    if( DNDS_USE_CCACHE )
        set_property(TARGET ${LIBNAME} PROPERTY CXX_COMPILER_LAUNCHER ccache)
    endif()
    target_include_directories(${LIBNAME} PRIVATE ${DNDS_EXTERNAL_INCLUDES} PRIVATE ${DNDS_INCLUDES})
    target_link_libraries(${LIBNAME} PRIVATE ${LINKS})
    add_fast_flags(${LIBNAME})
endfunction(dnds_add_lib_pch_fast)

function(dnds_add_lib_pch LIBNAME CPPS HPPS LINKS)
    message(DEBUG "lib pch ${LIBNAME} CPPS is ${CPPS}") 
    message(DEBUG "lib pch ${LIBNAME} HPPS is ${HPPS}")
    add_library(${LIBNAME} STATIC ${CPPS})
    if((CMAKE_VERSION VERSION_GREATER_EQUAL "3.16") AND DNDS_USE_PRECOMPILED_HEADER)
        target_precompile_headers(${LIBNAME} PRIVATE ${HPPS})
    endif()
    if( DNDS_USE_CCACHE )
        set_property(TARGET ${LIBNAME} PROPERTY CXX_COMPILER_LAUNCHER ccache)
    endif()
    target_include_directories(${LIBNAME} PRIVATE ${DNDS_EXTERNAL_INCLUDES} PRIVATE ${DNDS_INCLUDES})
    target_link_libraries(${LIBNAME} PRIVATE ${LINKS})
endfunction(dnds_add_lib_pch)

macro(dnds_variable_to_parent_scope V)
    set(${V} ${${V}} PARENT_SCOPE)
endmacro()
