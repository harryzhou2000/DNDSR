# message("source dir is ${PROJECT_SOURCE_DIR}")

set(LIBNAME euler_library)

set(LINKS
# cfv
# geom
# dnds
# ${DNDS_EXTERNAL_LIBS}
)

set(LINKS_SHARED
cfv_shared
geom_shared
dnds_shared
${DNDS_EXTERNAL_LIBS}
)

set(CPPS
EulerP_BC.cpp
EulerP_Evaluator.cpp
EulerP_Evaluator_impl.cpp
)
if (DNDS_USE_CUDA)
    list(APPEND CPPS
        EulerP_BC.cu
        EulerP_Evaluator_impl.cu
    )
endif()

# file(GLOB EXPLICIT_INST_CPP_FILES "_explicit_instantiation/*.cpp")
# list(APPEND CPPS ${EXPLICIT_INST_CPP_FILES})

dnds_add_lib_pair(eulerP "${CPPS}" "${LINKS}" "${LINKS_SHARED}" "${DNDS_PCH_TARGETS_FAST}"
    ON ON) # FAST, EXCLUDE

install(TARGETS eulerP_shared
    LIBRARY DESTINATION "DNDSR/lib"
    ARCHIVE DESTINATION "DNDSR/lib"
    COMPONENT py OPTIONAL
)

# Also install into the python package tree for editable installs
install(TARGETS eulerP_shared
    LIBRARY DESTINATION "${PROJECT_SOURCE_DIR}/python/DNDSR/_lib"
    COMPONENT py OPTIONAL
)


set(CPPS_BIND
eulerP_pybind11.cpp
)

dnds_add_py_module(eulerP_pybind11 "${CPPS_BIND}" "eulerP_shared;${LINKS_SHARED}" "${DNDS_PCH_TARGETS_FAST}"
    ON ON ON)  # SHARED,FAST, EXCLUDE

install(
  TARGETS eulerP_pybind11
  DESTINATION "DNDSR/EulerP/_ext"
  COMPONENT py OPTIONAL
)
install(
  TARGETS eulerP_pybind11
  DESTINATION "${PROJECT_SOURCE_DIR}/python/DNDSR/EulerP/_ext"
  COMPONENT py OPTIONAL
)