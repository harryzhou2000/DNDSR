set(SOLVER_CPPS
    AggregationAMG_Structure.cpp
    AggregationAMG.cpp
)

if(UNIX)
    add_library(solver_objects OBJECT EXCLUDE_FROM_ALL ${SOLVER_CPPS})
    set_target_properties(solver_objects PROPERTIES POSITION_INDEPENDENT_CODE ON)
    dnds_configure_library_compilation(solver_objects "${DNDS_PCH_TARGETS_FAST}" ON)
    target_link_libraries(solver_objects PRIVATE MPI::MPI_CXX ${DNDS_EXTERNAL_LIBS})
    dnds_add_lib(solver "$<TARGET_OBJECTS:solver_objects>" "dnds" "" OFF ON ON)
else()
    dnds_add_lib(solver "${SOLVER_CPPS}" "dnds" "${DNDS_PCH_TARGETS_FAST}" OFF ON ON)
endif()
