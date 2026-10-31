# message("source dir is ${PROJECT_SOURCE_DIR}")

# project(GEOM LANGUAGES CXX)

set(LIBNAME geom)

set(LINKS
# dnds
# ${DNDS_EXTERNAL_LIBS}
)

set(LINKS_SHARED
dnds_shared
${DNDS_EXTERNAL_LIBS}
)

message(STATUS "DNDS_PCH_TARGETS: ${DNDS_PCH_TARGETS}")

set(CPPS
Elements.cpp
PeriodicInfo.cpp
Interface/BoundaryInterface.cpp
Connectivity/MeshConnectivity.cpp
Connectivity/MeshConnectivity_Native.cpp
Connectivity/AdjacencyRegistry.cpp
Connectivity/MeshConnectivity_Ghost.cpp
Unstructured/UnstructuredMesh.cpp
Access/MeshAccess.cpp
Geometry/MeshGeometryEdit.cpp
Geometry/MeshEdit.cpp
Unstructured/UnstructuredMeshSnapshot.cpp
Unstructured/UnstructuredMeshState.cpp
Unstructured/NativeMeshBuilder.cpp
Transition/NativeMeshTransition.cpp
Unstructured/NativeMeshPreparation.cpp
Unstructured/OwnedMeshPreparation.cpp
Transition/NativeMeshAssembler.cpp
IO/NativeMeshArchive.cpp
Compat/NativeMeshLegacyArchive.cpp
IO/NativeMeshCGNS.cpp
IO/NativeMeshOpenFOAM.cpp
IO/NativeMeshPeriodicImport.cpp
Connectivity/NativeMeshTopology.cpp
Connectivity/NativeMeshClosure.cpp
Refinement/NativeMeshRefinement.cpp
Refinement/NativeMeshElevation.cpp
Refinement/NativeMeshExtraction.cpp
Geometry/NativeMeshSmoothing.cpp
IO/NativeMeshOutput.cpp
IO/NativeMeshSerialOutput.cpp
Transition/MeshTransition.cpp
Access/EntitySpace.cpp
IO/UnstructuredMeshOutput.cpp
Compat/UnstructuredMeshSerialPartition.cpp
Refinement/Mesh_Elevation.cpp
Refinement/Mesh_Elevation_SmoothSolver.cpp
Refinement/Mesh_Elevation_SmoothV2.cpp
Geometry/MeshWallDistance.cpp
Transition/MeshReorder.cpp
Transition/SerialAdjReordering.cpp
)

if (DNDS_USE_CUDA)
list(APPEND CPPS PeriodicInfo.cu)
endif()

dnds_add_lib(${LIBNAME} "${CPPS}" "${LINKS}" "${DNDS_PCH_TARGETS}"
    OFF ON ON)  # SHARED,FAST, EXCLUDE

dnds_add_lib(${LIBNAME}_shared "${CPPS}" "${LINKS_SHARED}" "${DNDS_PCH_TARGETS}"
    ON ON ON)  # SHARED,FAST, EXCLUDE

# install(TARGETS ${LIBNAME}
#     LIBRARY DESTINATION "DNDSR/lib"
#     ARCHIVE DESTINATION "DNDSR/lib"
#     COMPONENT libs
# )

install(TARGETS ${LIBNAME}_shared
    LIBRARY DESTINATION "DNDSR/lib"
    ARCHIVE DESTINATION "DNDSR/lib"
    COMPONENT py
)

# Also install into the python package tree for editable installs
install(TARGETS ${LIBNAME}_shared
    LIBRARY DESTINATION "${PROJECT_SOURCE_DIR}/python/DNDSR/_lib"
    COMPONENT py
)

set(CPPS_BIND
Interface/BoundaryInterface_bind.cpp
geom_pybind11.cpp
Elements_bind.cpp
Unstructured/UnstructuredMesh_bind.cpp
)

list(APPEND CPPS_BIND
    Compat/AdjPairTracked_bind_inst/AdjPairTracked_bind_1_1_N.cpp
    Compat/AdjPairTracked_bind_inst/AdjPairTracked_bind_2_2_N.cpp
    Compat/AdjPairTracked_bind_inst/AdjPairTracked_bind_I_I_N.cpp)

dnds_add_py_module(${LIBNAME}_pybind11 "${CPPS_BIND}" "${LIBNAME}_shared;${LINKS_SHARED}" "${DNDS_PCH_TARGETS_FAST}"
    ON ON ON)  # SHARED,FAST, EXCLUDE

install(
  TARGETS ${LIBNAME}_pybind11
  DESTINATION "${PROJECT_SOURCE_DIR}/python/DNDSR/Geom/_ext"
  RUNTIME_DEPENDENCY_SET DNDS_LIB_DEPENDENCY_XXX
  COMPONENT py
)

