# DNDSR Ideas

## Next Step:

- [x] wrap mesh for better access
- [ ] check dist topo and comm with a global BFS data
- [x] start CFV methods
- [x] check on general json parsing
- [ ] octree and periodic bc (using CGAL)
- [ ] mesh reader: CGNS abutting interface with 1-to-1 face
- [x] CFV limiters
- [x] Euler Solver
- [ ] Facial Value output in Euler Solver
- [ ] kdtree for euler solver's wall dist finding
- [ ] mesh: automatic global refinement


## About Periodic:

- periodic is defined with bnd faces
- periodic makes faces 2x duplicate, and nodes 8x duplicate max

steps in doing periodic ?: 

at least in serial reader:

meshinfo:

- coordSerial
- cell2nodeSerial
- bnd2nodeSerial
- cellElemInfoSerial
- bndElemInfoSerial
- bnd2cellSerial

well, when we have this, we need to add something:

- 1. detect face-to-face_donor
- 2. decide node de-duplication, and get new set of nodes_deduplicated
- 3. alter cell2node/bnd2node to point to correct nodes_deduplicated
- 4. coord now stores nodes_deduplicated
- partitioned:
- 5. use faceID to detect cells that are affected by periodic, record them
- 6. mask any mesh->coord[iNode] queries
- 7. in cfv, mask any coord-related (center, quadrature...) queries

the intention is to make topology unique and complete, treating faces with periodic like the inner ones

to ease cell2node query, in parallel:

- create appended physcial_coord
- for affected cells, convert cell2node(pointing to local) to point to physcial_coord
 (with minus meaning the physical_coord indices) (need a new adj state)
- create means of inverting the process (to get original cell2node(pointing to local) topo)
- when printing vis files, physcial_coord should be appended, and cell2node need to point to those, and coord data on physical_coord should copy those of original coord
- mesh serialization state should be unaffected by periodic (not pointing to appended, or the physical_node_to_actual_node mapping should also be serialized)
