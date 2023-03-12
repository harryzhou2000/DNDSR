# DNDSR Ideas

## 1.Primary Demands

- implement standard high order unstructured mesh compact finite volume method
  - standard VR reconstruction
  - standard compressible flow Riemann solver based RHS
  - standard LUSGS and LUSGS-GMRES solver
  - standard SDIRK ESDIRK and BDF implicit time marcher
  - standard WBAP and CWBAP limiter
- standard CFD operation capabilities
  - CGNS,tecplot or other input/output method
  - define (solver type?) input output and computation parameters via input (ascii notation) file
  - computation restart mechanism
- user defined capabilities
  - define user-specified analytic source-term/bc
  - define user-specified data source-term/bc
  - scripting capability?
- second order specific
  - traditional reconstruction and limiter
  - second order-precise reconstruction derivatives?
- precise derivatives
  - autodiff-ed RHS(how much)?

## 2.Development Paradigm

### Programming Language

C++ 17 standard!!!

### Third Party Infrastructure

- MPI standard process-wise parallelism
- CUDA gpu parallelism (optional)
- OpenMP thread-wise parallelism (optional)
- Eigen (+Autodiff) as basic dense data interface
- Eigen + blas/lapack series as dense solver
- PETSc as sparse solver (optional)
- CGNS/Gmsh/tecio/HDF5 as data io
- Rapidjson as configuration handler
- Python3 + numpy as script engine (optional)

+ Cantera for chemikinetics

- CMake for building? or xmake? or anything else?? or DIY?

### Feature Adoption

- Use concise and understandable templates.
- if constexpr used for template
- if constexpr used for solver distinction
- When dynamic needed, use dynamic
  - Use dynamic functional if possible
  - Use class inheriting if functional is inconvenient
  - Use shared pointer when sharing is possible
  - Use int/enum-key dictionary/switch if dynamic and mesh-wide
  - Use string-key dictionary if not mesh-wide
- Use lambda to rearrange code/in static functional.

### Framework

- Modularity
  - Use functional (static) in upper level method(ODE/linsolver), to be mesh/rhs irrelevant
  - Generic programmed main controller.
  - Use manual registering serialize and deserialize method.
  - Singleton profiler.
  - Generic Array which supports user-given data layout (template receives MPI_DATA_TYPE); automatically supports several simple ones: (int8-int64, float16-float64)
  - Tree of parameters: use rapidjson or else?
    - use manual registration
  - manual registration: list of std::function s;
    - needed by restart serialization
    - and param tree serialization
  - Element lib only handles one/compact-stencil element topo and geom
  - Mesh: provides cell centered data structure

- Data structure
  - Generic Array
    - like `Array<T, T_MPIType, T_ByteIndexer, T_ParIndexer>`
    - !or like `Array<T, T_MPIType, rowSize, T_ParIndexer, maxRowSize>`? using rowSize==-1 as Uniform, ==-2 as Dynamic, -3 as FixedMax
    - !or like `DistArray<T_ParIndexer, T_MPIType>: Array<T, rowSize, maxRowSize>`
    - Dynamic should be able to auto-compress (no manual thing)
    - 
    - R&W: 2D indexing nested
    - MPI auto decide and auto conduct
    - Local Indexer(To byte) and para indexer(To label) sharing
    - nested Ghost tree (with identical type of array)
    - Local Indexer and Para Indexer as template member (to be costless on simpler arrays)
    - Local Indexer can support: CSR, Table, and MaxlenTable, MaxlenCSR
    - For Tables, len and maxlen are dynamic as whole (no fully fixed)
    - indexing returns a T reference
    - T must be trivially copyable
    - Comm conducts on raw buffer with MPI_DATA_TYPE known, without manual buffering
  - Blocked Sparse Matrix/Dense Matrix Table
    - Use something like `Array<real,auto,Table/CSR,RankSequential>` to implement, size is 2 int_64 s put at front, and sets row size = max_nMat * (max_mat_size + 2) + 1
    - Col index nested in a collaborating `Array<index,auto,Table/CSR,...>`, logically cells2cells or nodes2nodes etc.
  - Blocked Vector
    - supports mat-vec operation (at least with matrix less)
    - wrapper of array-table
  - Sparse Adjacency Matrix
    - Use `Array<index,auto,Table/CSR,...>`
  - Scattered data packing/unpacking and transferring queue
    - Packing raw data, while only supports float64/int64
    - Metadata packing automated for [vector of eigen matrices], and other frequently used ones
  - Boundary and Volume conditions general representation !!
    - Serial kind of BC/VC/IC dictionary, directly from config file (no mpi)
    - For data driven BC/VC/IC, solution 1.use reader process; 2.mpi_io
  - File io queue/stream for restart file (de)serialization
  - class with dynamic metadata table (function table), for manual reflection registration
  - for restart data, use global object pool to avoid shared data writing
    - object pool maps address in memory and references in restart file
    
## 3.Modular Design