#pragma once
#include <mutex>

namespace DNDS::Serializer
{
    /// Process-local serialization for HDF5 builds without internal thread
    /// safety. SerializerH5 holds it for one open file session; Geom's direct
    /// VTKHDF writer holds it for its whole transaction. Nested files on the
    /// same thread are allowed. It does NOT establish distributed collective
    /// ordering: asynchronous writers still need a dedicated communicator and
    /// MPI_THREAD_MULTIPLE. Drain writers before raw external HDF5/CGNS calls
    /// or mesh import; those providers do not participate in this lock.
    inline std::recursive_mutex &HDF5Mutex()
    {
        static std::recursive_mutex mutex;
        return mutex;
    }
    using HDF5Access = std::unique_lock<std::recursive_mutex>;
}
