#pragma once
#include "NativeMeshAssembler.hpp"
#include "DNDS/Serializer/SerializerBase.hpp"

namespace DNDS::Geom
{
    /// Decode native full archives or legacy primary archives after exact
    /// collectively agreed schema selection. The serializer is already open.
    std::shared_ptr<const NativeMeshState> ReadCompatibleNativeMeshArchive(
        const MPIInfo &mpi, Serializer::SerializerBaseSSP serializer, const std::string &name,
        Serializer::ArrayGlobalOffset partition = Serializer::ArrayGlobalOffset_Unknown,
        int ghostLayers = 1);

    /// One-way legacy primary archive import. Decode owned legacy array wire
    /// records directly into Runtime storage; saved ghosts/maps are ignored.
    /// Original IDs survive; a fresh namespace is created once on import.
    class NativeMeshLegacyArchive
    {
    public:
        static NativeOwnedMeshData ReadOwned(const MPIInfo &mpi, Serializer::SerializerBaseSSP serializer,
                                             const std::string &name,
                                             Serializer::ArrayGlobalOffset partition = Serializer::ArrayGlobalOffset_Unknown);
        static std::shared_ptr<const NativeMeshState> Read(const MPIInfo &mpi, Serializer::SerializerBaseSSP serializer,
                                                           const std::string &name,
                                                           Serializer::ArrayGlobalOffset partition = Serializer::ArrayGlobalOffset_Unknown,
                                                           int ghostLayers = 1);
    };
}
