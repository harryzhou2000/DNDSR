#pragma once
#include "Geom/Transition/NativeMeshAssembler.hpp"
#include "DNDS/Serializer/SerializerBase.hpp"

namespace DNDS::Geom
{
    /// Owned-only schema selection/decoding; no intermediate readable mesh.
    NativeOwnedMeshData ReadCompatibleNativeMeshArchiveOwned(
        const MPIInfo &mpi, Serializer::SerializerBaseSSP serializer, const std::string &name,
        Serializer::ArrayGlobalOffset partition = Serializer::ArrayGlobalOffset_Unknown);
    /// Decode native full archives or legacy primary archives after exact
    /// collectively agreed schema selection. The serializer is already open.
    std::shared_ptr<const UnstructuredMeshState> ReadCompatibleNativeMeshArchive(
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
        static std::shared_ptr<const UnstructuredMeshState> Read(const MPIInfo &mpi, Serializer::SerializerBaseSSP serializer,
                                                                 const std::string &name,
                                                                 Serializer::ArrayGlobalOffset partition = Serializer::ArrayGlobalOffset_Unknown,
                                                                 int ghostLayers = 1);
    };
}
