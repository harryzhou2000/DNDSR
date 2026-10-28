#pragma once

#include "Geom/Access/MeshAccess.hpp"
#include "Geom/Compat/MeshLegacyView.hpp"

namespace DNDS::Geom
{
    inline UnstructuredMeshLegacyView MeshReadAccess::LegacyHostView() const { return UnstructuredMeshLegacyView(HostView()); }
    inline UnstructuredMeshLegacyView MeshReadAccess::LegacyDeviceView() const { return UnstructuredMeshLegacyView(DeviceView()); }

    /// Convenience owner for legacy readonly syntax. Operator implementations
    /// bind HostView/DeviceView once in their prologue; loops use that plain
    /// borrowed value. Raw copies never extend this scope. Complete/Wait obey
    /// MeshReadAccess's same stream-join and sealing contract.
    class UnstructuredMeshLegacyRead
    {
        MeshReadAccess _read;
        UnstructuredMeshLegacyView _view;

    public:
        explicit UnstructuredMeshLegacyRead(MeshReadAccess read)
            : _read(std::move(read)), _view(_read._context.device == Runtime::Device::Host() ? _read.LegacyHostView() : _read.LegacyDeviceView()) {}
        UnstructuredMeshLegacyRead(const UnstructuredMeshLegacyRead &) = delete;
        UnstructuredMeshLegacyRead(UnstructuredMeshLegacyRead &&) noexcept = default;
        UnstructuredMeshLegacyRead &operator=(const UnstructuredMeshLegacyRead &) = delete;
        UnstructuredMeshLegacyRead &operator=(UnstructuredMeshLegacyRead &&) = delete;
        const UnstructuredMeshLegacyView &HostView() const
        {
            _read.HostView();
            return _view;
        }
        const UnstructuredMeshLegacyView &DeviceView() const
        {
            _read.DeviceView();
            return _view;
        }
        const UnstructuredMeshLegacyView *operator->() const { return &HostView(); }
        void Complete(Runtime::Completion completion) { _read.Complete(std::move(completion)); }
        void Wait() { _read.Wait(); }
    };
}
