#pragma once

#include "Buffer.hpp"
#include <string>

namespace DNDS::Runtime
{
    /// Placement metadata is immutable once bound to a field. Stable entity IDs
    /// and routing maps belong to the domain/mesh rather than the byte allocation.
    struct Distribution
    {
        uint64_t domain = 0;
        uint64_t epoch = 0;
        size_t owned = 0;
        size_t ghosts = 0;
    };

    struct DenseLayout
    {
        size_t rows = 0;
        size_t components = 1;
        size_t stride = 1;

        size_t Elements() const
        {
            DNDS_check_throw_info(components && stride >= components, "invalid dense field layout");
            return CheckedSize::Multiply(rows, stride);
        }
    };

    /// Initial contiguous field representation. Ragged topology stays in Array;
    /// future SoA/chunked views can coexist without changing Buffer or routing.
    template <class T>
    class Field
    {
        static_assert(std::is_trivially_copyable_v<T>);
        Buffer _values;
        DenseLayout _layout;
        std::shared_ptr<const Distribution> _distribution;
        std::string _association;

    public:
        Field(DenseLayout layout, Distribution distribution, std::string association,
              Device device = Device::Host())
            : _layout(layout), _distribution(std::make_shared<const Distribution>(distribution)),
              _association(std::move(association))
        {
            DNDS_check_throw_info(distribution.owned <= layout.rows && distribution.ghosts == layout.rows - distribution.owned,
                                  "field distribution does not cover its rows");
            _values = Buffer::Allocate(CheckedSize::Multiply(layout.Elements(), sizeof(T)), device);
        }
        const DenseLayout &Layout() const { return _layout; }
        const Distribution &GetDistribution() const { return *_distribution; }
        const std::string &Association() const { return _association; }
        const Buffer &Values() const { return _values; }
        ReadAccess Read(ExecutionContext context = ExecutionContext::Host()) const { return _values.Read(context); }
        WriteAccess Write(ExecutionContext context = ExecutionContext::Host()) { return _values.Write(context); }
        CopyOperation CopyFrom(const Field &source, ExecutionContext context = ExecutionContext::Host())
        {
            DNDS_check_throw_info(source._layout.rows == _layout.rows && source._layout.components == _layout.components &&
                                      source._layout.stride == _layout.stride && source._distribution->domain == _distribution->domain &&
                                      source._distribution->epoch == _distribution->epoch &&
                                      source._distribution->owned == _distribution->owned && _association == source._association,
                                  "field copy requires identical association/layout/distribution; use an explicit transfer otherwise");
            return Copy(source._values, _values, context);
        }
    };
}
