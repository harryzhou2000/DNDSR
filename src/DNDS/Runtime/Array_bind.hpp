#pragma once

#include "Array.hpp"
#include "RowBuilder.hpp"
#include <pybind11/numpy.h>
#include <functional>
#include <typeindex>
#include <cstring>

namespace DNDS::Runtime
{
    /// Binding-side POD codec. The runtime allocation contains no Python objects.
    struct PythonElement
    {
        std::type_index type;
        pybind11::dtype dtype;
        std::string schema;
        size_t alignment;
        std::function<pybind11::object(const void *)> load;
        std::function<void(void *, pybind11::handle)> store;
    };
    using PythonElementShared = std::shared_ptr<const PythonElement>;

    inline PythonElementShared GetPythonElement(pybind11::handle type)
    {
        if (!pybind11::hasattr(type, "_dnds_array_element"))
            return {};
        auto capsule = pybind11::getattr(type, "_dnds_array_element").cast<pybind11::capsule>();
        auto *value = PyCapsule_GetPointer(capsule.ptr(), "DNDS.Runtime.PythonElement.v1");
        if (!value)
            throw pybind11::error_already_set();
        return *static_cast<PythonElementShared *>(value);
    }

    /// Register once per bound POD class, independently of Array widths/layouts.
    /// A structured NumPy dtype is optional; otherwise numpy() exposes opaque
    /// sizeof(T)-byte records and class-based element access remains available.
    template <class T, class... Options>
    void RegisterArrayElement(pybind11::class_<T, Options...> &type, std::string schema,
                              pybind11::object numpyDtype = pybind11::none())
    {
        static_assert(std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T> && alignof(T) <= 64);
        DNDS_check_throw_info(!schema.empty() && !pybind11::hasattr(type, "_dnds_array_element"), "duplicate or unnamed POD Array registration");
        auto dtype = numpyDtype.is_none() ? pybind11::dtype("V" + std::to_string(sizeof(T))) : pybind11::dtype::from_args(numpyDtype);
        DNDS_check_throw_info(size_t(dtype.itemsize()) == sizeof(T) && !pybind11::cast<bool>(dtype.attr("hasobject")) &&
                                  pybind11::cast<bool>(dtype.attr("isnative")),
                              "POD dtype size/representation mismatch");
        auto element = std::make_shared<PythonElement>(PythonElement{
            typeid(T), dtype, std::move(schema), alignof(T),
            [](const void *source) -> pybind11::object
            { T value; std::memcpy(&value, source, sizeof(T)); return pybind11::cast(std::move(value)); },
            [](void *destination, pybind11::handle object)
            { auto value = pybind11::cast<T>(object); std::memcpy(destination, &value, sizeof(T)); }});
        // The capsule belongs to the Python class; the codec does not retain
        // that class, so class -> capsule -> codec has no ownership cycle.
        type.attr("_dnds_array_element") = pybind11::capsule(new PythonElementShared(std::move(element)),
                                                             "DNDS.Runtime.PythonElement.v1", [](PyObject *capsule)
                                                             { delete static_cast<PythonElementShared *>(PyCapsule_GetPointer(capsule, "DNDS.Runtime.PythonElement.v1")); });
    }

    /// Python's dtype facade over the exact native allocation/layout record.
    /// Include this header in a binding TU; no width-specific Python classes.
    class PythonArray
    {
        ArrayStorage::Shared _storage;
        pybind11::dtype _dtype;
        PythonElementShared _element;
        pybind11::object _elementType = pybind11::none();

        static void ValidateDtype(const pybind11::dtype &dtype)
        {
            DNDS_check_throw_info(dtype.itemsize() > 0 && !pybind11::cast<bool>(dtype.attr("hasobject")) &&
                                      pybind11::cast<bool>(dtype.attr("isnative")) &&
                                      pybind11::cast<size_t>(dtype.attr("alignment")) <= 64,
                                  "Array requires a native POD dtype with alignment <= 64");
        }

    public:
        PythonArray(const PythonArray &other)
            : _storage(ArrayStorage::Share(other._storage)), _dtype(other._dtype), _element(other._element), _elementType(other._elementType) {}
        PythonArray &operator=(const PythonArray &other)
        {
            if (this != &other)
            {
                auto shared = ArrayStorage::Share(other._storage);
                _storage = std::move(shared);
                _dtype = other._dtype;
                _element = other._element;
                _elementType = other._elementType;
            }
            return *this;
        }
        PythonArray(PythonArray &&) noexcept = default;
        PythonArray &operator=(PythonArray &&) noexcept = default;
        PythonArray(RowLayout layout, pybind11::object dtypeOrClass, Device device = Device::Host())
        {
            _element = GetPythonElement(dtypeOrClass);
            _dtype = _element ? _element->dtype : pybind11::dtype::from_args(dtypeOrClass);
            if (_element)
                _elementType = std::move(dtypeOrClass);
            ValidateDtype(_dtype);
            _storage = ArrayStorage::Allocate(std::move(layout), _dtype.itemsize(), device);
        }
        PythonArray(RowLayout layout, pybind11::dtype dtype, Device device = Device::Host()) : _dtype(std::move(dtype))
        {
            ValidateDtype(_dtype);
            _storage = ArrayStorage::Allocate(std::move(layout), _dtype.itemsize(), device);
        }
        PythonArray(ArrayStorage::Shared storage, pybind11::dtype dtype)
            : _storage(std::move(storage)), _dtype(std::move(dtype))
        {
            ValidateDtype(_dtype);
            DNDS_check_throw_info(_storage && _storage->ElementBytes() == size_t(_dtype.itemsize()), "Array dtype size mismatch");
            const auto alignment = std::max<size_t>(1, pybind11::cast<size_t>(_dtype.attr("alignment")));
            DNDS_check_throw_info(reinterpret_cast<uintptr_t>(Storage().LegacyPointer()) % alignment == 0,
                                  "Array dtype alignment mismatch");
        }
        template <class T, class L>
        static PythonArray Share(const Array<T, L> &array)
        {
            if constexpr (!std::is_arithmetic_v<T>)
            {
                auto info = pybind11::detail::get_type_info(typeid(T), false);
                if (info)
                {
                    auto type = pybind11::reinterpret_borrow<pybind11::object>(reinterpret_cast<PyObject *>(info->type));
                    if (auto element = GetPythonElement(type))
                    {
                        PythonArray result(array.SharedStorage(), element->dtype);
                        result._element = std::move(element);
                        result._elementType = std::move(type);
                        return result;
                    }
                }
            }
            return {array.SharedStorage(), pybind11::dtype::of<T>()};
        }
        template <class T>
        bool MatchesType() const
        {
            if (_element)
                return _element->type == std::type_index(typeid(T));
            if constexpr (!std::is_arithmetic_v<T>)
            {
                // A class-only POD codec does not accept untyped opaque bytes.
                auto info = pybind11::detail::get_type_info(typeid(T), false);
                if (info && GetPythonElement(pybind11::handle(reinterpret_cast<PyObject *>(info->type))))
                    return false;
            }
            return _dtype.equal(pybind11::dtype::of<T>());
        }
        template <class T, class L = RowLayout>
        Array<T, L> AsArray() const
        {
            if (!MatchesType<T>())
                throw pybind11::type_error("native Array dtype mismatch (no implicit conversion)");
            return Array<T, L>::FromSharedStorage(_storage);
        }
        ArrayStorage::Shared SharedStorage() const { return ArrayStorage::Share(_storage); }
        ExclusiveRows EditRows() { return ExclusiveRows(_storage); }
        bool IsValid() const { return _storage->IsValid(); }
        uint64_t ShapeVersion() const { return _storage->ShapeVersion(); }
        RowLayout Layout() const { return _storage->Layout(); }
        const pybind11::dtype &Dtype() const { return _dtype; }
        const PythonElementShared &Element() const { return _element; }
        const pybind11::object &ElementType() const { return _elementType; }
        bool SameElement(const PythonArray &other) const
        {
            return _dtype.equal(other._dtype) && bool(_element) == bool(other._element) &&
                   (!_element || (_element->type == other._element->type && _element->schema == other._element->schema));
        }
        PythonArray New(RowLayout layout, Device device) const
        {
            auto result = PythonArray(std::move(layout), _dtype, device);
            result._element = _element;
            result._elementType = _elementType;
            return result;
        }
        PythonArray WithStorage(ArrayStorage::Shared storage) const
        {
            PythonArray result(std::move(storage), _dtype);
            result._element = _element;
            result._elementType = _elementType;
            if (_element)
                DNDS_check_throw_info(reinterpret_cast<uintptr_t>(result.Storage().LegacyPointer()) % _element->alignment == 0,
                                      "Array POD alignment mismatch");
            return result;
        }
        Buffer Storage() const { return _storage->Values(); }
        ExecutionContext Context(uintptr_t stream = 0) const { return {Storage().GetDevice(), reinterpret_cast<void *>(stream)}; }
        ReadAccess Read(ExecutionContext context) const { return _storage->Read(context); }
        WriteAccess Write(ExecutionContext context) { return _storage->Write(context); }
        CopyOperation CopyFrom(const PythonArray &source, ExecutionContext context)
        {
            DNDS_check_throw_info(SameElement(source), "Array copy dtype mismatch");
            return _storage->CopyFrom(*source._storage, context);
        }
    };
}

namespace pybind11::detail
{
    /// Converts shallow handles, never values or row metadata. Array Reset on a
    /// converted argument rebinds that handle only, just like a C++ Array copy.
    /// Pass Arrays by value for element updates; use explicit setters to replace
    /// a solver-owned Array. The shared allocation outlives either language owner.
    template <class T, class L>
    struct type_caster<DNDS::Runtime::Array<T, L>>
    {
        using Native = DNDS::Runtime::Array<T, L>;
        PYBIND11_TYPE_CASTER(Native, const_name("DNDSR.DNDS.Runtime.Array"));

        bool load(handle source, bool)
        {
            using Facade = DNDS::Runtime::PythonArray;
            if (!isinstance<Facade>(source))
                return false;
            const auto &array = pybind11::cast<const Facade &>(source);
            if (!array.template MatchesType<T>())
                return false;
            value = array.template AsArray<T, L>();
            return true;
        }
        static handle cast(const Native &array, return_value_policy, handle)
        {
            return pybind11::cast(DNDS::Runtime::PythonArray::Share(array)).release();
        }
    };
}
