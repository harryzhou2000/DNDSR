#pragma once

#include "DNDS/MPI.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <typeindex>
#include <type_traits>
#include <utility>
#include <vector>

namespace DNDS::Runtime
{
    enum class ElementScalar
    {
        Byte,
        I8,
        U8,
        I16,
        U16,
        I32,
        U32,
        I64,
        U64,
        F32,
        F64
    };

    struct ElementField
    {
        std::string name;
        size_t offset;
        ElementScalar scalar;
        size_t count = 1;
    };

    template <class T>
    struct ElementSchemaProvider;

    /// Immutable element meaning and scalar-word layout. Record descriptions
    /// and registration are usable before MPI initialization; Type() prepares
    /// the retained MPI typemap only when communication first needs it.
    class ElementSchema
    {
        struct State;
        std::shared_ptr<const State> _state;
        std::string _valueType, _representationType;
        explicit ElementSchema(std::shared_ptr<const State> state) : _state(std::move(state)) {}
        static ElementSchema RegisterNative(std::type_index type, const std::string &cppName,
                                            const std::string &binding, ElementSchema schema,
                                            size_t bytes, size_t alignment);
        static std::shared_ptr<const ElementSchema> LookupNative(std::type_index type);
        static ElementSchema NativeLongDouble();
        template <class T>
        static ElementScalar ArithmeticScalar()
        {
            if constexpr (std::is_floating_point_v<T>)
            {
                static_assert(sizeof(T) == 4 || sizeof(T) == 8, "unsupported floating element representation");
                return sizeof(T) == 4 ? ElementScalar::F32 : ElementScalar::F64;
            }
            else
            {
                static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8,
                              "unsupported integer element representation");
                if constexpr (std::is_signed_v<T>)
                    return sizeof(T) == 1 ? ElementScalar::I8 : sizeof(T) == 2 ? ElementScalar::I16
                                                            : sizeof(T) == 4   ? ElementScalar::I32
                                                                               : ElementScalar::I64;
                else
                    return sizeof(T) == 1 ? ElementScalar::U8 : sizeof(T) == 2 ? ElementScalar::U16
                                                            : sizeof(T) == 4   ? ElementScalar::U32
                                                                               : ElementScalar::U64;
            }
        }

    public:
        /// Legacy MPI descriptors retain their input immediately: the caller
        /// may free it after construction. This overload requires live MPI.
        ElementSchema(MPI_Datatype type, size_t objectBytes, std::string identity,
                      std::string valueType = {}, std::string representationType = {});
        static ElementSchema Record(std::string identity, size_t objectBytes,
                                    std::vector<ElementField> fields, std::string cppName = {}, size_t alignment = 1);
        MPI_Datatype Type() const;
        size_t Extent() const;
        size_t Alignment() const;
        const std::string &Identity() const;
        const std::string &CppTypeName() const;
        const std::string &BindingIdentity() const { return _valueType; }
        const std::vector<ElementField> &Fields() const;
        /// Canonical structure, independent of host endianness and C++ RTTI.
        const std::string &Definition() const;
        bool CanSwapWords() const;
        bool SharesDefinition(const ElementSchema &other) const { return _state == other._state; }
        static size_t ScalarBytes(ElementScalar scalar);
        static std::shared_ptr<const ElementSchema> Find(const std::string &identity);
        static const char *ScalarName(ElementScalar scalar);
        /// Reverse each described scalar word, leaving object padding alone.
        /// Opaque legacy MPI descriptors cannot provide portable byte swapping.
        void SwapWords(void *base, size_t elementCount) const;

        template <class T>
        static std::string TypeIdentity()
        {
            if constexpr (std::is_same_v<T, bool>)
                return "bool";
            if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>)
                return "float" + std::to_string(sizeof(T) * 8);
            if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t> || std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t> ||
                          std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> || std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>)
                return std::string(std::is_signed_v<T> ? "int" : "uint") + std::to_string(sizeof(T) * 8);
            return typeid(T).name();
        }
        bool Compatible(const std::string &type) const { return _valueType.empty() || _valueType == type; }
        bool CompatibleRepresentation(const std::string &type) const { return _representationType.empty() ? Compatible(type) : _representationType == type; }
        ElementSchema WithBinding(std::string type) const
        {
            auto result = *this;
            result._valueType = std::move(type);
            return result;
        }
        ElementSchema WithRepresentation(std::string type) const
        {
            auto result = *this;
            result._representationType = std::move(type);
            return result;
        }

        template <class T>
        static ElementSchema Register(const ElementSchema &schema)
        {
            static_assert(std::is_trivially_copyable_v<T>, "element schemas require trivially copyable native values");
            if constexpr (std::is_arithmetic_v<T>)
            {
                const auto expected = []
                {
                    if constexpr (std::is_same_v<T, long double>)
                        return NativeLongDouble();
                    else
                        return Record(TypeIdentity<T>(), sizeof(T), {{"value", 0, ArithmeticScalar<T>()}}, typeid(T).name(), alignof(T));
                }();
                DNDS_check_throw_info(schema.Identity() == expected.Identity() && schema.Definition() == expected.Definition(),
                                      "arithmetic element schema cannot redefine its native scalar representation");
            }
            return RegisterNative(typeid(T), typeid(T).name(), TypeIdentity<T>(), schema, sizeof(T), alignof(T));
        }
        template <class T>
        static ElementSchema For()
        {
            using Value = std::remove_cv_t<T>;
            if (auto registered = LookupNative(typeid(Value)))
                return *registered;
            if constexpr (std::is_arithmetic_v<Value>)
            {
                if constexpr (std::is_same_v<Value, long double>)
                    return Register<Value>(NativeLongDouble());
                else
                    return Register<Value>(Record(TypeIdentity<Value>(), sizeof(Value), {{"value", 0, ArithmeticScalar<Value>()}}, typeid(Value).name(), alignof(Value)));
            }
            else
                return Register<Value>(ElementSchemaProvider<Value>::Get());
        }
        template <class T>
        static ElementSchema Arithmetic()
        {
            static_assert(std::is_arithmetic_v<T>);
            return For<T>();
        }
    };

    template <class T>
    struct ElementSchemaProvider
    {
        static ElementSchema Get()
        {
            DNDS_check_throw_info(false, "native POD element type needs explicit ElementSchema registration or a provider");
            return ElementSchema::Record("unreachable", 1, {{"value", 0, ElementScalar::Byte}});
        }
    };
}
