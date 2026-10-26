#include "ElementSchema.hpp"
#include <algorithm>
#include <climits>
#include <limits>
#include <locale>
#include <map>
#include <mutex>
#include <set>
#include <sstream>

namespace DNDS::Runtime
{
    namespace
    {
        bool MPIAlive()
        {
            int initialized = 0, finalized = 0;
            MPI_Initialized(&initialized);
            if (initialized)
                MPI_Finalized(&finalized);
            return initialized && !finalized;
        }
        MPI_Datatype ScalarType(ElementScalar scalar)
        {
            switch (scalar)
            {
            case ElementScalar::Byte:
                return MPI_BYTE;
            case ElementScalar::I8:
                return MPI_INT8_T;
            case ElementScalar::U8:
                return MPI_UINT8_T;
            case ElementScalar::I16:
                return MPI_INT16_T;
            case ElementScalar::U16:
                return MPI_UINT16_T;
            case ElementScalar::I32:
                return MPI_INT32_T;
            case ElementScalar::U32:
                return MPI_UINT32_T;
            case ElementScalar::I64:
                return MPI_INT64_T;
            case ElementScalar::U64:
                return MPI_UINT64_T;
            case ElementScalar::F32:
                return MPI_FLOAT;
            case ElementScalar::F64:
                return MPI_DOUBLE;
            }
            DNDS_check_throw_info(false, "unknown element scalar");
            return MPI_DATATYPE_NULL;
        }
        bool NamedScalar(MPI_Datatype type, ElementScalar &scalar)
        {
            if (type == MPI_BYTE)
                scalar = ElementScalar::Byte;
            else if (type == MPI_INT8_T || type == MPI_SIGNED_CHAR)
                scalar = ElementScalar::I8;
            else if (type == MPI_UINT8_T || type == MPI_UNSIGNED_CHAR || type == MPI_CXX_BOOL)
                scalar = ElementScalar::U8;
            else if (type == MPI_CHAR)
                scalar = std::is_signed_v<char> ? ElementScalar::I8 : ElementScalar::U8;
            else if (type == MPI_INT16_T || type == MPI_SHORT)
                scalar = ElementScalar::I16;
            else if (type == MPI_UINT16_T || type == MPI_UNSIGNED_SHORT)
                scalar = ElementScalar::U16;
            else if (type == MPI_INT32_T || type == MPI_INT)
                scalar = ElementScalar::I32;
            else if (type == MPI_UINT32_T || type == MPI_UNSIGNED)
                scalar = ElementScalar::U32;
            else if (type == MPI_INT64_T || type == MPI_LONG_LONG_INT)
                scalar = ElementScalar::I64;
            else if (type == MPI_UINT64_T || type == MPI_UNSIGNED_LONG_LONG)
                scalar = ElementScalar::U64;
            else if (type == MPI_LONG)
                scalar = sizeof(long) == 8 ? ElementScalar::I64 : ElementScalar::I32;
            else if (type == MPI_UNSIGNED_LONG)
                scalar = sizeof(unsigned long) == 8 ? ElementScalar::U64 : ElementScalar::U32;
            else if (type == MPI_FLOAT)
                scalar = ElementScalar::F32;
            else if (type == MPI_DOUBLE)
                scalar = ElementScalar::F64;
            else
                return false;
            int bytes = 0;
            MPI_Type_size(type, &bytes);
            return bytes >= 0 && size_t(bytes) == ElementSchema::ScalarBytes(scalar);
        }

        // Flatten the common explicit POD constructors. Unknown MPI combiners
        // remain native-only instead of inventing one sizeof(POD)-wide word.
        bool DescribeMPI(MPI_Datatype type, MPI_Aint offset, std::vector<ElementField> &fields, int depth = 0)
        {
            if (depth > 32 || offset < 0)
                return false;
            int ni = 0, na = 0, nd = 0, combiner = 0;
            MPI_Type_get_envelope(type, &ni, &na, &nd, &combiner);
            if (combiner == MPI_COMBINER_NAMED)
            {
                ElementScalar scalar;
                if (!NamedScalar(type, scalar))
                    return false;
                fields.push_back({"word" + std::to_string(fields.size()), size_t(offset), scalar});
                return true;
            }
            std::vector<int> integers(size_t(ni), 0);
            std::vector<MPI_Aint> addresses(size_t(na), 0);
            std::vector<MPI_Datatype> types(size_t(nd), MPI_DATATYPE_NULL);
            MPI_Type_get_contents(type, ni, na, nd, integers.data(), addresses.data(), types.data());
            bool valid = true;
            if (combiner == MPI_COMBINER_DUP || combiner == MPI_COMBINER_RESIZED)
                valid = nd == 1 && DescribeMPI(types[0], offset, fields, depth + 1);
            else if (combiner == MPI_COMBINER_CONTIGUOUS && nd == 1)
            {
                MPI_Aint lower = 0, extent = 0;
                MPI_Type_get_extent(types[0], &lower, &extent);
                valid = ni == 1 && integers[0] >= 0 && extent > 0;
                for (int i = 0; valid && i < integers[0]; ++i)
                {
                    valid = i <= (std::numeric_limits<MPI_Aint>::max() - offset) / extent;
                    if (valid)
                        valid = DescribeMPI(types[0], offset + MPI_Aint(i) * extent, fields, depth + 1);
                }
            }
            else if (combiner == MPI_COMBINER_STRUCT && ni >= 1 && nd == integers[0] && na == nd)
            {
                for (int i = 0; valid && i < nd; ++i)
                {
                    MPI_Aint lower = 0, extent = 0;
                    MPI_Type_get_extent(types[size_t(i)], &lower, &extent);
                    valid = integers[size_t(i) + 1] >= 0 && addresses[size_t(i)] >= 0 &&
                            addresses[size_t(i)] <= std::numeric_limits<MPI_Aint>::max() - offset && extent > 0;
                    const auto first = valid ? offset + addresses[size_t(i)] : 0;
                    for (int j = 0; valid && j < integers[size_t(i) + 1]; ++j)
                    {
                        valid = j <= (std::numeric_limits<MPI_Aint>::max() - first) / extent;
                        if (valid)
                            valid = DescribeMPI(types[size_t(i)], first + MPI_Aint(j) * extent, fields, depth + 1);
                    }
                }
            }
            else
                valid = false;
            // MPI returns retained handles for derived contents. Named handles
            // need no release and must never be passed to MPI_Type_free.
            for (auto &nested : types)
            {
                int a = 0, b = 0, c = 0, kind = 0;
                MPI_Type_get_envelope(nested, &a, &b, &c, &kind);
                if (kind != MPI_COMBINER_NAMED)
                    MPI_Type_free(&nested);
            }
            return valid;
        }
        struct Registry
        {
            std::mutex mutex;
            std::map<std::type_index, std::shared_ptr<const ElementSchema>> native;
            std::map<std::string, std::shared_ptr<const ElementSchema>> identities;
        };
        Registry &Schemas()
        {
            static Registry result;
            return result;
        }
    }

    struct ElementSchema::State
    {
        size_t extent = 0, alignment = 1;
        std::string identity, cppName, definition;
        std::vector<ElementField> fields;
        bool opaque = false, nativeLongDouble = false;
        mutable std::mutex mutex;
        mutable MPI_Datatype type = MPI_DATATYPE_NULL;
        ~State()
        {
            if (type != MPI_DATATYPE_NULL && MPIAlive())
                MPI_Type_free(&type);
        }
    };

    size_t ElementSchema::ScalarBytes(ElementScalar scalar)
    {
        switch (scalar)
        {
        case ElementScalar::Byte:
        case ElementScalar::I8:
        case ElementScalar::U8:
            return 1;
        case ElementScalar::I16:
        case ElementScalar::U16:
            return 2;
        case ElementScalar::I32:
        case ElementScalar::U32:
        case ElementScalar::F32:
            return 4;
        case ElementScalar::I64:
        case ElementScalar::U64:
        case ElementScalar::F64:
            return 8;
        }
        DNDS_check_throw_info(false, "unknown element scalar");
        return 0;
    }
    const char *ElementSchema::ScalarName(ElementScalar scalar)
    {
        switch (scalar)
        {
        case ElementScalar::Byte:
            return "byte";
        case ElementScalar::I8:
            return "int8";
        case ElementScalar::U8:
            return "uint8";
        case ElementScalar::I16:
            return "int16";
        case ElementScalar::U16:
            return "uint16";
        case ElementScalar::I32:
            return "int32";
        case ElementScalar::U32:
            return "uint32";
        case ElementScalar::I64:
            return "int64";
        case ElementScalar::U64:
            return "uint64";
        case ElementScalar::F32:
            return "float32";
        case ElementScalar::F64:
            return "float64";
        }
        DNDS_check_throw_info(false, "unknown element scalar");
        return "";
    }

    ElementSchema ElementSchema::Record(std::string identity, size_t objectBytes,
                                        std::vector<ElementField> fields, std::string cppName, size_t alignment)
    {
        DNDS_check_throw_info(objectBytes && !identity.empty() && alignment && !(alignment & (alignment - 1)) &&
                                  objectBytes % alignment == 0 && !fields.empty(),
                              "invalid record identity, size, alignment or empty field layout");
        std::sort(fields.begin(), fields.end(), [](const auto &a, const auto &b)
                  { return a.offset < b.offset || (a.offset == b.offset && a.name < b.name); });
        size_t previous = 0;
        std::set<std::string> names;
        std::ostringstream definition;
        definition.imbue(std::locale::classic());
        definition << "record.v1;bytes=" << objectBytes << ";alignment=" << alignment << ";fields=" << fields.size();
        for (const auto &field : fields)
        {
            const auto bytes = ScalarBytes(field.scalar);
            DNDS_check_throw_info(!field.name.empty() && names.insert(field.name).second && field.count &&
                                      field.offset >= previous && field.offset <= objectBytes &&
                                      field.count <= (objectBytes - field.offset) / bytes,
                                  "record fields overlap, exceed storage, or have invalid names/counts");
            previous = field.offset + field.count * bytes;
            definition << ";" << field.name.size() << ":" << field.name << "@" << field.offset << ":" << ScalarName(field.scalar) << "[" << field.count << "]";
        }
        auto state = std::make_shared<State>();
        state->extent = objectBytes;
        state->alignment = alignment;
        state->identity = std::move(identity);
        state->cppName = std::move(cppName);
        state->fields = std::move(fields);
        state->definition = definition.str();
        return ElementSchema(std::move(state));
    }

    ElementSchema::ElementSchema(MPI_Datatype type, size_t objectBytes, std::string identity,
                                 std::string valueType, std::string representationType)
        : _valueType(std::move(valueType)), _representationType(std::move(representationType))
    {
        DNDS_check_throw_info(MPIAlive() && type != MPI_DATATYPE_NULL && objectBytes && !identity.empty(), "invalid element schema or MPI is inactive");
        MPI_Aint lower = 0, extent = 0, trueLower = 0, trueExtent = 0;
        MPI_Type_get_extent(type, &lower, &extent);
        MPI_Type_get_true_extent(type, &trueLower, &trueExtent);
        DNDS_check_throw_info(lower == 0 && extent > 0 && uint64_t(extent) == objectBytes && trueLower >= 0 &&
                                  trueExtent >= 0 && trueLower <= extent && trueExtent <= extent - trueLower,
                              "MPI typemap exceeds element storage or extent differs from stride");
        std::vector<ElementField> fields;
        std::shared_ptr<State> state;
        if (DescribeMPI(type, 0, fields) && !fields.empty())
        {
            auto described = Record(identity, objectBytes, std::move(fields));
            state = std::const_pointer_cast<State>(described._state);
        }
        else
        {
            state = std::make_shared<State>();
            state->extent = objectBytes;
            state->identity = std::move(identity);
            state->opaque = true;
            state->definition = "native-opaque.v1;bytes=" + std::to_string(objectBytes);
        }
        state->cppName = _valueType;
        DNDS_check_throw_info(MPI_Type_dup(type, &state->type) == MPI_SUCCESS, "cannot retain element MPI datatype");
        _state = std::move(state);
    }

    ElementSchema ElementSchema::NativeLongDouble()
    {
        auto state = std::make_shared<State>();
        state->extent = sizeof(long double);
        state->alignment = alignof(long double);
        state->identity = TypeIdentity<long double>();
        state->cppName = typeid(long double).name();
        state->opaque = state->nativeLongDouble = true;
        state->definition = "native-long-double.v1;bytes=" + std::to_string(sizeof(long double)) +
                            ";alignment=" + std::to_string(alignof(long double));
        return ElementSchema(std::move(state));
    }

    MPI_Datatype ElementSchema::Type() const
    {
        DNDS_check_throw_info(MPIAlive(), "element MPI datatype requires initialized MPI");
        std::lock_guard<std::mutex> lock(_state->mutex);
        if (_state->type == MPI_DATATYPE_NULL)
        {
            if (_state->nativeLongDouble)
            {
                DNDS_check_throw_info(MPI_Type_dup(MPI_LONG_DOUBLE, &_state->type) == MPI_SUCCESS, "cannot retain native long double MPI type");
                return _state->type;
            }
            DNDS_check_throw_info(_state->extent <= size_t(std::numeric_limits<MPI_Aint>::max()), "element MPI extent overflow");
            std::vector<int> counts;
            std::vector<MPI_Aint> offsets;
            std::vector<MPI_Datatype> types;
            for (const auto &field : Fields())
            {
                size_t first = 0;
                while (first < field.count)
                {
                    const auto count = std::min(field.count - first, size_t(INT_MAX));
                    DNDS_check_throw_info(counts.size() < size_t(INT_MAX), "too many element MPI fields");
                    counts.push_back(int(count));
                    offsets.push_back(MPI_Aint(field.offset + first * ScalarBytes(field.scalar)));
                    types.push_back(ScalarType(field.scalar));
                    first += count;
                }
            }
            MPI_Datatype raw = MPI_DATATYPE_NULL, resized = MPI_DATATYPE_NULL;
            DNDS_check_throw_info(MPI_Type_create_struct(int(counts.size()), counts.data(), offsets.data(), types.data(), &raw) == MPI_SUCCESS,
                                  "cannot create element MPI datatype");
            const auto resizeError = MPI_Type_create_resized(raw, 0, MPI_Aint(Extent()), &resized);
            MPI_Type_free(&raw);
            DNDS_check_throw_info(resizeError == MPI_SUCCESS, "cannot resize element MPI datatype");
            const auto commitError = MPI_Type_commit(&resized);
            if (commitError != MPI_SUCCESS)
                MPI_Type_free(&resized);
            DNDS_check_throw_info(commitError == MPI_SUCCESS, "cannot commit element MPI datatype");
            _state->type = resized;
        }
        return _state->type;
    }
    size_t ElementSchema::Extent() const { return _state->extent; }
    size_t ElementSchema::Alignment() const { return _state->alignment; }
    const std::string &ElementSchema::Identity() const { return _state->identity; }
    const std::string &ElementSchema::CppTypeName() const { return _state->cppName; }
    const std::vector<ElementField> &ElementSchema::Fields() const { return _state->fields; }
    const std::string &ElementSchema::Definition() const { return _state->definition; }
    bool ElementSchema::CanSwapWords() const { return !_state->opaque; }
    void ElementSchema::SwapWords(void *base, size_t elementCount) const
    {
        DNDS_check_throw_info(!_state->opaque, "native opaque MPI element schema cannot convert byte order");
        DNDS_check_throw_info((base || !elementCount) && elementCount <= SIZE_MAX / Extent(), "invalid element byte-swap range");
        auto *bytes = static_cast<unsigned char *>(base);
        for (size_t element = 0; element < elementCount; ++element)
            for (const auto &field : Fields())
            {
                const auto width = ScalarBytes(field.scalar);
                for (size_t item = 0; item < field.count; ++item)
                {
                    auto *word = bytes + element * Extent() + field.offset + item * width;
                    std::reverse(word, word + width);
                }
            }
    }

    std::shared_ptr<const ElementSchema> ElementSchema::Find(const std::string &identity)
    {
        auto &registry = Schemas();
        std::lock_guard<std::mutex> lock(registry.mutex);
        const auto found = registry.identities.find(identity);
        return found == registry.identities.end() ? nullptr : found->second;
    }

    std::shared_ptr<const ElementSchema> ElementSchema::LookupNative(std::type_index type)
    {
        auto &registry = Schemas();
        std::lock_guard<std::mutex> lock(registry.mutex);
        const auto found = registry.native.find(type);
        return found == registry.native.end() ? nullptr : found->second;
    }
    ElementSchema ElementSchema::RegisterNative(std::type_index type, const std::string &cppName,
                                                const std::string &binding, ElementSchema schema,
                                                size_t bytes, size_t alignment)
    {
        DNDS_check_throw_info(schema.Extent() == bytes && schema.Alignment() <= alignment &&
                                  (!schema._state->opaque || schema._state->nativeLongDouble),
                              "registered native element size/alignment differs or schema is opaque");
        if (!schema._state->opaque && (schema.Alignment() != alignment || schema.CppTypeName().empty()))
        {
            auto described = Record(schema.Identity(), schema.Extent(), schema.Fields(),
                                    schema.CppTypeName().empty() ? cppName : schema.CppTypeName(), alignment);
            schema._state = std::move(described._state);
        }
        auto &registry = Schemas();
        std::lock_guard<std::mutex> lock(registry.mutex);
        const auto found = registry.native.find(type);
        if (found != registry.native.end())
        {
            DNDS_check_throw_info(found->second->Identity() == schema.Identity() && found->second->Definition() == schema.Definition(),
                                  "native element type already has an incompatible schema");
            return *found->second;
        }
        const auto identity = registry.identities.find(schema.Identity());
        if (identity != registry.identities.end())
        {
            DNDS_check_throw_info(identity->second->Definition() == schema.Definition(), "element schema identity already has an incompatible definition");
            schema._state = identity->second->_state;
        }
        // Python-generated records initially bind to their logical dtype name.
        // Explicit native registration retains that definition while attaching
        // the C++ type tag used by typed Array bindings.
        DNDS_check_throw_info(schema._valueType.empty() || schema._valueType == binding || schema._valueType == schema.Identity(),
                              "element schema native binding type differs");
        schema._valueType = binding;
        auto retained = std::make_shared<const ElementSchema>(schema);
        registry.native.emplace(type, retained);
        registry.identities.emplace(schema.Identity(), retained);
        return schema;
    }
}
