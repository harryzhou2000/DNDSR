#pragma once

#include "Array.hpp"
#include "Version.hpp"
#include "DNDS/MPI.hpp"
#include "CommunicationContext.hpp"
#include <functional>
#include <string>
#include <cstring>
#include <cmath>
#include <typeinfo>

namespace DNDS::Runtime
{
    /// One incoming relation edge. Destination rows need not be consecutive.
    struct RowRequest
    {
        int sourceRank = 0;
        int64_t sourceRow = 0;
        size_t destinationRow = 0;
    };

    /// Collective immutable row relation. Caller serializes setup/Begin in the
    /// same order on every rank, supplies correct semantic row identities and
    /// stamps affected mesh epochs. Mutation requires draining old operations;
    /// a stamp is invalidation detection, not a lock held through execution.
    class RowTransferPlan
    {
        struct State;
        std::shared_ptr<const State> _state;
        bool _reverse = false;
        RowTransferPlan() = default;

    public:
        RowTransferPlan(MPI_Comm comm, size_t sourceRows, size_t destinationRows,
                        const std::vector<RowRequest> &requests, std::vector<VersionStamp> dependencies = {});
        RowTransferPlan(std::shared_ptr<CommunicationContext> context, size_t sourceRows, size_t destinationRows,
                        const std::vector<RowRequest> &requests, std::vector<VersionStamp> dependencies = {});
        /// Peer row lists in matching per-peer order bypass owner/row discovery.
        /// Counts, bounds and collisions are checked collectively.
        /// Caller guarantees that corresponding send/receive entries mean the
        /// same row; equal counts cannot detect a semantically wrong permutation.
        static RowTransferPlan Known(std::shared_ptr<CommunicationContext> context, size_t sourceRows, size_t destinationRows,
                                     std::vector<std::vector<size_t>> send, std::vector<std::vector<size_t>> receive,
                                     std::vector<VersionStamp> dependencies = {});
        RowTransferPlan Reverse() const;
        MPI_Comm Communicator() const;
        size_t SourceRows() const;
        size_t DestinationRows() const;
        const std::vector<std::vector<size_t>> &SendRows() const;
        const std::vector<std::vector<size_t>> &ReceiveRows() const;
        bool HasCollisions() const;
        bool Valid() const;
        /// Setup exchange only: values are not moved. Unrequested rows have width 0.
        RowLayout DestinationLayout(const RowLayout &source) const;
    };

    /// Retained MPI typemap plus application schema identity; extent is stride.
    /// Caller supplies a committed, nonoverlapping typemap within sizeof(POD)
    /// and an identity denoting the same signature/meaning on every rank. Explicit
    /// untagged schemas and raw Buffer bindings cannot infer the C++ object type.
    class ElementSchema
    {
        struct State;
        std::shared_ptr<const State> _state;
        std::string _valueType;
        std::string _representationType;

    public:
        ElementSchema(MPI_Datatype type, size_t objectBytes, std::string identity, std::string valueType = {}, std::string representationType = {});
        MPI_Datatype Type() const;
        size_t Extent() const;
        const std::string &Identity() const;
        /// Shared native/Python names for the supported arithmetic wire types.
        /// Other native types retain their local RTTI identity; custom schemas
        /// remain explicit application descriptions, not portable object codecs.
        template <class T>
        static std::string TypeIdentity()
        {
            if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>)
                return "float" + std::to_string(sizeof(T) * 8);
            if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t> || std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t> ||
                          std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> || std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>)
                return std::string(std::is_signed_v<T> ? "int" : "uint") + std::to_string(sizeof(T) * 8);
            return typeid(T).name();
        }
        /// Factories retain a binding type tag. Explicit POD MPI descriptors
        /// without a tag remain an application contract.
        bool Compatible(const std::string &type) const { return _valueType.empty() || _valueType == type; }
        /// A Python reducer receives NumPy records, even for a registered class.
        /// Representation compatibility never substitutes for binding type checks.
        bool CompatibleRepresentation(const std::string &type) const { return _representationType.empty() ? Compatible(type) : _representationType == type; }
        template <class T>
        static ElementSchema Arithmetic()
        {
            static_assert(std::is_arithmetic_v<T>);
            auto type = BasicType_To_MPIIntType<T>();
            DNDS_check_throw(type.second == 1);
            return {type.first, sizeof(T), TypeIdentity<T>(), TypeIdentity<T>()};
        }
    };

    enum class Reduction
    {
        Sum,
        Min,
        Max,
        BitOr,
        BitXor,
        BitAnd
    };
    enum class DestinationInit
    {
        Preserve,
        Identity
    };
    enum class TransferTransport
    {
        Direct,
        HostPacked
    };

    struct Contribution
    {
        size_t sourceElement = 0, destinationElement = 0, count = 0;
    };

    struct Reducer;
#ifdef DNDS_USE_CUDA
    void InstallBuiltinCUDA(Reducer &, Reduction, size_t bytes, bool integral, bool isSigned, DestinationInit);
#endif

    /// Dispatch once per operation. Custom CPU/CUDA/vendor implementations use
    /// the same contribution groups; no MPI or Array ownership in the callback.
    /// Caller gives identical combine/initialization semantics to a shared name
    /// on every rank. Do not read untransmitted POD fields; do not retain pointers
    /// or issue unmatched MPI collectives in a combine. Device combines must use
    /// the supplied stream. Floating reductions are not partition-bitwise stable.
    struct Reducer
    {
        std::string identity;
        std::string schema;
        size_t elementBytes = 0;
        DestinationInit initialization = DestinationInit::Preserve;
        std::function<void(const void *, void *, const std::vector<Contribution> &, ExecutionContext)> host;
        std::function<void(const void *, void *, const Contribution *, size_t, ExecutionContext)> device;

        template <class T, class Combine>
        static Reducer Custom(const ElementSchema &element, std::string name, Combine combine, T identityValue,
                              DestinationInit init = DestinationInit::Preserve)
        {
            DNDS_check_throw_info(element.Extent() == sizeof(T) && element.Compatible(ElementSchema::TypeIdentity<T>()), "reducer element type/stride mismatch");
            Reducer result;
            result.identity = std::move(name);
            result.schema = element.Identity();
            result.elementBytes = sizeof(T);
            result.initialization = init;
            result.host = [combine, identityValue, init](const void *input, void *output,
                                                         const std::vector<Contribution> &groups, ExecutionContext)
            {
                auto *src = static_cast<const T *>(input);
                auto *dst = static_cast<T *>(output);
                // Groups are stably ordered by destination then contributor.
                size_t previous = SIZE_MAX;
                for (const auto &group : groups)
                {
                    if (init == DestinationInit::Identity && group.destinationElement != previous)
                        std::fill_n(dst + group.destinationElement, group.count, identityValue);
                    for (size_t j = 0; j < group.count; ++j)
                        combine(dst[group.destinationElement + j], src[group.sourceElement + j]);
                    previous = group.destinationElement;
                }
            };
            return result;
        }

        template <class T>
        static Reducer Builtin(Reduction op, DestinationInit init = DestinationInit::Preserve)
        {
            static_assert(std::is_arithmetic_v<T> && !std::is_same_v<T, bool>);
            DNDS_check_throw_info(std::is_integral_v<T> || op <= Reduction::Max, "bitwise reduction requires integers");
            T identityValue{};
            if (op == Reduction::Min)
                identityValue = std::numeric_limits<T>::has_infinity ? std::numeric_limits<T>::infinity() : std::numeric_limits<T>::max();
            if (op == Reduction::Max)
                identityValue = std::numeric_limits<T>::has_infinity ? -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::lowest();
            if constexpr (std::is_integral_v<T>)
                if (op == Reduction::BitAnd)
                    identityValue = T(~T(0));
            auto result = Custom<T>(ElementSchema::Arithmetic<T>(), "builtin." + std::to_string(int(op)), [op](T &dst, const T &src)
                                    {
                                 switch (op)
                                 {
                                 case Reduction::Sum:
                                     if constexpr (std::is_integral_v<T>)
                                     {
                                         using U = std::make_unsigned_t<T>;
                                         U bits = U(dst) + U(src); // Defined modulo width, including signed payloads.
                                         std::memcpy(&dst, &bits, sizeof(T));
                                     }
                                     else dst += src;
                                     break;
                                 case Reduction::Min:
                                     if constexpr (std::is_floating_point_v<T>) dst = std::fmin(dst, src);
                                     else dst = std::min(dst, src);
                                     break;
                                 case Reduction::Max:
                                     if constexpr (std::is_floating_point_v<T>) dst = std::fmax(dst, src);
                                     else dst = std::max(dst, src);
                                     break;
                                 default:
                                     if constexpr (std::is_integral_v<T>)
                                     {
                                         if (op == Reduction::BitOr) dst |= src;
                                         if (op == Reduction::BitXor) dst ^= src;
                                         if (op == Reduction::BitAnd) dst &= src;
                                     }
                                 } }, identityValue, init);
#ifdef DNDS_USE_CUDA
            InstallBuiltinCUDA(result, op, sizeof(T), std::is_integral_v<T>, std::is_signed_v<T>, init);
#endif
            return result;
        }
    };

    class TransferBinding;
    class TransferOperation;

    class PayloadPlan
    {
        struct State;
        std::shared_ptr<const State> _state;
        friend class TransferBinding;
        friend class TransferOperation;

    public:
        PayloadPlan(RowTransferPlan route, RowLayout source, RowLayout destination, ElementSchema schema);
        void CheckLayouts(const RowLayout &source, const RowLayout &destination, size_t elementBytes, bool elementCompatible = true) const;
        bool Compatible(const std::string &type) const;
        size_t SendSegments() const;
        size_t ReceiveSegments() const;
        TransferBinding Bind(Buffer source, Buffer destination, TransferTransport transport = TransferTransport::Direct, TransferOptions options = {}) const;
        TransferBinding Bind(Buffer source, Buffer destination, Reducer reducer, TransferTransport transport = TransferTransport::Direct, TransferOptions options = {}) const;
        template <class T, class L, class M>
        TransferBinding Bind(const Array<T, L> &source, const Array<T, M> &destination, TransferTransport transport = TransferTransport::Direct, TransferOptions options = {}) const;
        template <class T, class L, class M>
        TransferBinding Bind(const Array<T, L> &source, const Array<T, M> &destination, Reducer reducer, TransferTransport transport = TransferTransport::Direct, TransferOptions options = {}) const;
    };

    class TransferBinding
    {
        struct State;
        std::shared_ptr<State> _state;
        friend class PayloadPlan;
        friend class TransferOperation;
        TransferBinding(std::shared_ptr<const PayloadPlan::State> plan, Buffer source, Buffer destination,
                        std::shared_ptr<const Reducer> reducer, TransferTransport transport, TransferOptions options);
        TransferOperation Start(ExecutionContext context, bool scheduled) const;

    public:
        /// One active operation per binding. Default remains isolated persistent
        /// requests. Options select shared context/tags and transient requests.
        /// Collective: all ranks call in the same order, including empty ranks.
        /// This binding retains old buffers after Array::Reset; rebuild it after
        /// changing storage/layout or semantic row order. Finish before mutation
        /// or MPI_Finalize, on a thread allowed by the initialized MPI level.
        TransferOperation Begin(ExecutionContext context = ExecutionContext::Host()) const;
        /// Caller-ordered eager reservations allow dependent scheduled kernels.
        /// CPU producers must already be complete; CUDA readiness is observed
        /// on the supplied stream before portable MPI is posted.
        /// Caller joins consumers explicitly and finishes CPU producers first;
        /// eager reservations permit conflicts but do not make races safe.
        TransferOperation BeginScheduled(ExecutionContext context = ExecutionContext::Host()) const;
    };

    class TransferOperation
    {
        struct State;
        std::unique_ptr<State> _state;
        friend class TransferBinding;
        explicit TransferOperation(std::unique_ptr<State> state);

    public:
        TransferOperation();
        TransferOperation(const TransferOperation &) = delete;
        TransferOperation &operator=(const TransferOperation &) = delete;
        TransferOperation(TransferOperation &&) noexcept;
        TransferOperation &operator=(TransferOperation &&) noexcept;
        ~TransferOperation();
        /// Failed completion remains failed across repeated observation and
        /// moves (including MPIProgress import). Destruction only drains once.
        /// Caller explicitly observes errors. WaitDestinationReady releases only
        /// receives; source send ownership lasts until Wait/Test completion.
        /// Raw failure is rank-local; coordinate solver recovery across ranks.
        void WaitDestinationReady();
        void Wait();
        bool Test();
    };

    template <class T, class L, class M>
    TransferBinding PayloadPlan::Bind(const Array<T, L> &source, const Array<T, M> &destination, TransferTransport transport, TransferOptions options) const
    {
        CheckLayouts(source.Layout(), destination.Layout(), sizeof(T), Compatible(ElementSchema::TypeIdentity<T>()));
        return Bind(source.Storage(), destination.Storage(), transport, std::move(options));
    }
    template <class T, class L, class M>
    TransferBinding PayloadPlan::Bind(const Array<T, L> &source, const Array<T, M> &destination, Reducer reducer, TransferTransport transport, TransferOptions options) const
    {
        CheckLayouts(source.Layout(), destination.Layout(), sizeof(T), Compatible(ElementSchema::TypeIdentity<T>()));
        return Bind(source.Storage(), destination.Storage(), std::move(reducer), transport, std::move(options));
    }
}
