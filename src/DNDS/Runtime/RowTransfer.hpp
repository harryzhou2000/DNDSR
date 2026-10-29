#pragma once

#include "Array.hpp"
#include "ElementSchema.hpp"
#include "Version.hpp"
#include "DNDS/MPI.hpp"
#include "CommunicationContext.hpp"
#include <functional>
#include <string>
#include <cstring>
#include <cmath>
#include <typeinfo>
#include <optional>
#include <utility>

namespace DNDS::Runtime
{
    /// One incoming relation edge. Destination rows need not be consecutive.
    struct RowRequest
    {
        int sourceRank = 0;
        index sourceRow = 0;
        index destinationRow = 0;
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
        RowTransferPlan(MPI_Comm comm, index sourceRows, index destinationRows,
                        const std::vector<RowRequest> &requests, std::vector<VersionStamp> dependencies = {});
        RowTransferPlan(std::shared_ptr<CommunicationContext> context, index sourceRows, index destinationRows,
                        const std::vector<RowRequest> &requests, std::vector<VersionStamp> dependencies = {});
        /// Peer row lists in matching per-peer order bypass owner/row discovery.
        /// Counts, bounds and collisions are checked collectively.
        /// Caller guarantees that corresponding send/receive entries mean the
        /// same row; equal counts cannot detect a semantically wrong permutation.
        static RowTransferPlan Known(std::shared_ptr<CommunicationContext> context, index sourceRows, index destinationRows,
                                     std::vector<std::vector<index>> send, std::vector<std::vector<index>> receive,
                                     std::vector<VersionStamp> dependencies = {});
        RowTransferPlan Reverse() const;
        /// Collective schedule derivation for a complete source permutation.
        /// Every source row must occur exactly once across the send lists.
        /// Returns {adjusted route, old-to-new source row permutation}; caller
        /// applies the permutation to its ordinary arrays before binding.
        /// Peer correspondence and receive slots are unchanged. No payload,
        /// row-ID/count exchange or communicator duplication is performed.
        std::pair<RowTransferPlan, std::vector<index>> ReorderSourceByPeer() const;
        MPI_Comm Communicator() const;
        /// Shared matching namespace for sequential payload bindings. Concurrent
        /// bindings must reserve distinct explicit tags or use separate contexts.
        std::shared_ptr<CommunicationContext> Context() const;
        index SourceRows() const;
        index DestinationRows() const;
        const std::vector<std::vector<index>> &SendRows() const;
        const std::vector<std::vector<index>> &ReceiveRows() const;
        bool HasCollisions() const;
        bool Valid() const;
        /// Setup exchange only: values are not moved. Unrequested rows have width 0.
        RowLayout DestinationLayout(const RowLayout &source) const;
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
        PayloadPlan(RowTransferPlan route, RowLayout source, std::optional<RowLayout> destination, ElementSchema schema);
        bool LayoutsMatch(const RowLayout &source, const RowLayout &destination, size_t elementBytes, bool elementCompatible) const;
        template <class T>
        bool CaptureBuffers(const Array<T> &source, const Array<T> &destination, Buffer &input, Buffer &output) const
        {
            try
            {
                // Pin the published shapes while checking and obtaining their
                // value buffers. An unpublished/failed shape on one rank must
                // reach binding's collective rejection, not strand its peers.
                const auto sourceStorage = source.SharedStorage(), destinationStorage = destination.SharedStorage();
                DNDS_check_throw_info(sourceStorage && destinationStorage, "binding requires live Array handles");
                const bool valid = LayoutsMatch(sourceStorage->Layout(), destinationStorage->Layout(), sizeof(T),
                                                Compatible(ElementSchema::TypeIdentity<T>()));
                input = sourceStorage->Values();
                output = destinationStorage->Values();
                return valid;
            }
            catch (...)
            {
                return false;
            }
        }

    public:
        PayloadPlan(RowTransferPlan route, RowLayout source, RowLayout destination, ElementSchema schema);
        /// Infer destination rows from one owner-width exchange, compiling the
        /// payload in the same operation. Avoids a second identical shape exchange.
        PayloadPlan(RowTransferPlan route, RowLayout source, ElementSchema schema);
        RowLayout DestinationLayout() const;
        void CheckLayouts(const RowLayout &source, const RowLayout &destination, size_t elementBytes, bool elementCompatible = true) const;
        bool Compatible(const std::string &type) const;
        size_t SendSegments() const;
        size_t ReceiveSegments() const;
        TransferBinding Bind(Buffer source, Buffer destination, TransferTransport transport = TransferTransport::Direct, TransferOptions options = {}) const;
        TransferBinding Bind(Buffer source, Buffer destination, Reducer reducer, TransferTransport transport = TransferTransport::Direct, TransferOptions options = {}) const;
        template <class T>
        TransferBinding Bind(const Array<T> &source, const Array<T> &destination, TransferTransport transport = TransferTransport::Direct, TransferOptions options = {}) const;
        template <class T>
        TransferBinding Bind(const Array<T> &source, const Array<T> &destination, Reducer reducer, TransferTransport transport = TransferTransport::Direct, TransferOptions options = {}) const;
    };

    class TransferBinding
    {
        struct State;
        std::shared_ptr<State> _state;
        friend class PayloadPlan;
        friend class TransferOperation;
        TransferBinding(std::shared_ptr<const PayloadPlan::State> plan, Buffer source, Buffer destination,
                        std::shared_ptr<const Reducer> reducer, TransferTransport transport, TransferOptions options,
                        bool layoutsValid = true);
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

    template <class T>
    TransferBinding PayloadPlan::Bind(const Array<T> &source, const Array<T> &destination, TransferTransport transport, TransferOptions options) const
    {
        Buffer input, output;
        const bool valid = CaptureBuffers(source, destination, input, output);
        return TransferBinding(_state, std::move(input), std::move(output), {}, transport, std::move(options), valid);
    }
    template <class T>
    TransferBinding PayloadPlan::Bind(const Array<T> &source, const Array<T> &destination, Reducer reducer, TransferTransport transport, TransferOptions options) const
    {
        Buffer input, output;
        const bool valid = CaptureBuffers(source, destination, input, output);
        return TransferBinding(_state, std::move(input), std::move(output), std::make_shared<const Reducer>(std::move(reducer)),
                               transport, std::move(options), valid);
    }
}
