#include "RowTransfer.hpp"
#include <climits>
#include <numeric>
#include <optional>
#include <set>
#include <mutex>

namespace DNDS::Runtime
{
    namespace
    {
        void CheckAll(bool valid, MPI_Comm comm, const char *message)
        {
            int all = valid;
            MPI_Allreduce(MPI_IN_PLACE, &all, 1, MPI_INT, MPI_MIN, comm);
            DNDS_check_throw_info(all, message);
        }
        void Agree(const std::string &value, MPI_Comm comm)
        {
            int rank = 0;
            MPI_Comm_rank(comm, &rank);
            CheckAll(value.size() <= INT_MAX, comm, "transfer schema identifier too long");
            int size = rank == 0 ? int(value.size()) : 0;
            MPI_Bcast(&size, 1, MPI_INT, 0, comm);
            std::string reference(size, '\0');
            if (rank == 0)
                reference = value;
            MPI_Bcast(reference.data(), size, MPI_CHAR, 0, comm);
            CheckAll(value == reference, comm, "transfer schema/policy differs between ranks");
        }
        bool MPIAlive()
        {
            int final = 0;
            MPI_Finalized(&final);
            return !final;
        }
        struct Datatype
        {
            MPI_Datatype value = MPI_DATATYPE_NULL;
            ~Datatype()
            {
                if (value != MPI_DATATYPE_NULL && MPIAlive())
                    MPI_Type_free(&value);
            }
        };
        using Type = std::shared_ptr<Datatype>;

        std::vector<int> Counts(const std::vector<std::vector<size_t>> &rows)
        {
            std::vector<int> counts;
            for (const auto &peer : rows)
                counts.push_back(int(peer.size()));
            return counts;
        }
        std::vector<int> Offsets(const std::vector<int> &counts)
        {
            std::vector<int> result(counts.size() + 1);
            for (size_t i = 0; i < counts.size(); ++i)
                result[i + 1] = result[i] + counts[i];
            return result;
        }
        std::vector<uint64_t> ExchangeLengths(const RowTransferPlan &route, const RowLayout &source)
        {
            CheckAll(source.Rows() == route.SourceRows() && route.Valid(), route.Communicator(), "row transfer source layout or epoch mismatch");
            const auto sendCounts = Counts(route.SendRows()), receiveCounts = Counts(route.ReceiveRows());
            const auto sendOffsets = Offsets(sendCounts), receiveOffsets = Offsets(receiveCounts);
            std::vector<uint64_t> send, receive(receiveOffsets.back());
            for (const auto &peer : route.SendRows())
                for (auto row : peer)
                    send.push_back(source.RowSize(row));
            MPI_Alltoallv(send.data(), sendCounts.data(), sendOffsets.data(), MPI_UINT64_T,
                          receive.data(), receiveCounts.data(), receiveOffsets.data(), MPI_UINT64_T, route.Communicator());
            return receive;
        }
        Type MakeType(const std::vector<size_t> &rows, const RowLayout &layout, const ElementSchema &schema, size_t &segments)
        {
            std::vector<int> lengths;
            std::vector<MPI_Aint> displacements;
            for (auto row : rows)
            {
                auto remaining = layout.RowSize(row);
                size_t offset = layout.Offset(row);
                while (remaining)
                {
                    const auto n = int(std::min(remaining, size_t(INT_MAX)));
                    DNDS_check_throw_info(offset <= size_t(std::numeric_limits<MPI_Aint>::max()) / schema.Extent(), "MPI displacement overflow");
                    auto displacement = MPI_Aint(offset * schema.Extent());
                    if (!lengths.empty() && n <= INT_MAX - lengths.back() &&
                        displacement - displacements.back() == MPI_Aint(lengths.back()) * MPI_Aint(schema.Extent()))
                        lengths.back() += n;
                    else
                    {
                        lengths.push_back(n);
                        displacements.push_back(displacement);
                    }
                    offset += n;
                    remaining -= n;
                }
            }
            DNDS_check_throw_info(lengths.size() <= INT_MAX, "MPI datatype has too many segments");
            segments += lengths.size();
            auto type = std::make_shared<Datatype>();
            MPI_Type_create_hindexed(int(lengths.size()), lengths.data(), displacements.data(), schema.Type(), &type->value);
            MPI_Type_commit(&type->value);
            return type;
        }
    }

    struct RowTransferPlan::State
    {
        MPI_Comm comm = MPI_COMM_NULL;
        size_t sources = 0, destinations = 0;
        std::vector<std::vector<size_t>> send, receive;
        bool collisions = false, reverseCollisions = false;
        std::vector<VersionStamp> dependencies;
        ~State()
        {
            if (comm != MPI_COMM_NULL && MPIAlive())
                MPI_Comm_free(&comm);
        }
    };

    RowTransferPlan::RowTransferPlan(MPI_Comm comm, size_t sourceRows, size_t destinationRows,
                                     const std::vector<RowRequest> &requests, std::vector<VersionStamp> dependencies)
    {
        DNDS_check_throw_info(comm != MPI_COMM_NULL, "row transfer needs a communicator");
        auto state = std::make_shared<State>();
        MPI_Comm_dup(comm, &state->comm);
        int size = 0;
        MPI_Comm_size(comm, &size);
        bool valid = sourceRows <= INT64_MAX && requests.size() <= INT_MAX;
        for (auto request : requests)
            valid = valid && request.sourceRank >= 0 && request.sourceRank < size && request.sourceRow >= 0 && request.destinationRow < destinationRows;
        CheckAll(valid, state->comm, "invalid row transfer request");
        state->sources = sourceRows;
        state->destinations = destinationRows;
        state->dependencies = std::move(dependencies);
        state->send.resize(size);
        state->receive.resize(size);
        std::vector<std::vector<int64_t>> requested(size);
        std::set<size_t> targets;
        for (auto request : requests)
        {
            requested[request.sourceRank].push_back(request.sourceRow);
            state->receive[request.sourceRank].push_back(request.destinationRow);
            state->collisions |= !targets.insert(request.destinationRow).second;
        }
        auto requestCounts = Counts(state->receive);
        std::vector<int> incomingCounts(size);
        MPI_Alltoall(requestCounts.data(), 1, MPI_INT, incomingCounts.data(), 1, MPI_INT, state->comm);
        auto incomingTotal = std::accumulate(incomingCounts.begin(), incomingCounts.end(), int64_t(0));
        CheckAll(incomingTotal <= INT_MAX, state->comm, "row transfer request table exceeds MPI int range");
        auto requestOffsets = Offsets(requestCounts), incomingOffsets = Offsets(incomingCounts);
        std::vector<int64_t> outgoing, incoming(size_t(incomingTotal), 0);
        for (const auto &peer : requested)
            outgoing.insert(outgoing.end(), peer.begin(), peer.end());
        MPI_Alltoallv(outgoing.data(), requestCounts.data(), requestOffsets.data(), MPI_INT64_T,
                      incoming.data(), incomingCounts.data(), incomingOffsets.data(), MPI_INT64_T, state->comm);
        targets.clear();
        for (int peer = 0; peer < size; ++peer)
            for (int j = incomingOffsets[peer]; j < incomingOffsets[peer + 1]; ++j)
            {
                valid = valid && incoming[j] >= 0 && uint64_t(incoming[j]) < sourceRows;
                state->send[peer].push_back(size_t(incoming[j]));
                state->reverseCollisions |= !targets.insert(size_t(incoming[j])).second;
            }
        CheckAll(valid, state->comm, "row transfer source index does not exist");
        _state = std::move(state);
    }
    RowTransferPlan RowTransferPlan::Reverse() const
    {
        auto result = *this;
        result._reverse = !result._reverse;
        return result;
    }
    MPI_Comm RowTransferPlan::Communicator() const { return _state->comm; }
    size_t RowTransferPlan::SourceRows() const { return _reverse ? _state->destinations : _state->sources; }
    size_t RowTransferPlan::DestinationRows() const { return _reverse ? _state->sources : _state->destinations; }
    const std::vector<std::vector<size_t>> &RowTransferPlan::SendRows() const { return _reverse ? _state->receive : _state->send; }
    const std::vector<std::vector<size_t>> &RowTransferPlan::ReceiveRows() const { return _reverse ? _state->send : _state->receive; }
    bool RowTransferPlan::HasCollisions() const { return _reverse ? _state->reverseCollisions : _state->collisions; }
    bool RowTransferPlan::Valid() const
    {
        for (const auto &dependency : _state->dependencies)
            if (!dependency.Valid())
                return false;
        return true;
    }
    RowLayout RowTransferPlan::DestinationLayout(const RowLayout &source) const
    {
        const auto received = ExchangeLengths(*this, source);
        std::vector<size_t> lengths(DestinationRows(), SIZE_MAX);
        size_t edge = 0;
        bool valid = true;
        for (const auto &peer : ReceiveRows())
            for (auto row : peer)
            {
                valid = valid && (lengths[row] == SIZE_MAX || lengths[row] == received[edge]);
                lengths[row] = received[edge++];
            }
        CheckAll(valid, Communicator(), "fan-in contributions have incompatible row widths");
        for (auto &n : lengths)
            if (n == SIZE_MAX)
                n = 0;
        return RowLayout::Ragged(lengths);
    }

    struct ElementSchema::State
    {
        std::shared_ptr<Datatype> type;
        size_t extent;
        std::string identity;
    };
    ElementSchema::ElementSchema(MPI_Datatype type, size_t objectBytes, std::string identity)
    {
        DNDS_check_throw_info(type != MPI_DATATYPE_NULL && objectBytes && !identity.empty(), "invalid element schema");
        MPI_Aint lb, extent, trueLB, trueExtent;
        MPI_Type_get_extent(type, &lb, &extent);
        MPI_Type_get_true_extent(type, &trueLB, &trueExtent);
        DNDS_check_throw_info(lb == 0 && extent > 0 && uint64_t(extent) == objectBytes && trueLB >= 0 &&
                                  trueExtent >= 0 && trueLB <= extent && trueExtent <= extent - trueLB,
                              "MPI typemap exceeds element storage or extent differs from stride");
        auto state = std::make_shared<State>();
        state->type = std::make_shared<Datatype>();
        MPI_Type_dup(type, &state->type->value);
        state->extent = objectBytes;
        state->identity = std::move(identity);
        _state = std::move(state);
    }
    MPI_Datatype ElementSchema::Type() const { return _state->type->value; }
    size_t ElementSchema::Extent() const { return _state->extent; }
    const std::string &ElementSchema::Identity() const { return _state->identity; }

    struct PayloadPlan::State
    {
        RowTransferPlan route;
        RowLayout source, destination, contributions;
        ElementSchema schema;
        std::vector<Type> sendTypes, receiveTypes, contributionTypes;
        std::vector<Contribution> groups;
        size_t sendSegments = 0, receiveSegments = 0;
        State(RowTransferPlan r, RowLayout s, RowLayout d, ElementSchema e)
            : route(std::move(r)), source(std::move(s)), destination(std::move(d)), schema(std::move(e)) {}
    };
    PayloadPlan::PayloadPlan(RowTransferPlan route, RowLayout source, RowLayout destination, ElementSchema schema)
    {
        auto state = std::make_shared<State>(std::move(route), std::move(source), std::move(destination), std::move(schema));
        const auto comm = state->route.Communicator();
        Agree(state->schema.Identity() + ":" + std::to_string(state->schema.Extent()), comm);
        CheckAll(state->destination.Rows() == state->route.DestinationRows(), comm, "row transfer destination shape mismatch");
        const auto lengths = ExchangeLengths(state->route, state->source);
        size_t edge = 0;
        bool valid = true;
        std::vector<size_t> widths;
        for (const auto &peer : state->route.ReceiveRows())
            for (auto row : peer)
            {
                valid = valid && state->destination.RowSize(row) == lengths[edge];
                widths.push_back(size_t(lengths[edge++]));
            }
        CheckAll(valid, comm, "row transfer edge width mismatch");
        state->contributions = RowLayout::Ragged(widths);
        edge = 0;
        try
        {
            for (size_t peer = 0; peer < state->route.SendRows().size(); ++peer)
            {
                state->sendTypes.push_back(MakeType(state->route.SendRows()[peer], state->source, state->schema, state->sendSegments));
                // Colliding final destinations are never used as receive types.
                if (!state->route.HasCollisions())
                    state->receiveTypes.push_back(MakeType(state->route.ReceiveRows()[peer], state->destination, state->schema, state->receiveSegments));
                std::vector<size_t> slots;
                for (auto row : state->route.ReceiveRows()[peer])
                {
                    slots.push_back(edge);
                    if (widths[edge])
                        state->groups.push_back({state->contributions.Offset(edge), state->destination.Offset(row), widths[edge]});
                    ++edge;
                }
                size_t segments = 0;
                state->contributionTypes.push_back(MakeType(slots, state->contributions, state->schema, segments));
            }
            std::stable_sort(state->groups.begin(), state->groups.end(), [](auto a, auto b)
                             { return a.destinationElement < b.destinationElement; });
        }
        catch (...)
        {
            valid = false;
        }
        CheckAll(valid, comm, "cannot compile row payload MPI layout");
        _state = std::move(state);
    }
    size_t PayloadPlan::SendSegments() const { return _state->sendSegments; }
    size_t PayloadPlan::ReceiveSegments() const { return _state->receiveSegments; }
    void PayloadPlan::CheckLayouts(const RowLayout &source, const RowLayout &destination, size_t bytes, bool elementCompatible) const
    {
        CheckAll(elementCompatible && source == _state->source && destination == _state->destination && bytes == _state->schema.Extent(),
                 _state->route.Communicator(), "Array layout differs from compiled payload");
    }

    struct TransferBinding::State
    {
        std::shared_ptr<const PayloadPlan::State> plan;
        Buffer source, destination, workspace, deviceGroups;
        Buffer hostSource, hostDestination;
        TransferTransport transport = TransferTransport::Direct;
        std::vector<std::vector<uint8_t>> packedSend, packedReceive;
        std::vector<int> packedSendSizes;
        std::shared_ptr<const Reducer> reducer;
        MPI_Comm comm = MPI_COMM_NULL;
        std::vector<MPI_Request> sends, receives;
        bool active = false;
        std::mutex mutex;
        ~State()
        {
            if (MPIAlive())
            {
                for (auto &r : sends)
                    MPI_Request_free(&r);
                for (auto &r : receives)
                    MPI_Request_free(&r);
                if (comm != MPI_COMM_NULL)
                    MPI_Comm_free(&comm);
            }
        }
    };
    TransferBinding PayloadPlan::Bind(Buffer source, Buffer destination, TransferTransport transport) const
    {
        return TransferBinding(_state, std::move(source), std::move(destination), {}, transport);
    }
    TransferBinding PayloadPlan::Bind(Buffer source, Buffer destination, Reducer reducer, TransferTransport transport) const
    {
        return TransferBinding(_state, std::move(source), std::move(destination), std::make_shared<const Reducer>(std::move(reducer)), transport);
    }
    TransferBinding::TransferBinding(std::shared_ptr<const PayloadPlan::State> plan, Buffer source, Buffer destination,
                                     std::shared_ptr<const Reducer> reducer, TransferTransport transport)
    {
        auto state = std::make_shared<State>();
        state->plan = std::move(plan);
        state->source = std::move(source);
        state->destination = std::move(destination);
        state->reducer = std::move(reducer);
        state->transport = transport;
        MPI_Comm_dup(state->plan->route.Communicator(), &state->comm);
        const auto &p = *state->plan;
        const auto policy = state->reducer ? state->reducer->identity + ":" + std::to_string(int(state->reducer->initialization)) : "copy";
        Agree(policy + ":" + std::to_string(int(transport)), state->comm);
        bool valid = state->source.Identity() && state->destination.Identity() && state->source.Identity() != state->destination.Identity();
        valid = valid && state->source.Bytes() / p.schema.Extent() == p.source.Elements() && state->source.Bytes() % p.schema.Extent() == 0;
        valid = valid && state->destination.Bytes() / p.schema.Extent() == p.destination.Elements() && state->destination.Bytes() % p.schema.Extent() == 0;
        valid = valid && (!p.route.HasCollisions() || state->reducer);
        const bool gpu = state->source.GetDevice().space == MemorySpace::CUDA || state->destination.GetDevice().space == MemorySpace::CUDA;
        valid = valid && (transport == TransferTransport::HostPacked || !gpu || MPI::isCudaAware());
        if (state->reducer)
            valid = valid && state->reducer->schema == p.schema.Identity() && state->reducer->elementBytes == p.schema.Extent() &&
                    (transport == TransferTransport::HostPacked || state->destination.GetDevice().space == MemorySpace::Host ? bool(state->reducer->host) : bool(state->reducer->device));
        CheckAll(valid, state->comm, "invalid transfer binding: collision, schema, placement, size or alias");
        try
        {
            if (state->reducer)
            {
                state->workspace = Buffer::Allocate(CheckedSize::Multiply(p.contributions.Elements(), p.schema.Extent()),
                                                    transport == TransferTransport::HostPacked ? Device::Host() : state->destination.GetDevice());
                if (transport == TransferTransport::Direct && state->destination.GetDevice().space == MemorySpace::CUDA)
                {
                    auto host = Buffer::Allocate(CheckedSize::Multiply(p.groups.size(), sizeof(Contribution)));
                    {
                        auto w = host.Write();
                        std::copy(p.groups.begin(), p.groups.end(), w.View<Contribution>().data);
                    }
                    state->deviceGroups = Buffer::Allocate(host.Bytes(), state->destination.GetDevice());
                    Copy(host, state->deviceGroups, {state->destination.GetDevice(), nullptr}).Wait();
                }
            }
            if (transport == TransferTransport::HostPacked)
            {
                state->hostSource = Buffer::Allocate(state->source.Bytes());
                state->hostDestination = Buffer::Allocate(state->destination.Bytes());
                const size_t peers = p.route.SendRows().size();
                state->packedSend.resize(peers);
                state->packedReceive.resize(peers);
                state->packedSendSizes.resize(peers);
                for (size_t peer = 0; peer < peers; ++peer)
                {
                    auto receiveType = state->reducer ? p.contributionTypes[peer] : p.receiveTypes[peer];
                    MPI_Count sendBytes = 0, receiveBytes = 0;
                    MPI_Type_size_x(p.sendTypes[peer]->value, &sendBytes);
                    MPI_Type_size_x(receiveType->value, &receiveBytes);
                    DNDS_check_throw_info(sendBytes < INT_MAX && receiveBytes < INT_MAX, "host-packed peer payload exceeds MPI int count");
                    int sendSize = 0, receiveSize = 0;
                    MPI_Pack_size(1, p.sendTypes[peer]->value, state->comm, &sendSize);
                    MPI_Pack_size(1, receiveType->value, state->comm, &receiveSize);
                    state->packedSend[peer].resize(std::max(sendSize, 1));
                    state->packedReceive[peer].resize(std::max(receiveSize, 1));
                    int position = 0;
                    if (sendBytes)
                        MPI_Pack(state->hostSource.LegacyPointer(), 1, p.sendTypes[peer]->value,
                                 state->packedSend[peer].data(), int(state->packedSend[peer].size()), &position, state->comm);
                    state->packedSendSizes[peer] = position;
                }
            }
        }
        catch (...)
        {
            valid = false;
        }
        CheckAll(valid, state->comm, "cannot allocate/compile transfer workspace");
        auto *receiveBase = state->reducer ? state->workspace.LegacyPointer() : state->destination.LegacyPointer();
        for (int peer = 0; peer < int(p.route.SendRows().size()); ++peer)
        {
            if (!p.route.ReceiveRows()[peer].empty())
            {
                state->receives.push_back(MPI_REQUEST_NULL);
                auto type = state->reducer ? p.contributionTypes[peer] : p.receiveTypes[peer];
                if (transport == TransferTransport::Direct)
                    MPI_Recv_init(receiveBase, 1, type->value, peer, 0, state->comm, &state->receives.back());
                else
                    MPI_Recv_init(state->packedReceive[peer].data(), int(state->packedReceive[peer].size()), MPI_PACKED,
                                  peer, 0, state->comm, &state->receives.back());
            }
            if (!p.route.SendRows()[peer].empty())
            {
                state->sends.push_back(MPI_REQUEST_NULL);
                if (transport == TransferTransport::Direct)
                    MPI_Send_init(state->source.LegacyPointer(), 1, p.sendTypes[peer]->value, peer, 0, state->comm, &state->sends.back());
                else
                    MPI_Send_init(state->packedSend[peer].data(), state->packedSendSizes[peer], MPI_PACKED,
                                  peer, 0, state->comm, &state->sends.back());
            }
        }
        _state = std::move(state);
    }

    struct TransferOperation::State
    {
        std::shared_ptr<TransferBinding::State> binding;
        ReadAccess source;
        WriteAccess destination, workspace;
        ExecutionContext context;
        bool started = false, destinationReady = false;
    };
    TransferOperation TransferBinding::Begin(ExecutionContext context) const
    {
        auto op = std::make_unique<TransferOperation::State>();
        op->binding = _state;
        op->context = context;
        bool valid = _state->plan->route.Valid();
        {
            std::lock_guard<std::mutex> lock(_state->mutex);
            valid = valid && !_state->active;
        }
        CheckAll(valid, _state->comm, "transfer binding is active or routing is stale");
        try
        {
            auto ctx = [context](const Buffer &b)
            { return b.GetDevice() == Device::Host() ? ExecutionContext::Host() : context; };
            op->source = _state->source.Read(ctx(_state->source));
            op->destination = _state->destination.Write(ctx(_state->destination));
            if (_state->reducer)
                op->workspace = _state->workspace.Write(ctx(_state->workspace));
            // Producers must have released their write leases. This event also
            // orders prior work on the caller's stream before portable MPI.
            Completion::Record(context).Wait();
            if (_state->transport == TransferTransport::HostPacked)
            {
                auto stage = [context](const void *src, Device sd, void *dst, size_t bytes)
                { Detail::CopyBytes(src, sd, dst, Device::Host(), bytes, sd == Device::Host() ? ExecutionContext::Host() : context).Wait(); };
                stage(op->source.View<uint8_t>().data, _state->source.GetDevice(), _state->hostSource.LegacyPointer(), _state->source.Bytes());
                stage(op->destination.View<uint8_t>().data, _state->destination.GetDevice(), _state->hostDestination.LegacyPointer(), _state->destination.Bytes());
                for (size_t peer = 0; peer < _state->packedSend.size(); ++peer)
                {
                    if (!_state->packedSendSizes[peer])
                        continue;
                    int position = 0;
                    MPI_Pack(_state->hostSource.LegacyPointer(), 1, _state->plan->sendTypes[peer]->value,
                             _state->packedSend[peer].data(), int(_state->packedSend[peer].size()), &position, _state->comm);
                    DNDS_check_throw_info(position == _state->packedSendSizes[peer], "MPI packed payload size changed");
                }
            }
        }
        catch (...)
        {
            valid = false;
        }
        try
        {
            CheckAll(valid, _state->comm, "transfer access/stream preflight failed on a rank");
        }
        catch (...)
        {
            op->source.CancelUnmodified();
            op->destination.CancelUnmodified();
            op->workspace.CancelUnmodified();
            throw;
        }
        {
            std::lock_guard<std::mutex> lock(_state->mutex);
            _state->active = true;
        }
        if (!_state->receives.empty())
            MPI_Startall(int(_state->receives.size()), _state->receives.data());
        if (!_state->sends.empty())
            MPI_Startall(int(_state->sends.size()), _state->sends.data());
        op->started = true;
        return TransferOperation(std::move(op));
    }
    TransferOperation::TransferOperation() = default;
    TransferOperation::TransferOperation(std::unique_ptr<State> state) : _state(std::move(state)) {}
    TransferOperation::TransferOperation(TransferOperation &&) noexcept = default;
    TransferOperation &TransferOperation::operator=(TransferOperation &&other) noexcept
    {
        if (this != &other)
        {
            try
            {
                Wait();
            }
            catch (...)
            {
                std::terminate();
            }
            _state = std::move(other._state);
        }
        return *this;
    }
    TransferOperation::~TransferOperation()
    {
        try
        {
            Wait();
        }
        catch (...)
        {
            // A reducer failure has drained communication and poisoned output.
            // Explicit Wait reports it; destruction must never retry the reducer.
            if (_state && _state->started)
                std::terminate();
        }
    }
    void TransferOperation::WaitDestinationReady()
    {
        if (!_state || !_state->started || _state->destinationReady)
            return;
        auto &s = *_state;
        auto &b = *s.binding;
        MPI_Waitall(int(b.receives.size()), b.receives.data(), MPI_STATUSES_IGNORE);
        try
        {
            const bool packed = b.transport == TransferTransport::HostPacked;
            if (packed)
            {
                for (size_t peer = 0; peer < b.packedReceive.size(); ++peer)
                {
                    if (b.plan->route.ReceiveRows()[peer].empty())
                        continue;
                    int position = 0;
                    auto type = b.reducer ? b.plan->contributionTypes[peer] : b.plan->receiveTypes[peer];
                    MPI_Count bytes = 0;
                    MPI_Type_size_x(type->value, &bytes);
                    if (!bytes)
                        continue;
                    MPI_Unpack(b.packedReceive[peer].data(), int(b.packedReceive[peer].size()), &position,
                               b.reducer ? s.workspace.View<uint8_t>().data : b.hostDestination.LegacyPointer(), 1, type->value, b.comm);
                }
            }
            if (b.reducer)
            {
                if (packed || b.destination.GetDevice() == Device::Host())
                    b.reducer->host(s.workspace.View<std::byte>().data, packed ? b.hostDestination.LegacyPointer() : s.destination.View<std::byte>().data,
                                    b.plan->groups, ExecutionContext::Host());
                else
                    b.reducer->device(s.workspace.View<std::byte>().data, s.destination.View<std::byte>().data,
                                      static_cast<const Contribution *>(b.deviceGroups.LegacyPointer()), b.plan->groups.size(), s.context);
                if (!packed && b.destination.GetDevice().space == MemorySpace::CUDA)
                {
                    auto done = Completion::Record(s.context);
                    s.destination.Complete(done);
                    s.workspace.Complete(done);
                }
            }
            if (packed)
                Detail::CopyBytes(b.hostDestination.LegacyPointer(), Device::Host(), s.destination.View<uint8_t>().data,
                                  b.destination.GetDevice(), b.destination.Bytes(), b.destination.GetDevice() == Device::Host() ? ExecutionContext::Host() : s.context)
                    .Wait();
            s.workspace.Wait();
            s.destination.Wait();
            s.destinationReady = true;
        }
        catch (...)
        {
            auto error = std::current_exception();
            // Incoming messages are already drained. Outgoing messages must also
            // finish before releasing storage, even if a custom combine throws.
            MPI_Waitall(int(b.sends.size()), b.sends.data(), MPI_STATUSES_IGNORE);
            try
            {
                s.destination.Fail();
            }
            catch (...)
            {
            }
            try
            {
                s.workspace.Fail();
            }
            catch (...)
            {
            }
            try
            {
                s.source.Wait();
            }
            catch (...)
            {
            }
            s.started = false;
            {
                std::lock_guard<std::mutex> lock(b.mutex);
                b.active = false;
            }
            std::rethrow_exception(error);
        }
    }
    void TransferOperation::Wait()
    {
        if (!_state || !_state->started)
            return;
        WaitDestinationReady();
        auto &s = *_state;
        auto &b = *s.binding;
        MPI_Waitall(int(b.sends.size()), b.sends.data(), MPI_STATUSES_IGNORE);
        s.source.Wait();
        s.started = false;
        std::lock_guard<std::mutex> lock(b.mutex);
        b.active = false;
    }
    bool TransferOperation::Test()
    {
        if (!_state || !_state->started)
            return true;
        auto &b = *_state->binding;
        int send = 0, receive = 0;
        MPI_Testall(int(b.sends.size()), b.sends.data(), &send, MPI_STATUSES_IGNORE);
        MPI_Testall(int(b.receives.size()), b.receives.data(), &receive, MPI_STATUSES_IGNORE);
        if (send && receive)
        {
            Wait();
            return true;
        }
        return false;
    }
}
