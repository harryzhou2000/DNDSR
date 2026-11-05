// E4/E5 opt-in host assessment. Actual Runtime payloads stay unchanged between
// caller/compute-worker/MPI-worker schedules; no production executor is added.
#include "DNDS/Runtime/Communication/RowTransfer.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <filesystem>
#include <future>
#include <iostream>
#include <sched.h>
#include <thread>

namespace R = DNDS::Runtime;
using I = DNDS::index;
using J = nlohmann::ordered_json;
using Clock = std::chrono::steady_clock;

namespace
{
    double Seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
    struct Trace
    {
        Clock::time_point origin;
        double pack = 0, unpack = 0, posting = 0, progress = 0, preflight = 0;
        double firstPost = -1, lastPack = 0;
        I packCalls = 0, unpackCalls = 0, requests = 0, agreements = 0;
    };
    thread_local Trace *trace = nullptr;
    struct JoinTask
    {
        std::future<void> &completion;
        ~JoinTask()
        {
            if (completion.valid())
                completion.wait();
        }
    };
    int TopologyID(int cpu, const char *name)
    {
        std::ifstream input("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/" + name);
        int value = -1;
        input >> value;
        DNDS_check_throw_info(value >= 0, "assessment needs Linux CPU topology");
        return value;
    }
    int NUMANode(int cpu)
    {
        for (const auto &entry : std::filesystem::directory_iterator("/sys/devices/system/cpu/cpu" + std::to_string(cpu)))
        {
            const auto name = entry.path().filename().string();
            if (name.size() > 4 && name.compare(0, 4, "node") == 0 &&
                std::all_of(name.begin() + 4, name.end(), [](char c)
                            { return c >= '0' && c <= '9'; }))
                return std::stoi(name.substr(4));
        }
        return -1; // Explicitly unknown, never infer NUMA identity from a socket.
    }
    void Pin(int cpu)
    {
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(cpu, &mask);
        DNDS_check_throw_info(sched_setaffinity(0, sizeof(mask), &mask) == 0, "cannot bind assessment thread");
    }
    class Worker
    {
        std::mutex _mutex;
        std::condition_variable _changed;
        std::deque<std::packaged_task<void()>> _queue;
        bool _stop = false;
        std::thread _thread;

    public:
        explicit Worker(int cpu) : _thread([this]
                                           {
            for (;;)
            {
                std::packaged_task<void()> task;
                {
                    std::unique_lock<std::mutex> lock(_mutex);
                    _changed.wait(lock, [&] { return _stop || !_queue.empty(); });
                    if (_queue.empty())
                        return;
                    task = std::move(_queue.front());
                    _queue.pop_front();
                }
                task();
            } })
        {
            // Check the worker's actual affinity through a task before MPI work.
            auto ready = Submit([cpu]
                                {
                Pin(cpu);
                cpu_set_t mask;
                CPU_ZERO(&mask);
                DNDS_check_throw_info(sched_getaffinity(0, sizeof(mask), &mask) == 0 && CPU_ISSET(cpu, &mask) && CPU_COUNT(&mask) == 1,
                                      "worker did not obtain its requested CPU"); });
            try
            {
                ready.get();
            }
            catch (...)
            {
                {
                    std::lock_guard<std::mutex> lock(_mutex);
                    _stop = true;
                }
                _changed.notify_one();
                _thread.join();
                throw;
            }
        }
        template <class Action>
        std::future<void> Submit(Action &&action)
        {
            std::packaged_task<void()> task(std::forward<Action>(action));
            auto result = task.get_future();
            {
                std::lock_guard<std::mutex> lock(_mutex);
                _queue.push_back(std::move(task));
            }
            _changed.notify_one();
            return result;
        }
        ~Worker()
        {
            {
                std::lock_guard<std::mutex> lock(_mutex);
                _stop = true;
            }
            _changed.notify_one();
            _thread.join();
        }
    };
}

// PMPI records actual transfer phases. Clock reads are host-only; the controller
// never calls MPI while a communication task is active on the worker.
extern "C" int MPI_Pack(const void *in, int count, MPI_Datatype type, void *out, int bytes, int *position, MPI_Comm comm)
{
    const auto start = Clock::now();
    const int result = PMPI_Pack(in, count, type, out, bytes, position, comm);
    if (trace)
    {
        trace->pack += Seconds(start);
        ++trace->packCalls;
        trace->lastPack = Seconds(trace->origin);
    }
    return result;
}
extern "C" int MPI_Unpack(const void *in, int bytes, int *position, void *out, int count, MPI_Datatype type, MPI_Comm comm)
{
    const auto start = Clock::now();
    const int result = PMPI_Unpack(in, bytes, position, out, count, type, comm);
    if (trace)
    {
        trace->unpack += Seconds(start);
        ++trace->unpackCalls;
    }
    return result;
}
extern "C" int MPI_Startall(int count, MPI_Request *requests)
{
    const auto start = Clock::now();
    const int result = PMPI_Startall(count, requests);
    if (trace)
    {
        if (trace->firstPost < 0)
            trace->firstPost = std::chrono::duration<double>(start - trace->origin).count();
        trace->posting += Seconds(start);
        trace->requests += count;
    }
    return result;
}
extern "C" int MPI_Waitall(int count, MPI_Request *requests, MPI_Status *status)
{
    const auto start = Clock::now();
    const int result = PMPI_Waitall(count, requests, status);
    if (trace)
        trace->progress += Seconds(start);
    return result;
}
extern "C" int MPI_Allreduce(const void *send, void *receive, int count, MPI_Datatype type, MPI_Op operation, MPI_Comm comm)
{
    const auto start = Clock::now();
    const int result = PMPI_Allreduce(send, receive, count, type, operation, comm);
    if (trace)
    {
        trace->preflight += Seconds(start);
        ++trace->agreements;
    }
    return result;
}

namespace
{
    double Value(int rank, I iRow, I component, int field)
    {
        return .5 + .001 * rank + .00001 * iRow + .0001 * component + .1 * field;
    }
    void Run(int argc, char **argv)
    {
        DNDS_check_throw_info(argc == 10, "arguments: owned-rows halo-per-peer modes direct|packed separate|grouped|compute-worker|mpi-worker preflight core-work repetitions trials");
        const I rows = std::stoll(argv[1]), halo = std::stoll(argv[2]), modes = std::stoll(argv[3]);
        const std::string transportName = argv[4], schedule = argv[5];
        const bool checked = std::stoi(argv[6]);
        const int work = std::stoi(argv[7]), repetitions = std::stoi(argv[8]), trials = std::stoi(argv[9]);
        DNDS_check_throw_info(rows >= 32 && rows <= 10000 && halo > 0 && halo <= rows && (modes == 1 || modes == 9) &&
                                  (transportName == "direct" || transportName == "packed") &&
                                  (schedule == "separate" || schedule == "grouped" || schedule == "compute-worker" || schedule == "mpi-worker") &&
                                  work >= 0 && work <= 8 && repetitions > 0 && repetitions <= 100 && trials > 0 && trials <= 4,
                              "transfer scheduling assessment arguments outside bounds");
        int rank = 0, size = 0, provided = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Query_thread(&provided);
        DNDS_check_throw_info(provided >= MPI_THREAD_SERIALIZED, "communication worker needs actual SERIALIZED support");
        cpu_set_t allowed;
        CPU_ZERO(&allowed);
        DNDS_check_throw_info(sched_getaffinity(0, sizeof(allowed), &allowed) == 0, "cannot query assessment affinity");
        std::vector<int> cpus;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
            if (CPU_ISSET(cpu, &allowed))
                cpus.push_back(cpu);
        DNDS_check_throw_info(!cpus.empty(), "empty rank CPU mask");
        const int controllerCPU = cpus.front();
        int workerCPU = -1;
        for (const auto cpu : cpus)
            if (TopologyID(cpu, "core_id") != TopologyID(controllerCPU, "core_id") ||
                TopologyID(cpu, "physical_package_id") != TopologyID(controllerCPU, "physical_package_id"))
            {
                workerCPU = cpu;
                break;
            }
        DNDS_check_throw_info(workerCPU >= 0, "assessment requires two allocated physical cores per rank, including caller controls");
        int placement[]{controllerCPU, workerCPU, TopologyID(controllerCPU, "physical_package_id"), TopologyID(controllerCPU, "core_id"), NUMANode(controllerCPU),
                        TopologyID(workerCPU, "physical_package_id"), TopologyID(workerCPU, "core_id"), NUMANode(workerCPU)};
        std::vector<int> placements(size_t(size) * 8);
        MPI_Allgather(placement, 8, MPI_INT, placements.data(), 8, MPI_INT, MPI_COMM_WORLD);
        std::vector<std::pair<int, int>> physicalCores;
        for (int iRank = 0; iRank < size; ++iRank)
        {
            physicalCores.emplace_back(placements[8 * iRank + 2], placements[8 * iRank + 3]);
            physicalCores.emplace_back(placements[8 * iRank + 5], placements[8 * iRank + 6]);
        }
        std::sort(physicalCores.begin(), physicalCores.end());
        DNDS_check_throw_info(std::adjacent_find(physicalCores.begin(), physicalCores.end()) == physicalCores.end(),
                              "MPI mapping did not reserve two distinct physical cores per rank");
        Pin(controllerCPU);
        int dims[2]{};
        MPI_Dims_create(size, 2, dims);
        const int x = rank / dims[1], y = rank % dims[1];
        std::vector<int> peers{((x + 1) % dims[0]) * dims[1] + y, ((x + dims[0] - 1) % dims[0]) * dims[1] + y,
                               x * dims[1] + (y + 1) % dims[1], x * dims[1] + (y + dims[1] - 1) % dims[1]};
        std::sort(peers.begin(), peers.end());
        peers.erase(std::unique(peers.begin(), peers.end()), peers.end());
        if (size > 1)
            peers.erase(std::remove(peers.begin(), peers.end(), rank), peers.end());
        const I ghosts = I(peers.size()) * halo;
        const auto setupStart = Clock::now();
        auto context = std::make_shared<R::CommunicationContext>(MPI_COMM_WORLD, checked);
        std::vector<R::RowRequest> requests;
        for (I peer = 0; peer < I(peers.size()); ++peer)
            for (I iRow = 0; iRow < halo; ++iRow)
                requests.push_back({peers[peer], (17 * iRow + (rank ^ peers[peer])) % rows, peer * halo + iRow});
        R::RowTransferPlan route(context, rows, ghosts, requests);
        std::vector<R::Array<double>> input, output, calculation;
        std::vector<R::TransferBinding> bindings;
        const auto transport = transportName == "direct" ? R::TransferTransport::Direct : R::TransferTransport::HostPacked;
        for (int field = 0; field < 2; ++field)
        {
            const I width = (field == 0 ? 5 : 12) * modes;
            input.emplace_back(R::RowLayout::Dense(rows, width));
            output.emplace_back(R::RowLayout::Dense(ghosts, width));
            calculation.emplace_back(R::RowLayout::Dense(rows, width));
            auto write = input.back().Write();
            const auto values = write.View<R::TableRows>();
            for (I iRow = 0; iRow < rows; ++iRow)
                for (I j = 0; j < width; ++j)
                    values(iRow, j) = Value(rank, iRow, j, field);
            R::PayloadPlan payload(route, input.back().Layout(), output.back().Layout(), R::ElementSchema::For<double>());
            bindings.push_back(payload.Bind(input.back(), output.back(), transport, {R::TransferMode::Persistent, context, field + 10}));
        }
        R::TransferGroup group(context, bindings);
        const double setupSeconds = Seconds(setupStart);
        const auto generations = std::array<uint64_t, 2>{input[0].Version(), input[1].Version()};
        // Declared after all MPI owners, so joining the worker precedes their
        // controller-thread destruction even when a computation throws.
        Worker worker(workerCPU);
        const auto compute = [&]
        {
            if (!work)
                return;
            for (int field = 0; field < 2; ++field)
            {
                auto read = input[field].Read();
                auto write = calculation[field].Write();
                const auto in = read.View<R::TableRows>();
                const auto out = write.View<R::TableRows>();
                for (I iRow = 0; iRow < rows; ++iRow)
                    for (I j = 0; j < in.Width(); ++j)
                    {
                        double value = in(iRow, j);
                        const double neighbor = in((iRow + 1) % rows, j);
                        for (int step = 0; step < work; ++step)
                            value = .9 * value + .1 * (std::sqrt(1 + value * value) + std::exp(-.1 * value * value) + .05 * neighbor);
                        out(iRow, j) = value;
                    }
            }
        };
        const auto validate = [&]
        {
            for (int field = 0; field < 2; ++field)
            {
                DNDS_check_throw_info(input[field].Version() == generations[field], "transfer/core workload changed frozen owned input");
                auto read = output[field].Read();
                const auto values = read.View<R::TableRows>();
                for (I peer = 0; peer < I(peers.size()); ++peer)
                    for (I iRow = 0; iRow < halo; ++iRow)
                        for (I j = 0; j < values.Width(); ++j)
                            DNDS_check_throw_info(values(peer * halo + iRow, j) == Value(peers[peer], (17 * iRow + (rank ^ peers[peer])) % rows, j, field),
                                                  "exact independent halo payload oracle failed");
                if (work)
                {
                    auto result = calculation[field].Read();
                    const auto computed = result.View<R::TableRows>();
                    for (I iRow = 0; iRow < rows; ++iRow)
                        for (I j = 0; j < values.Width(); ++j)
                        {
                            double expected = Value(rank, iRow, j, field);
                            const auto neighbor = Value(rank, (iRow + 1) % rows, j, field);
                            for (int step = 0; step < work; ++step)
                                expected = .9 * expected + .1 * (std::sqrt(1 + expected * expected) + std::exp(-.1 * expected * expected) + .05 * neighbor);
                            DNDS_check_throw_info(computed(iRow, j) == expected, "owned-only computation oracle failed");
                        }
                }
            }
        };
        for (int trial = -1; trial < trials; ++trial)
        {
            double totals[14]{};
            I counters[5]{};
            for (int iteration = 0; iteration < repetitions; ++iteration)
            {
                MPI_Barrier(MPI_COMM_WORLD);
                Trace sample;
                sample.origin = Clock::now();
                double communicationStart = 0, beginEnd = 0, destinationReady = 0, fullReady = 0, computeStart = 0, computeEnd = 0, exposed = 0;
                std::vector<R::TransferOperation> operations;
                const auto begin = [&]
                {
                    communicationStart = Seconds(sample.origin);
                    trace = &sample;
                    if (schedule == "grouped")
                        operations = group.Begin();
                    else
                        for (const auto &binding : bindings)
                            operations.push_back(binding.Begin());
                    beginEnd = Seconds(sample.origin);
                };
                const auto finish = [&]
                {
                    for (auto &operation : operations)
                        operation.WaitDestinationReady();
                    destinationReady = Seconds(sample.origin);
                    for (auto &operation : operations)
                        operation.Wait();
                    fullReady = Seconds(sample.origin);
                    trace = nullptr;
                };
                const auto core = [&]
                { computeStart = Seconds(sample.origin); compute(); computeEnd = Seconds(sample.origin); };
                if (schedule == "mpi-worker")
                {
                    auto completion = worker.Submit([&]
                                                    { begin(); finish(); });
                    const JoinTask join{completion};
                    core();
                    const auto start = Clock::now();
                    completion.get();
                    exposed = Seconds(start);
                }
                else if (schedule == "compute-worker")
                {
                    auto completion = worker.Submit(core);
                    const JoinTask join{completion};
                    begin();
                    finish();
                    completion.get();
                }
                else
                {
                    begin();
                    core();
                    const auto start = Clock::now();
                    finish();
                    exposed = Seconds(start);
                }
                const double total = Seconds(sample.origin);
                const double overlap = std::max(0., std::min(computeEnd, fullReady) - std::max(computeStart, communicationStart));
                const double values[]{total, beginEnd, computeEnd - computeStart, destinationReady, fullReady, exposed, overlap,
                                      sample.preflight, sample.pack, sample.posting, sample.progress, sample.unpack, sample.firstPost, communicationStart};
                for (int j = 0; j < 14; ++j)
                    totals[j] += values[j] / repetitions;
                const I counts[]{sample.agreements, sample.packCalls, sample.unpackCalls, sample.requests, I(peers.size()) * 2 * 2};
                for (int j = 0; j < 5; ++j)
                    counters[j] += counts[j];
                validate(); // Outside every timing, before the next MPI barrier.
            }
            MPI_Allreduce(MPI_IN_PLACE, totals, 14, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, counters, 5, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);
            DNDS_check_throw_info(counters[0] == (checked ? (schedule == "grouped" ? 1 : 2) * I(repetitions) : 0) && counters[3] == counters[4],
                                  "actual group agreement/request count differs");
            DNDS_check_throw_info(counters[1] == (transportName == "packed" ? 2 * I(peers.size()) * repetitions : 0) && counters[2] == counters[1],
                                  "actual pack/unpack call count differs");
            if (rank == 0 && trial >= 0)
            {
                const char *names[]{"mock_stage", "begin_end_offset", "core", "destination_ready_offset", "full_ready_offset", "exposed_wait", "host_overlap_interval",
                                    "preflight", "pack", "posting", "mpi_wait", "unpack", "first_post_offset", "communication_start_offset"};
                J phases;
                for (int j = 0; j < 14; ++j)
                    phases[names[j]] = totals[j];
                std::cout << "PLAN53_TRANSFER " << J{{"np", size}, {"rows", rows}, {"halo", halo}, {"modes", modes}, {"transport", transportName}, {"schedule", schedule}, {"preflight", checked}, {"core_work", work}, {"trial", trial}, {"repetitions", repetitions}, {"provided", provided}, {"setup_seconds_rank0", setupSeconds}, {"placement_columns", {"controller_cpu", "worker_cpu", "controller_socket", "controller_core", "controller_numa", "worker_socket", "worker_core", "worker_numa"}}, {"placements", placements}, {"all_oracles", true}, {"source_generations_unchanged", true}, {"receive_payload_bytes", 8 * ghosts * modes * 17}, {"explicit_staging_copy_bytes", transportName == "packed" ? 8 * (rows + 2 * ghosts) * modes * 17 : 0}, {"max_seconds", phases}, {"counts", counters}}.dump() << std::endl;
            }
        }
    }
}
int main(int argc, char **argv)
{
    const int initialized = DNDS::MPI::Init_thread(&argc, &argv);
    if (initialized != MPI_SUCCESS)
        return initialized;
    try
    {
        Run(argc, argv);
    }
    catch (const std::exception &error)
    {
        std::cerr << "transfer scheduling assessment: " << error.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    DNDS::MPI::Finalize();
    return 0;
}
