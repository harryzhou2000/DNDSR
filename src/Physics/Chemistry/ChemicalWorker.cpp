#include "Physics/Chemistry/ChemicalWorker.hpp"
#include <mutex>
#include <utility>

namespace DNDS::Physics::Chemistry
{
    struct ChemicalWorkerPool::State
    {
        struct Worker
        {
            std::unique_ptr<ChemicalSource> source;
            ChemicalWorkBuffers buffers;
            Worker(std::unique_ptr<ChemicalSource> context, int species) : source(std::move(context)), buffers(species) {}
        };
        int species;
        std::mutex mutex;
        std::vector<Worker> workers;
        std::vector<index> available;
        State(const ChemicalSource &prototype, index count) : species(prototype.nSpecies())
        {
            DNDS_check_throw_info(count > 0, "chemical worker count must be positive");
            workers.reserve(count);
            available.reserve(count);
            for (index iWorker = 0; iWorker < count; ++iWorker)
            {
                workers.emplace_back(prototype.clone(), species);
                available.push_back(count - 1 - iWorker);
            }
        }
    };
    ChemicalWorkerPool::ChemicalWorkerPool(const ChemicalSource &prototype, index workers)
        : _state(std::make_shared<State>(prototype, workers)) {}
    index ChemicalWorkerPool::WorkerNumber() const { return index(_state->workers.size()); }
    int ChemicalWorkerPool::SpeciesNumber() const { return _state->species; }
    ChemicalWorkerPool::Lease ChemicalWorkerPool::Acquire() const
    {
        std::lock_guard<std::mutex> guard(_state->mutex);
        DNDS_check_throw_info(!_state->available.empty(), "all chemical worker contexts are in use");
        const auto slot = _state->available.back();
        _state->available.pop_back();
        return Lease(_state, slot);
    }
    void ChemicalWorkerPool::Lease::Release() noexcept
    {
        auto state = std::move(_state);
        if (!state)
            return;
        {
            std::lock_guard<std::mutex> guard(state->mutex);
            // Capacity was reserved for every slot; release cannot allocate.
            state->available.push_back(_slot);
        }
        _slot = -1;
    }
    ChemicalWorkerPool::Lease::~Lease() { Release(); }
    ChemicalWorkerPool::Lease::Lease(Lease &&other) noexcept
        : _state(std::move(other._state)), _slot(std::exchange(other._slot, -1)) {}
    ChemicalWorkerPool::Lease &ChemicalWorkerPool::Lease::operator=(Lease &&other) noexcept
    {
        if (this != &other)
        {
            Release();
            _state = std::move(other._state);
            _slot = std::exchange(other._slot, -1);
        }
        return *this;
    }
    ChemicalSource &ChemicalWorkerPool::Lease::Source() const
    {
        DNDS_check_throw_info(_state, "inactive chemical worker lease");
        return *_state->workers[_slot].source;
    }
    ChemicalWorkBuffers &ChemicalWorkerPool::Lease::Buffers() const
    {
        DNDS_check_throw_info(_state, "inactive chemical worker lease");
        return _state->workers[_slot].buffers;
    }
}
