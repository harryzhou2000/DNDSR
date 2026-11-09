#pragma once

#include "DNDS/MPI/MPI.hpp"
#include <array>
#include <chrono>
#include <vector>

// External diagnostic clocks shared by separately compiled private production
// copies. No solver timer counter or production instrumentation dependency.
namespace DiagnosticTiming
{
    enum class Block
    {
        Auxiliary,
        Spatial,
        Matrix,
        Linear,
        AuxiliaryPrepare,
        AuxiliaryExchange,
        AuxiliaryCompute,
        MeanValidation,
        DefectAssembly,
        PseudoTimePP,
        CorrectionLocal,
        CandidateAcceptance,
        CorrectionStatistics,
        Gradient,
        Reconstruction,
        PolynomialPP,
        FaceRHS,
        SourceSampling,
        VolumeSource,
        CandidateValidation,
        CandidateAgreement,
        JacobianSetup,
        JacobianFreeze,
        JacobianSource,
        JacobianFaces
    };
    struct Sample
    {
        double auxiliary = 0, spatialInclusive = 0;
        int auxiliaryCalls = 0, spatialCalls = 0;
        std::array<double, 3> levelAuxiliary{}, levelSpatialInclusive{};
        std::array<int, 3> levelAuxiliaryCalls{}, levelSpatialCalls{};
        std::array<double, 3> levelMatrixLinear{};
        std::array<int, 3> levelMatrixLinearCalls{};
        std::array<double, 3> levelMatrix{}, levelLinear{};
        std::array<double, 3> levelAuxiliaryPrepare{}, levelAuxiliaryExchange{}, levelAuxiliaryCompute{};
        // Work audit only. Recorded outside elapsed timing; no extra collective
        // or pointwise counter is inserted in the source sampling loops.
        std::array<DNDS::index, 3> correctionSourceCalls{}, correctionSourcePoints{};
        static constexpr int detailNumber = int(Block::JacobianFaces) - int(Block::MeanValidation) + 1;
        std::array<std::array<double, 3>, detailNumber> details{};
        // Optional single-node causal tracing. Capacity is reserved before
        // timing; emitting an event neither allocates nor communicates.
        bool traceAuxiliary = false, traceOverflow = false;
        std::array<DNDS::index, 3> auxiliaryOrdinals{};
        using AuxiliaryEvent = std::array<int64_t, 11>;
        std::vector<AuxiliaryEvent> auxiliaryEvents;
    };
    inline thread_local Sample *active = nullptr;
    inline thread_local int currentLevel = 0;
    /// Diagnostic-only CPU-inline chronology, using a shared OS monotonic
    /// clock on one node. This does not assume MPI_Wtime clocks are global.
    /// Full evaluations and correction-only entries share per-level ordinals.
    struct AuxiliaryTrace
    {
        enum Event
        {
            Entered,
            LaunchBegin,
            LaunchEnd,
            BoundaryEnd,
            WaitBegin,
            WaitEnd,
            ComputeBegin,
            ComputeEnd
        };
        Sample *sample = nullptr;
        Sample::AuxiliaryEvent row{};
        explicit AuxiliaryTrace(bool enabled, int path, int level)
        {
            if (enabled && active && active->traceAuxiliary)
            {
                sample = active;
                row[0] = level;
                row[1] = sample->auxiliaryOrdinals[level]++;
                row[2] = path;
                Mark(Entered);
            }
        }
        void Mark(Event event)
        {
            if (sample)
                row[3 + int(event)] = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now().time_since_epoch())
                                          .count();
        }
        ~AuxiliaryTrace()
        {
            if (!sample)
                return;
            if (sample->auxiliaryEvents.size() == sample->auxiliaryEvents.capacity())
                sample->traceOverflow = true;
            else
                sample->auxiliaryEvents.push_back(row);
        }
    };
    inline void ObserveCorrectionSource(DNDS::index points)
    {
        if (active)
        {
            ++active->correctionSourceCalls[currentLevel];
            active->correctionSourcePoints[currentLevel] += points;
        }
    }
    struct Level
    {
        int previous = currentLevel;
        explicit Level(int level) { currentLevel = level; }
        ~Level() { currentLevel = previous; }
    };
    struct Scope
    {
        Sample *sample = active;
        Block block;
        int level = currentLevel;
        double start = 0;
        explicit Scope(Block section) : block(section)
        {
            if (sample)
                start = MPI_Wtime();
        }
        Scope(const Scope &) = delete;
        Scope &operator=(const Scope &) = delete;
        void Finish()
        {
            if (!sample)
                return;
            const auto seconds = MPI_Wtime() - start;
            // Detail clocks are nested observations only. Never add them to
            // broad totals or matrix/linear counts, which already enclose them.
            if (block >= Block::MeanValidation)
                sample->details[int(block) - int(Block::MeanValidation)][level] += seconds;
            else if (block == Block::Auxiliary || block == Block::AuxiliaryPrepare ||
                     block == Block::AuxiliaryExchange || block == Block::AuxiliaryCompute)
            {
                sample->auxiliary += seconds;
                ++sample->auxiliaryCalls;
                sample->levelAuxiliary[level] += seconds;
                ++sample->levelAuxiliaryCalls[level];
                if (block == Block::AuxiliaryPrepare)
                    sample->levelAuxiliaryPrepare[level] += seconds;
                if (block == Block::AuxiliaryExchange)
                    sample->levelAuxiliaryExchange[level] += seconds;
                if (block == Block::AuxiliaryCompute)
                    sample->levelAuxiliaryCompute[level] += seconds;
            }
            else if (block == Block::Spatial)
            {
                sample->spatialInclusive += seconds;
                ++sample->spatialCalls;
                sample->levelSpatialInclusive[level] += seconds;
                ++sample->levelSpatialCalls[level];
            }
            else
            {
                sample->levelMatrixLinear[level] += seconds;
                ++sample->levelMatrixLinearCalls[level];
                (block == Block::Matrix ? sample->levelMatrix : sample->levelLinear)[level] += seconds;
            }
            sample = nullptr;
        }
        ~Scope() { Finish(); }
    };
    template <class Action>
    void MeasureMatrix(Action &&action)
    {
        const Scope scope(Block::Matrix);
        action();
    }
}
