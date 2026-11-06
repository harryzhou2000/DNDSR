#pragma once

#include "DNDS/MPI/MPI.hpp"
#include <array>

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
        AuxiliaryCompute
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
    };
    inline thread_local Sample *active = nullptr;
    inline thread_local int currentLevel = 0;
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
        ~Scope()
        {
            if (!sample)
                return;
            const auto seconds = MPI_Wtime() - start;
            if (block == Block::Auxiliary || block == Block::AuxiliaryPrepare ||
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
        }
    };
    template <class Action>
    void MeasureMatrix(Action &&action)
    {
        const Scope scope(Block::Matrix);
        action();
    }
}
