#include "Cauchy/Implicit/LinearProblem.hpp"
#include <optional>

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    using ReadView = R::ArrayView<const real, R::AccessBoundsChecks, R::UniformTableRows>;
    using WriteView = R::ArrayView<real, R::AccessBoundsChecks, R::UniformTableRows>;

    struct LinearProblemContext::SGSIteration::State
    {
        std::shared_ptr<const Pattern> pattern;
        index components;
        DiagonalStorage mode;
        // Pin numeric allocation/layout owners, not communicator owners. A
        // numerical scope may retire on a worker; MPI resources stay outside.
        R::Array<real> rhs, input, inputGhost, forward, forwardGhost, output;
        R::Array<real>::ReadLease diagonalRead, inverseRead, rhsRead, inputRead;
        Algebra::BlockMatrixStorage::ReadAccess blocksRead;
        std::optional<R::Array<real>::ReadLease> inputGhostRead, forwardRead, forwardGhostRead;
        std::optional<R::Array<real>::WriteLease> forwardWrite, outputWrite;
        uint64_t initialForwardGhost;
        bool middleExchange, started = false;
        int lower = 0, upper = 0; // 0=not started,1=first region complete,2=scan complete

        State(const LinearProblemContext &problem, const Field &b, const Field &old, Field &temporary, Field &result, bool middle)
            : pattern(problem._pattern), components(problem._components), mode(problem._diagonalStorage),
              rhs(b.Owned()), input(old.Owned()), inputGhost(old.Ghost()), forward(temporary.Owned()), forwardGhost(temporary.Ghost()), output(result.Owned()),
              diagonalRead(problem._diagonal.Read()), inverseRead(problem._inverse.Read()), rhsRead(rhs.Read()), inputRead(input.Read()),
              blocksRead(problem._offDiagonal.Read()), initialForwardGhost(forwardGhost.Version()), middleExchange(middle)
        {
            DNDS_check_throw_info(problem._inverseCurrent && problem._diagonal.Version() == problem._diagonalVersion &&
                                      problem._inverse.Version() == problem._inverseVersion,
                                  "linear diagonal inverse is missing or stale");
            forwardWrite.emplace(forward.Write());
            try
            {
                outputWrite.emplace(output.Write());
            }
            catch (...)
            {
                forwardWrite->CancelUnmodified();
                forwardWrite.reset();
                throw;
            }
        }
        ~State()
        {
            for (auto *write : {&forwardWrite, &outputWrite})
                if (*write)
                {
                    if (started)
                        (*write)->Fail();
                    else
                        (*write)->CancelUnmodified();
                }
        }
        template <bool Upper, class ForwardView>
        void Scan(ForwardView forwardValues, WriteView destination, ReadView ghostValues, const std::vector<index> &rows)
        {
            const auto b = Algebra::MakeEigenRowMatrices(rhsRead.View<R::UniformTableRows>(), components, 1);
            const auto old = Algebra::MakeEigenRowMatrices(inputRead.View<R::UniformTableRows>(), components, 1);
            const auto remote = Algebra::MakeEigenRowMatrices(ghostValues, components, 1);
            const auto y = Algebra::MakeEigenRowMatrices(forwardValues, components, 1);
            const auto result = Algebra::MakeEigenRowMatrices(destination, components, 1);
            const auto inv = Algebra::MakeEigenRowMatrices(inverseRead.View<R::TableRows>(), components, mode == DiagonalStorage::Block ? components : 1);
            const auto blocks = blocksRead.View();
            const index ownedRows = index(pattern->partition->owned);
            started = true;
            const auto run = [&](auto size)
            {
                Eigen::Matrix<real, decltype(size)::value, 1> value(components);
                for (index scan = 0; scan < index(rows.size()); ++scan)
                {
                    const index iCell = rows[Upper ? index(rows.size()) - 1 - scan : scan];
                    value = b[iCell];
                    const auto &columns = pattern->columns[iCell];
                    for (index j = 0; j < index(columns.size()); ++j)
                    {
                        const auto column = columns[j];
                        if (column >= ownedRows)
                            value.noalias() -= blocks(iCell, j) * remote[column - ownedRows];
                        else if (pattern->position[column] < pattern->position[iCell])
                            value.noalias() -= blocks(iCell, j) * y[column];
                        else if constexpr (Upper)
                            value.noalias() -= blocks(iCell, j) * result[column];
                        else
                            value.noalias() -= blocks(iCell, j) * old[column];
                    }
                    if (mode == DiagonalStorage::Block)
                        result[iCell].noalias() = inv[iCell] * value;
                    else
                        result[iCell].array() = inv[iCell].array() * value.array();
                }
            };
            if (components == 5)
                run(std::integral_constant<int, 5>{});
            else
                run(std::integral_constant<int, Eigen::Dynamic>{});
        }
    };

    LinearProblemContext::SGSIteration::SGSIteration(std::unique_ptr<State> state) : _state(std::move(state)) {}
    LinearProblemContext::SGSIteration::SGSIteration(SGSIteration &&) noexcept = default;
    LinearProblemContext::SGSIteration &LinearProblemContext::SGSIteration::operator=(SGSIteration &&) noexcept = default;
    LinearProblemContext::SGSIteration::~SGSIteration() = default;
    LinearProblemContext::SGSIteration LinearProblemContext::BeginSGS(const Field &rhs, const Field &input, Field &forward, Field &output, bool middleExchange) const
    {
        ValidateMatrix();
        ValidateOutputs(input, forward);
        ValidateOutputs(input, output);
        ValidateOutputs(rhs, forward);
        ValidateOutputs(rhs, output);
        ValidateOutputs(forward, output);
        return SGSIteration(std::make_unique<SGSIteration::State>(*this, rhs, input, forward, output, middleExchange));
    }
    void LinearProblemContext::SGSIteration::Lower(LinearRegion region)
    {
        DNDS_check_throw_info(_state, "SGS iteration is finished or moved");
        auto &state = *_state;
        DNDS_check_throw_info((region == LinearRegion::All && state.lower == 0) || (region == LinearRegion::Core && state.lower == 0) ||
                                  (region == LinearRegion::HaloDependent && state.lower == 1),
                              "SGS lower regions must be All or Core then HaloDependent");
        if (region != LinearRegion::Core && state.pattern->hasGhostColumns)
            state.inputGhostRead.emplace(state.inputGhost.Read());
        const auto ghost = state.inputGhostRead ? state.inputGhostRead->View<R::UniformTableRows>() : ReadView{};
        const auto target = state.forwardWrite->View<R::UniformTableRows>();
        const auto &rows = region == LinearRegion::All ? state.pattern->order : region == LinearRegion::Core ? state.pattern->core
                                                                                                             : state.pattern->dependent;
        state.Scan<false>(target, target, ghost, rows);
        state.lower = region == LinearRegion::Core ? 1 : 2;
        if (state.lower == 2)
        {
            state.forwardWrite.reset();
            state.forwardRead.emplace(state.forward.Read());
        }
    }
    void LinearProblemContext::SGSIteration::Upper(LinearRegion region)
    {
        DNDS_check_throw_info(_state, "SGS iteration is finished or moved");
        auto &state = *_state;
        DNDS_check_throw_info(state.lower == 2 && ((region == LinearRegion::All && state.upper == 0) ||
                                                   (region == LinearRegion::HaloDependent && state.upper == 0) ||
                                                   (region == LinearRegion::Core && state.upper == 1)),
                              "SGS upper needs complete lower work and All or HaloDependent then Core");
        if (state.middleExchange && state.pattern->hasGhostColumns && !state.forwardGhostRead)
        {
            // Generation alone cannot prove the routing, but prevents silently
            // using a previous iteration's unchanged forward halo. Read() also
            // rejects unobserved/active MPI writes before the scan starts.
            auto read = state.forwardGhost.Read();
            DNDS_check_throw_info(state.forwardGhost.Version() != state.initialForwardGhost, "SGS middle exchange has not published a new forward halo");
            state.forwardGhostRead.emplace(std::move(read));
        }
        const auto ghost = state.middleExchange ? (state.forwardGhostRead ? state.forwardGhostRead->View<R::UniformTableRows>() : ReadView{})
                                                : (state.inputGhostRead ? state.inputGhostRead->View<R::UniformTableRows>() : ReadView{});
        const auto &rows = region == LinearRegion::All ? state.pattern->order : region == LinearRegion::Core ? state.pattern->core
                                                                                                             : state.pattern->dependent;
        state.Scan<true>(state.forwardRead->View<R::UniformTableRows>(), state.outputWrite->View<R::UniformTableRows>(), ghost, rows);
        state.upper = region == LinearRegion::HaloDependent ? 1 : 2;
    }
    void LinearProblemContext::SGSIteration::Finish()
    {
        DNDS_check_throw_info(_state && _state->lower == 2 && _state->upper == 2, "cannot publish an incomplete SGS iteration");
        _state->outputWrite.reset();
        _state.reset();
    }
}
