#include "Cauchy/Implicit/LinearProblem.hpp"
#include "Cauchy/Implicit/LinearBlockKernels.hpp"
#include "Cauchy/State/EquationDofs.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    struct LinearProblemContext::EquationSGSIteration::State
    {
        std::shared_ptr<const Pattern> pattern;
        index components;
        DiagonalStorage mode;
        EquationDofs rhs, forward, output;
        std::optional<EquationDofs> input;
        R::Array<real>::ReadLease diagonalRead, inverseRead;
        OffDiagonalReadAccess blocksRead;
        EquationDofs::ReadScope rhsRead;
        std::optional<EquationDofs::ReadScope> inputRead, inputAllRead, forwardRead, forwardAllRead;
        std::optional<EquationDofs::WriteScope> forwardWrite, outputWrite;
        std::vector<uint64_t> initialForwardGhost;
        bool middleExchange, started = false;
        int lower = 0, upper = 0;
        State(const LinearProblemContext &problem, const EquationDofs &b, const EquationDofs *old, EquationDofs &temporary, EquationDofs &result, bool middle)
            : pattern(problem._pattern), components(problem._components), mode(problem._diagonalStorage),
              rhs(b.NumericAlias()), forward(temporary.NumericAlias()), output(result.NumericAlias()),
              diagonalRead(problem._diagonal.Read()), inverseRead(problem._inverse.Read()), blocksRead(problem.ReadOffDiagonal()),
              rhsRead(rhs.ReadOwned()), middleExchange(middle)
        {
            if (old)
            {
                input.emplace(old->NumericAlias());
                inputRead.emplace(input->ReadOwned());
            }
            DNDS_check_throw_info(problem._inverseCurrent && problem._diagonal.Version() == problem._diagonalVersion &&
                                      problem._inverse.Version() == problem._inverseVersion,
                                  "logical SGS inverse is absent or stale");
            for (index iField = 0; iField < forward.PhysicalFieldNumber(); ++iField)
                initialForwardGhost.push_back(forward.PhysicalField(iField).Ghost().Version());
            forwardWrite.emplace(forward.WriteOwned());
            try
            {
                outputWrite.emplace(output.WriteOwned());
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
        template <bool Upper, class Forward>
        void Scan(Forward forwardValues, Algebra::DofComponentView<real> destination, Algebra::DofComponentView<const real> ghost,
                  const std::vector<index> &rows)
        {
            const auto b = rhsRead.ComponentLayout();
            std::optional<Algebra::DofComponentLayout<const real>> old;
            if (inputRead)
                old.emplace(inputRead->ComponentLayout());
            const auto previous = old ? old->View() : b.View();
            const auto inverse = Algebra::MakeEigenRowMatrices(inverseRead.View<R::TableRows>(), components, mode == DiagonalStorage::Block ? components : 1);
            started = true;
            blocksRead.WithActions([&](const auto &blocks)
                                   {
                const auto scan = [&](auto zero)
                {
                    constexpr bool Zero = decltype(zero)::value;
                    if(components==5) Detail::ScanSGSRows<Upper,5,Zero>(b.View(),previous,forwardValues,destination,ghost,inverse,blocks,*pattern,rows,components,mode==DiagonalStorage::Block);
                    else Detail::ScanSGSRows<Upper,Eigen::Dynamic,Zero>(b.View(),previous,forwardValues,destination,ghost,inverse,blocks,*pattern,rows,components,mode==DiagonalStorage::Block);
                };
                if (!input && (!Upper || !middleExchange)) scan(std::true_type{});
                else scan(std::false_type{}); });
        }
    };
    LinearProblemContext::EquationSGSIteration::EquationSGSIteration(std::unique_ptr<State> state) : _state(std::move(state)) {}
    LinearProblemContext::EquationSGSIteration::EquationSGSIteration(EquationSGSIteration &&) noexcept = default;
    LinearProblemContext::EquationSGSIteration &LinearProblemContext::EquationSGSIteration::operator=(EquationSGSIteration &&) noexcept = default;
    LinearProblemContext::EquationSGSIteration::~EquationSGSIteration() = default;

    LinearProblemContext::EquationSGSIteration LinearProblemContext::BeginSGS(const EquationDofs &rhs, const EquationDofs &input,
                                                                              EquationDofs &forward, EquationDofs &output, bool middleExchange) const
    {
        ValidateMatrix();
        ValidateOutputs(input, forward);
        ValidateOutputs(input, output);
        ValidateOutputs(rhs, forward);
        ValidateOutputs(rhs, output);
        ValidateOutputs(forward, output);
        return EquationSGSIteration(std::make_unique<EquationSGSIteration::State>(*this, rhs, &input, forward, output, middleExchange));
    }
    LinearProblemContext::EquationSGSIteration LinearProblemContext::BeginSGSZero(const EquationDofs &rhs, EquationDofs &forward,
                                                                                  EquationDofs &output, bool middleExchange) const
    {
        ValidateMatrix();
        ValidateOutputs(rhs, forward);
        ValidateOutputs(rhs, output);
        ValidateOutputs(forward, output);
        return EquationSGSIteration(std::make_unique<EquationSGSIteration::State>(*this, rhs, nullptr, forward, output, middleExchange));
    }
    void LinearProblemContext::EquationSGSIteration::Lower(LinearRegion region)
    {
        DNDS_check_throw_info(_state, "logical SGS iteration is finished or moved");
        auto &state = *_state;
        DNDS_check_throw_info((region == LinearRegion::All && state.lower == 0) || (region == LinearRegion::Core && state.lower == 0) ||
                                  (region == LinearRegion::HaloDependent && state.lower == 1),
                              "logical SGS lower regions must be All or Core then HaloDependent");
        if (state.input && region != LinearRegion::Core && state.pattern->hasGhostColumns)
            state.inputAllRead.emplace(state.input->ReadWithGhost());
        std::optional<Algebra::DofComponentLayout<const real>> ghost;
        if (state.inputAllRead)
            ghost.emplace(state.inputAllRead->ComponentLayout());
        const auto target = state.forwardWrite->ComponentLayout();
        const auto &rows = region == LinearRegion::All ? state.pattern->order : region == LinearRegion::Core ? state.pattern->core
                                                                                                             : state.pattern->dependent;
        state.Scan<false>(target.View(), target.View(), ghost ? ghost->View().GhostPart() : Algebra::DofComponentView<const real>{}, rows);
        state.lower = region == LinearRegion::Core ? 1 : 2;
        if (state.lower == 2)
        {
            state.forwardWrite.reset();
            state.forwardRead.emplace(state.forward.ReadOwned());
        }
    }
    void LinearProblemContext::EquationSGSIteration::Upper(LinearRegion region)
    {
        DNDS_check_throw_info(_state, "logical SGS iteration is finished or moved");
        auto &state = *_state;
        DNDS_check_throw_info(state.lower == 2 && ((region == LinearRegion::All && state.upper == 0) ||
                                                   (region == LinearRegion::HaloDependent && state.upper == 0) || (region == LinearRegion::Core && state.upper == 1)),
                              "logical SGS upper needs complete lower work and All or HaloDependent then Core");
        if (state.middleExchange && state.pattern->hasGhostColumns && !state.forwardAllRead)
        {
            auto read = state.forward.ReadWithGhost();
            for (index iField = 0; iField < state.forward.PhysicalFieldNumber(); ++iField)
                DNDS_check_throw_info(state.forward.PhysicalField(iField).Ghost().Version() != state.initialForwardGhost[iField],
                                      "logical SGS middle exchange has not published every forward field");
            state.forwardAllRead.emplace(std::move(read));
        }
        std::optional<Algebra::DofComponentLayout<const real>> ghost;
        if (state.middleExchange && state.forwardAllRead)
            ghost.emplace(state.forwardAllRead->ComponentLayout());
        if (!state.middleExchange && state.inputAllRead)
            ghost.emplace(state.inputAllRead->ComponentLayout());
        const auto forward = state.forwardRead->ComponentLayout();
        const auto target = state.outputWrite->ComponentLayout();
        const auto &rows = region == LinearRegion::All ? state.pattern->order : region == LinearRegion::Core ? state.pattern->core
                                                                                                             : state.pattern->dependent;
        state.Scan<true>(forward.View(), target.View(), ghost ? ghost->View().GhostPart() : Algebra::DofComponentView<const real>{}, rows);
        state.upper = region == LinearRegion::HaloDependent ? 1 : 2;
    }
    void LinearProblemContext::EquationSGSIteration::Finish()
    {
        DNDS_check_throw_info(_state && _state->lower == 2 && _state->upper == 2, "cannot publish an incomplete logical SGS iteration");
        _state->outputWrite.reset();
        _state.reset();
    }
    void LinearProblemContext::ApplySGS(const EquationDofs &rhs, const EquationDofs &input, EquationDofs &forward, EquationDofs &output) const
    {
        auto iteration = BeginSGS(rhs, input, forward, output);
        iteration.Lower();
        iteration.Upper();
        iteration.Finish();
    }
}
