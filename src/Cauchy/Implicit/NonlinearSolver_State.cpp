#include "Cauchy/Implicit/NonlinearSolver.hpp"
#include <cmath>

namespace DNDS::Cauchy
{
    void to_json(Json &value, const NonlinearSolveState &state)
    {
        value = Json{{"schema_version", state.schemaVersion},
                     {"attempt", state.attempt},
                     {"iteration", state.iteration},
                     {"stage", state.stage},
                     {"coupled_stage", state.coupledStage},
                     {"converged", state.converged},
                     {"terminal", state.terminal},
                     {"capped", state.capped},
                     {"rejected", state.rejected},
                     {"residual", state.residual},
                     {"initial_residual", state.initialResidual},
                     {"reference_residual", state.referenceResidual},
                     {"convergence_components", state.convergenceComponents},
                     {"residual_target_ratio", state.residualTargetRatio},
                     {"pseudo_cfl", state.pseudoCFL},
                     {"linear_iterations", state.linearIterations},
                     {"multigrid_cycles", state.multigridCycles},
                     {"level_corrections", state.levelCorrections},
                     {"rhs_limited", state.rhsLimited},
                     {"increment_limited", state.incrementLimited},
                     {"minimum_rhs_alpha", state.minimumRHSAlpha},
                     {"minimum_increment_alpha", state.minimumIncrementAlpha},
                     {"reconstruction_iterations", state.reconstructionIterations},
                     {"reconstruction_relative_update", state.reconstructionRelativeUpdate},
                     {"reconstruction_update_measured", state.reconstructionUpdateMeasured}};
    }
    void from_json(const Json &value, NonlinearSolveState &state)
    {
        DNDS_check_throw_info(value.is_object() && value.value("schema_version", 0) == 1,
                              "unsupported nonlinear solve state schema");
        state.schemaVersion = 1;
        state.attempt = value.at("attempt").get<index>();
        state.iteration = value.at("iteration").get<index>();
        state.stage = value.at("stage").get<int>();
        state.coupledStage = value.at("coupled_stage").get<int>();
        state.converged = value.at("converged").get<bool>();
        state.terminal = value.at("terminal").get<bool>();
        state.capped = value.at("capped").get<bool>();
        state.rejected = value.at("rejected").get<bool>();
        state.residual = value.at("residual").get<std::vector<real>>();
        state.initialResidual = value.at("initial_residual").get<std::vector<real>>();
        state.referenceResidual = value.at("reference_residual").get<std::vector<real>>();
        state.convergenceComponents = value.at("convergence_components").get<std::vector<std::string>>();
        state.residualTargetRatio = value.at("residual_target_ratio").get<real>();
        state.pseudoCFL = value.at("pseudo_cfl").get<real>();
        state.linearIterations = value.at("linear_iterations").get<index>();
        state.multigridCycles = value.at("multigrid_cycles").get<index>();
        state.levelCorrections = value.at("level_corrections").get<std::vector<index>>();
        state.rhsLimited = value.at("rhs_limited").get<index>();
        state.incrementLimited = value.at("increment_limited").get<index>();
        state.minimumRHSAlpha = value.at("minimum_rhs_alpha").get<real>();
        state.minimumIncrementAlpha = value.at("minimum_increment_alpha").get<real>();
        state.reconstructionIterations = value.at("reconstruction_iterations").get<int>();
        state.reconstructionRelativeUpdate = value.at("reconstruction_relative_update").get<real>();
        state.reconstructionUpdateMeasured = value.value("reconstruction_update_measured", true);
        DNDS_check_throw_info(state.attempt >= 0 && state.iteration >= 0 && std::isfinite(state.pseudoCFL) &&
                                  std::isfinite(state.residualTargetRatio) && std::isfinite(state.minimumRHSAlpha) &&
                                  std::isfinite(state.minimumIncrementAlpha) && std::isfinite(state.reconstructionRelativeUpdate),
                              "invalid nonlinear solve state values");
    }
    Json NonlinearSolveState::ToJson() const { return Json(*this); }
    NonlinearSolveState NonlinearSolveState::FromJson(const Json &value)
    {
        return value.get<NonlinearSolveState>();
    }
}
