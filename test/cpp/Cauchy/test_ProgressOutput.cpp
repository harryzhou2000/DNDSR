#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "Cauchy/IO/ProgressOutput.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace C = DNDS::Cauchy;

namespace
{
    std::vector<std::string> Split(const std::string &line)
    {
        std::vector<std::string> fields;
        std::stringstream stream(line);
        for (std::string field; std::getline(stream, field, ',');)
            fields.push_back(std::move(field));
        if (!line.empty() && line.back() == ',')
            fields.emplace_back();
        return fields;
    }

    std::map<std::string, std::string> ReadRow(std::ifstream &stream, const std::vector<std::string> &header)
    {
        std::string line;
        REQUIRE(std::getline(stream, line));
        const auto fields = Split(line);
        REQUIRE(fields.size() == header.size());
        std::map<std::string, std::string> result;
        for (size_t i = 0; i < header.size(); ++i)
            result.emplace(header[i], fields[i]);
        return result;
    }
}

TEST_CASE("One progress controller constructs iteration stage and step CSV rows")
{
    C::EquationSchema schema;
    schema.family = "test";
    schema.blocks.push_back({"flow", {{"rho", "", "flow"}, {"rhoE", "", "flow"}}, {}});

    C::OutputSettings settings;
    settings.consoleInterval = 0;
    settings.implicit.consoleInterval = 0;
    settings.implicit.historyInterval = 2;
    settings.historySegmentRows = 2;
    settings.historyPrecision = 8;
    const auto serial = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto directory = std::filesystem::temp_directory_path() / ("dnds-cauchy-progress-" + std::to_string(serial));
    const auto bare = (directory / "case-D0-history").string();
    settings.prefix = (directory / "case").string();
    auto logger = std::make_shared<C::OperationalLogger>(settings, 0);
    {
        C::ProgressOutputController output(schema, settings, "fluid", bare, true, logger);
        output.BeginAttempt({3, -1, -1, -1, 1, .3, .01, 2});
        C::NonlinearProgress progress;
        progress.stage = 1;
        progress.coupledStage = 0;
        progress.residual = {2, 3};
        progress.initialResidual = {4, 5};
        progress.referenceResidual = {8, 10};
        progress.residualTargetRatio = .5;
        progress.pseudoCFL = 7;
        progress.iteration = 1; // filtered by the interval
        output.CaptureIteration(progress);
        progress.iteration = 2;
        output.CaptureIteration(progress);
        output.FlushIterations();

        C::EquationNonlinearIterationReport stage;
        stage.residual = {1, 1.5};
        stage.initialResidual = {4, 5};
        stage.referenceResidual = {8, 10};
        stage.residualTargetRatio = .25;
        stage.pseudoCFL = 9;
        output.StageEnd({3, 1, 0, 99, 1, .3, .01, 2.1}, stage);

        C::AcceptedStepProgress step;
        step.diagnostics[0] = 11;
        step.diagnostics[4] = .125;
        output.StepEnd({3, 7, 6, 5, 1, .3, .01, 2.2}, step, true, false);
        output.Flush();
    }

    std::ifstream first(bare + ".csv");
    REQUIRE(first);
    std::string line;
    REQUIRE(std::getline(first, line));
    const auto header = Split(line);
    CHECK(std::find(header.begin(), header.end(), "res0_rho") == header.end());
    CHECK(std::find(header.begin(), header.end(), "resref_rho") == header.end());
    const auto iteration = ReadRow(first, header);
    const auto stage = ReadRow(first, header);
    CHECK(iteration.at("step") == "3");
    CHECK(iteration.at("stage") == "1");
    CHECK(iteration.at("coupled_stage") == "0");
    CHECK(iteration.at("iteration") == "2");
    CHECK(iteration.at("attempt") == "1");
    CHECK(iteration.at("status") == "0");
    CHECK(iteration.at("res_rho") == "2.00000000e+00");
    CHECK(stage.at("stage") == "1");
    CHECK(stage.at("iteration") == "-1");
    CHECK(stage.at("status") == "1");
    CHECK_FALSE(std::getline(first, line));

    std::ifstream second(bare + "_1.csv");
    REQUIRE(second);
    REQUIRE(std::getline(second, line));
    CHECK(Split(line) == header);
    const auto accepted = ReadRow(second, header);
    CHECK(accepted.at("stage") == "-1");
    CHECK(accepted.at("coupled_stage") == "-1");
    CHECK(accepted.at("iteration") == "-1");
    CHECK(accepted.at("status") == "2");
    CHECK(accepted.at("mass") == "1.10000000e+01");
    CHECK_FALSE(std::getline(second, line));
    CHECK_FALSE(std::filesystem::exists(bare + "_2.csv"));
    std::filesystem::remove_all(directory);
}
