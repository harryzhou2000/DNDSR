#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "DNDS/IO/CsvLog.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>

TEST_CASE("Typed CSV rows preserve schema types blanks escaping and lazy rotation")
{
    namespace fs = std::filesystem;
    using namespace DNDS;
    const auto serial = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto directory = fs::temp_directory_path() / ("dnds-csv-" + std::to_string(serial));
    const auto bare = (directory / "history").string();
    CsvLog log(CsvSchema(std::vector<CsvColumn>{{"count", CsvColumnType::Integer}, {"value", CsvColumnType::Real}, {"note", CsvColumnType::Text}}),
               bare, ".csv", 1);

    auto first = log.MakeRow();
    first.Set("count", 7);
    first.Set("value", 2.5);
    first.Set("note", std::string("a,\"b\""));
    CHECK_THROWS(first.Set("count", 2.5));
    log.WriteLine(first);
    CHECK_FALSE(fs::exists(bare + "_1.csv"));

    auto second = log.MakeRow();
    second.Set("count", 8);
    log.WriteLine(second);
    log.Flush();
    CHECK_FALSE(fs::exists(bare + "_2.csv"));

    std::ifstream a(bare + ".csv"), b(bare + "_1.csv");
    REQUIRE(a);
    REQUIRE(b);
    std::string header, row;
    REQUIRE(std::getline(a, header));
    REQUIRE(std::getline(a, row));
    CHECK(header == "count,value,note");
    CHECK(row == "7,2.50000000000000000e+00,\"a,\"\"b\"\"\"");
    REQUIRE(std::getline(b, header));
    REQUIRE(std::getline(b, row));
    CHECK(header == "count,value,note");
    CHECK(row == "8,,");
    fs::remove_all(directory);
}
