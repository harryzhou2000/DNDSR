#pragma once

#include "Cauchy/IO/RunControl.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <map>
#include <sstream>
#include <chrono>

namespace DNDS::Cauchy
{
    /// HTTP-worker-owned index of complete numeric CSV records. No solver/MPI
    /// callbacks, retained field state or per-row JSON cache. One offset per 256
    /// rows permits bounded page reads; append scans begin at the last newline.
    class HistoryQuery
    {
        struct Segment
        {
            std::filesystem::path path;
            uintmax_t size = 0, end = 0, rows = 0;
            std::filesystem::file_time_type stamp{};
            std::string tail;
            std::vector<uintmax_t> offsets;
            std::vector<std::string> header;
        };
        struct Series
        {
            std::vector<Segment> segments;
            uint64_t generation = 0;
        };
        std::mutex mutex;
        std::map<std::string, Series> series;
        uint64_t serial = uint64_t(std::chrono::system_clock::now().time_since_epoch().count());

        static std::vector<std::string> Split(const std::string &line)
        {
            std::vector<std::string> fields;
            std::string value;
            bool quoted = false;
            for (size_t i = 0; i < line.size(); ++i)
            {
                const char c = line[i];
                if (c == '"')
                {
                    if (quoted && i + 1 < line.size() && line[i + 1] == '"')
                        value += line[++i];
                    else
                        quoted = !quoted;
                }
                else if (c == ',' && !quoted)
                    fields.push_back(value), value.clear();
                else if (c != '\r')
                    value += c;
            }
            if (quoted)
                throw std::runtime_error("multiline CSV records are not supported by numeric history");
            fields.push_back(value);
            return fields;
        }
        static std::string Tail(const Segment &s)
        {
            std::ifstream in(s.path, std::ios::binary);
            const auto n = std::min<uintmax_t>(64, s.end);
            in.seekg(std::streamoff(s.end - n));
            std::string text(size_t(n), '\0');
            in.read(text.data(), std::streamsize(n));
            text.resize(size_t(in.gcount()));
            return text;
        }
        static std::vector<std::filesystem::path> Paths(const std::filesystem::path &base)
        {
            std::map<uint64_t, std::filesystem::path> sorted;
            const auto parent = base.has_parent_path() ? base.parent_path() : std::filesystem::path(".");
            if (!std::filesystem::exists(parent))
                return {};
            const auto stem = base.stem().string(), ext = base.extension().string();
            for (const auto &entry : std::filesystem::directory_iterator(parent))
            {
                if (!entry.is_regular_file())
                    continue;
                const auto p = entry.path();
                if (p.filename() == base.filename())
                    sorted[0] = p;
                else if (p.extension() == ext && p.stem().string().find(stem + "_") == 0)
                {
                    const auto suffix = p.stem().string().substr(stem.size() + 1);
                    if (!suffix.empty() && suffix.find_first_not_of("0123456789") == std::string::npos)
                        sorted[std::stoull(suffix)] = p;
                }
            }
            std::vector<std::filesystem::path> result;
            for (const auto &item : sorted)
                result.push_back(item.second);
            return result;
        }
        static void Scan(Segment &s)
        {
            std::ifstream in(s.path, std::ios::binary);
            if (!in)
                throw std::runtime_error("cannot read history segment");
            in.seekg(std::streamoff(s.end));
            std::string line;
            while (true)
            {
                const auto start = in.tellg();
                if (!std::getline(in, line) || in.eof())
                    break; // defer incomplete tail
                if (s.header.empty())
                    s.header = Split(line);
                else
                {
                    if (s.rows % 256 == 0)
                        s.offsets.push_back(uintmax_t(start));
                    ++s.rows;
                }
                s.end = uintmax_t(in.tellg());
            }
            s.size = std::filesystem::file_size(s.path);
            s.stamp = std::filesystem::last_write_time(s.path);
            s.tail = Tail(s);
        }
        static uint64_t Unsigned(const std::string &s)
        {
            if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
                throw std::invalid_argument("expected nonnegative integer");
            return std::stoull(s);
        }

    public:
        /// Parameters: cursor (opaque generation:raw-row), limit (1..10000),
        /// kind (all/nonlinear/stage/step), axis (iteration/time/step/wall),
        /// start/end inclusive, columns comma separated. Cursor positions refer
        /// to raw rows, before filtering. Replacements return reset=true.
        Json Query(const std::filesystem::path &base, const std::map<std::string, std::string> &q)
        {
            std::lock_guard<std::mutex> lock(mutex);
            const auto arg = [&](const std::string &key, const std::string &fallback = "")
            { const auto it = q.find(key); return it == q.end() ? fallback : it->second; };
            const auto limit = Unsigned(arg("limit", "2000"));
            if (limit == 0 || limit > 10000)
                throw std::invalid_argument("limit must be 1..10000");
            const auto kind = arg("kind", "all"), axis = arg("axis", "iteration");
            if (kind != "all" && kind != "nonlinear" && kind != "stage" && kind != "step")
                throw std::invalid_argument("invalid kind");
            if (axis != "iteration" && axis != "time" && axis != "step" && axis != "wall")
                throw std::invalid_argument("invalid axis");
            auto number = [](const std::string &text)
            { size_t used = 0; const auto v = std::stod(text, &used); if (used != text.size() || !std::isfinite(v)) throw std::invalid_argument("invalid interval"); return v; };
            const auto lo = arg("start").empty() ? -std::numeric_limits<double>::infinity() : number(arg("start"));
            const auto hi = arg("end").empty() ? std::numeric_limits<double>::infinity() : number(arg("end"));
            if (lo > hi)
                throw std::invalid_argument("reversed interval");
            auto &cache = series[base.string()];
            const auto paths = Paths(base);
            bool changed = cache.generation == 0 || paths.size() < cache.segments.size();
            for (size_t i = 0; !changed && i < cache.segments.size(); ++i)
            {
                const auto &s = cache.segments[i];
                const auto size = std::filesystem::file_size(paths[i]);
                changed = paths[i] != s.path || size < s.size || Tail(s) != s.tail ||
                          (size == s.size && std::filesystem::last_write_time(paths[i]) != s.stamp) ||
                          (i + 1 < cache.segments.size() && size != s.size);
            }
            if (changed)
            {
                cache.segments.clear();
                cache.generation = ++serial;
            }
            for (size_t i = 0; i < paths.size(); ++i)
            {
                if (i == cache.segments.size())
                {
                    Segment s;
                    s.path = paths[i];
                    cache.segments.push_back(std::move(s));
                }
                Scan(cache.segments[i]);
            }
            uint64_t total = 0, position = 0;
            std::vector<std::string> header;
            for (const auto &s : cache.segments)
            {
                total += s.rows;
                if (header.empty())
                    header = s.header;
                else if (!s.header.empty() && header != s.header)
                    throw std::runtime_error("history segment schema changed");
            }
            bool reset = false;
            if (!arg("cursor").empty())
            {
                const auto cursor = arg("cursor");
                const auto colon = cursor.find(':');
                if (colon == std::string::npos)
                    throw std::invalid_argument("invalid cursor");
                position = Unsigned(cursor.substr(colon + 1));
                reset = Unsigned(cursor.substr(0, colon)) != cache.generation || position > total;
                if (reset)
                    position = 0;
            }
            auto columns = arg("columns").empty() ? header : Split(arg("columns"));
            for (const auto &c : columns)
                if (std::find(header.begin(), header.end(), c) == header.end() && !header.empty())
                    throw std::invalid_argument("unknown column");
            Json rows = Json::array(), indices = Json::array();
            uint64_t baseRow = 0;
            for (const auto &s : cache.segments)
            {
                if (position >= baseRow + s.rows)
                {
                    baseRow += s.rows;
                    continue;
                }
                std::ifstream in(s.path, std::ios::binary);
                auto local = position - baseRow;
                const auto block = local / 256;
                in.seekg(std::streamoff(s.offsets.at(size_t(block))));
                std::string line;
                for (auto skip = block * 256; skip < local; ++skip)
                    std::getline(in, line);
                for (; local < s.rows && rows.size() < limit; ++local)
                {
                    std::getline(in, line);
                    const auto values = Split(line);
                    if (values.size() != header.size())
                        throw std::runtime_error("malformed complete CSV row");
                    Json row = Json::object();
                    for (size_t j = 0; j < header.size(); ++j)
                    {
                        try
                        {
                            row[header[j]] = number(values[j]);
                        }
                        catch (...)
                        {
                            row[header[j]] = nullptr;
                        }
                    }
                    const auto ordinal = baseRow + local;
                    position = ordinal + 1;
                    const int status = kind == "nonlinear" ? 0 : kind == "stage" ? 1
                                                                                 : 2;
                    if (kind != "all" && row.value("status", Json()) != status)
                        continue;
                    if ((std::isfinite(lo) || std::isfinite(hi)) &&
                        (!row.contains(axis) || !row[axis].is_number() || row[axis].get<double>() < lo || row[axis].get<double>() > hi))
                        continue;
                    Json selected = Json::object();
                    for (const auto &c : columns)
                        selected[c] = row[c];
                    rows.push_back(std::move(selected));
                    indices.push_back(ordinal);
                }
                if (rows.size() == limit)
                    break;
                baseRow += s.rows;
            }
            return {{"schema", "dndsr.cauchy.history-query.v1"}, {"source", "csv"}, {"header", columns}, {"rows", rows}, {"indices", indices}, {"totalRows", total}, {"returnedRows", rows.size()}, {"generation", std::to_string(cache.generation)}, {"reset", reset}, {"hasMore", position < total}, {"nextCursor", std::to_string(cache.generation) + ":" + std::to_string(position)}};
        }
    };
}
