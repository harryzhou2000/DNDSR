#pragma once
/// @file CsvLog.hpp
/// @brief Typed schema/row CSV writer with automatic segmented output.

#include <iostream>
#include <iomanip>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "DNDS/Core/Defines.hpp"
#include "DNDS/IO/OutputDir.hpp"

namespace DNDS
{

    enum class CsvColumnType
    {
        Integer,
        Real,
        Text
    };

    struct CsvColumn
    {
        std::string name;
        CsvColumnType type = CsvColumnType::Text;
    };

    /// Immutable ordered CSV schema. Column names are unique and rows built
    /// from one schema are checked by CsvLog before serialization.
    class CsvSchema
    {
        std::vector<CsvColumn> _columns;
        std::map<std::string, size_t> _indices;

    public:
        CsvSchema() = default;
        explicit CsvSchema(std::vector<CsvColumn> columns) : _columns(std::move(columns))
        {
            DNDS_check_throw_info(!_columns.empty(), "CSV schema needs at least one column");
            for (size_t i = 0; i < _columns.size(); ++i)
                DNDS_check_throw_info(!_columns[i].name.empty() && _indices.emplace(_columns[i].name, i).second,
                                      "CSV schema has an empty or duplicate column");
        }
        explicit CsvSchema(const std::vector<std::string> &titles)
        {
            _columns.reserve(titles.size());
            for (const auto &title : titles)
                _columns.push_back({title, CsvColumnType::Text});
            DNDS_check_throw_info(!_columns.empty(), "CSV schema needs at least one column");
            for (size_t i = 0; i < _columns.size(); ++i)
                DNDS_check_throw_info(!_columns[i].name.empty() && _indices.emplace(_columns[i].name, i).second,
                                      "CSV schema has an empty or duplicate column");
        }
        const std::vector<CsvColumn> &Columns() const { return _columns; }
        size_t Index(const std::string &name) const
        {
            const auto found = _indices.find(name);
            DNDS_check_throw_info(found != _indices.end(), "unknown CSV column " + name);
            return found->second;
        }
    };

    using CsvCell = std::variant<std::monostate, int64_t, double, std::string>;

    /// Schema-bound typed row. Unset cells remain blank on disk. Set validates
    /// the declared column type before the row reaches any file stream.
    class CsvRow
    {
        friend class CsvLog;
        std::shared_ptr<const CsvSchema> _schema;
        std::vector<CsvCell> _cells;
        explicit CsvRow(std::shared_ptr<const CsvSchema> schema)
            : _schema(std::move(schema)), _cells(_schema->Columns().size()) {}
        const CsvColumn &Column(const std::string &name, size_t &index) const
        {
            index = _schema->Index(name);
            return _schema->Columns()[index];
        }

    public:
        template <class T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
        void Set(const std::string &name, T value)
        {
            size_t i = 0;
            const auto &column = Column(name, i);
            DNDS_check_throw_info(column.type == CsvColumnType::Integer, "CSV integer assigned to non-integer column " + name);
            if constexpr (std::is_unsigned_v<T>)
                DNDS_check_throw_info(value <= T(std::numeric_limits<int64_t>::max()), "CSV unsigned integer exceeds int64 range");
            _cells[i] = int64_t(value);
        }
        template <class T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
        void Set(const std::string &name, T value)
        {
            size_t i = 0;
            const auto &column = Column(name, i);
            DNDS_check_throw_info(column.type == CsvColumnType::Real, "CSV real assigned to non-real column " + name);
            _cells[i] = double(value);
        }
        void Set(const std::string &name, std::string value)
        {
            size_t i = 0;
            const auto &column = Column(name, i);
            DNDS_check_throw_info(column.type == CsvColumnType::Text, "CSV text assigned to non-text column " + name);
            _cells[i] = std::move(value);
        }
        void Clear(const std::string &name) { _cells[_schema->Index(name)] = std::monostate{}; }
    };

    /**
     * @brief Storage-agnostic "either int or double" value with CSV-friendly streaming.
     *
     * @details Each instance holds either an integer (`i`) or a double (`d`);
     * the unused field is set to the @ref UnInitIndex / @ref UnInitReal sentinel. The
     * assignment operator dispatches on the source type, and `operator<<`
     * prints whichever field is live. Designed for columns where some rows
     * are integer counts and others are floating residuals.
     */
    struct LogSimpleDIValue
    {
        int64_t i{UnInitIndex}; // UnInitIndex for double
        double d{0};

        template <class T>
        std::enable_if_t<std::is_integral_v<T>, LogSimpleDIValue &> operator=(T v)
        {
            i = v;
            if (i == UnInitIndex)
                d = UnInitReal;
            return *this;
        }

        template <class T>
        std::enable_if_t<!std::is_integral_v<T>, LogSimpleDIValue &> operator=(T v)
        {
            d = v;
            i = UnInitIndex;
            return *this;
        }

        friend std::ostream &operator<<(std::ostream &o, const LogSimpleDIValue &v)
        {
            if (v.i == UnInitIndex)
                o << v.d;
            else
                o << v.i;
            return o;
        }
    };

    /**
     * @brief Append-only CSV logger with automatic file rotation after a line-count limit.
     *
     * @details One instance writes rows to a file, using a caller-provided
     * ordered list of column titles. After `n_line_max` rows the output is
     * rotated to `<bare>_<block>.suffix` to keep any single file size bounded.
     *
     * Typical use: one @ref DNDS::CsvLog "CsvLog" per solver per rank-0 process, recording
     * per-iteration diagnostics with @ref DNDS::LogSimpleDIValue "LogSimpleDIValue" entries keyed by title.
     */
    class CsvLog
    {
        std::shared_ptr<const CsvSchema> schema;
        std::unique_ptr<std::ostream> pOs;
        int64_t n_line{0};
        std::string log_name_bare;
        std::string log_name_suffix;

        std::string delim = ",";
        int iBlock = 0;
        int64_t n_line_max = INT64_MAX;
        void update_ofstream(const std::string &fname)
        {
            createOutputDir(fname);
            pOs = std::make_unique<std::ofstream>(fname);
            DNDS_check_throw_info(*pOs, "csv file [" + fname + "] did not open");
        }
        void ensure_ofstream()
        {
            if (pOs)
                return;
            DNDS_check_throw_info(log_name_bare.length(), "need to give a log name");
            update_ofstream(log_name_bare + (iBlock ? "_" + std::to_string(iBlock) : "") + log_name_suffix);
        }
        static void WriteText(std::ostream &output, const std::string &value)
        {
            const bool quote = value.find_first_of(",\"\r\n") != std::string::npos;
            if (!quote)
            {
                output << value;
                return;
            }
            output << '"';
            for (const char c : value)
                output << c << (c == '"' ? "\"" : "");
            output << '"';
        }

    public:
        /// @brief Construct a logger with externally-provided titles and output stream.
        template <class T_titles>
        CsvLog(T_titles &&n_titles, std::unique_ptr<std::ostream> n_pOs)
            : schema(std::make_shared<CsvSchema>(std::vector<std::string>(std::forward<T_titles>(n_titles)))), pOs(std::move(n_pOs)) {}

        CsvLog(CsvSchema n_schema, std::unique_ptr<std::ostream> n_pOs)
            : schema(std::make_shared<CsvSchema>(std::move(n_schema))), pOs(std::move(n_pOs)) {}

        /// @brief Construct a logger that creates / manages its own output files.
        /// @param n_titles          Column titles (ordered).
        /// @param n_log_name_bare   Base path without suffix, e.g. "out/run".
        /// @param n_log_name_suffix Filename suffix including dot (default ".csv").
        /// @param n_n_line_max      Rotate to a new file every this many lines.
        template <class T_titles>
        CsvLog(T_titles &&n_titles,
               const std::string &n_log_name_bare,
               const std::string &n_log_name_suffix = ".csv",
               int64_t n_n_line_max = 10000)
            : schema(std::make_shared<CsvSchema>(std::vector<std::string>(std::forward<T_titles>(n_titles)))),
              log_name_bare(n_log_name_bare),
              log_name_suffix(n_log_name_suffix),
              n_line_max(n_n_line_max)
        {
            DNDS_check_throw_info(n_line_max > 0, "CSV rotation length must be positive");
            update_ofstream(log_name_bare + n_log_name_suffix);
        }

        CsvLog(CsvSchema n_schema, const std::string &n_log_name_bare,
               const std::string &n_log_name_suffix = ".csv", int64_t n_n_line_max = 10000)
            : schema(std::make_shared<CsvSchema>(std::move(n_schema))), log_name_bare(n_log_name_bare),
              log_name_suffix(n_log_name_suffix), n_line_max(n_n_line_max)
        {
            DNDS_check_throw_info(n_line_max > 0, "CSV rotation length must be positive");
            update_ofstream(log_name_bare + n_log_name_suffix);
        }

        CsvRow MakeRow() const { return CsvRow(schema); }
        const CsvSchema &Schema() const { return *schema; }

        void WriteLine(const CsvRow &row, int nPrecision = 17)
        {
            DNDS_check_throw_info(row._schema == schema, "CSV row belongs to another schema");
            ensure_ofstream();
            if (n_line == 0)
                WriteTitle();
            (*pOs) << std::setprecision(nPrecision) << std::scientific;
            for (size_t i = 0; i < row._cells.size(); ++i)
            {
                if (i)
                    (*pOs) << ',';
                const auto &cell = row._cells[i];
                if (const auto value = std::get_if<int64_t>(&cell))
                    (*pOs) << *value;
                else if (const auto value = std::get_if<double>(&cell))
                    (*pOs) << *value;
                else if (const auto value = std::get_if<std::string>(&cell))
                    WriteText(*pOs, *value);
            }
            (*pOs) << '\n';
            RotateIfNeeded();
        }

        /// @brief Append one CSV row by looking up each column in `title_to_value`.
        /// @tparam TMap  Map-like supporting `operator[](std::string)` returning
        ///               something streamable.
        /// @param nPrecision Decimal digits used for floating-point output.
        template <class TMap>
        void WriteLine(TMap &&title_to_value, int nPrecision)
        {
            ensure_ofstream();
            if (n_line == 0)
                WriteTitle();
            (*pOs) << std::setprecision(nPrecision) << std::scientific;
            const auto &columns = schema->Columns();
            for (size_t i = 0; i < columns.size(); i++)
                (*pOs) << title_to_value[columns[i].name] << ((i == (columns.size() - 1)) ? "" : ",");
            (*pOs) << '\n';
            RotateIfNeeded();
        }

        /// @brief Emit the title row (called automatically on the first @ref WriteLine).
        void WriteTitle()
        {
            ensure_ofstream();
            const auto &columns = schema->Columns();
            for (size_t i = 0; i < columns.size(); i++)
            {
                WriteText(*pOs, columns[i].name);
                (*pOs) << ((i == (columns.size() - 1)) ? "" : ",");
            }
            (*pOs) << '\n';
        }

        /// @brief Flush the current segment. A completed exact-size segment has
        /// no open stream, so this never creates a trailing empty file.
        void Flush()
        {
            if (!pOs)
                return;
            pOs->flush();
            DNDS_check_throw_info(*pOs, "csv file flush failed");
        }

    private:
        void RotateIfNeeded()
        {
            if (++n_line < n_line_max)
                return;
            pOs->flush();
            DNDS_check_throw_info(*pOs, "csv segment flush failed");
            pOs.reset();
            n_line = 0;
            ++iBlock;
        }
    };

    /// @brief Convenience alias: the `title -> value` map type passed to @ref DNDS::CsvLog "CsvLog"::WriteLine.
    using tLogSimpleDIValueMap = std::map<std::string, LogSimpleDIValue>;
}
