#include "Cauchy/IO/RunControl.hpp"

#include <algorithm>

namespace DNDS::Cauchy
{
    void RunControl::RetainHistory(index domain, const CsvRow &row)
    {
        HistoryEntry entry{domain, row};
        std::lock_guard<std::mutex> lock(_historyMutex);
        _history.push_back(std::move(entry));
        if (_history.size() > HistoryCapacity)
            _history.pop_front();
    }

    Json RunControl::CurrentHistory(size_t limit) const
    {
        size_t retained;
        std::vector<HistoryEntry> snapshot;
        {
            std::lock_guard<std::mutex> lock(_historyMutex);
            retained = _history.size();
            const auto count = std::min(limit, retained);
            snapshot.assign(_history.end() - std::ptrdiff_t(count), _history.end());
        }
        Json records = Json::array();
        for (const auto &entry : snapshot)
        {
            Json values = Json::object();
            const auto &columns = entry.row.Columns();
            const auto &cells = entry.row.Cells();
            for (size_t i = 0; i < cells.size(); ++i)
                std::visit([&](const auto &value)
                           {
                    using T = std::decay_t<decltype(value)>;
                    if constexpr (!std::is_same_v<T, std::monostate>)
                        values[columns[i].name] = value; }, cells[i]);
            records.push_back({{"domain", entry.domain}, {"values", std::move(values)}});
        }
        return {{"rank", _rank}, {"capacity", HistoryCapacity}, {"retained", retained}, {"records", std::move(records)}};
    }
}
