#include "Cauchy/Implicit/LinearPattern.hpp"

namespace DNDS::Cauchy
{
    namespace
    {
        Runtime::Array<index> Publish(const std::vector<index> &values)
        {
            Runtime::Array<index> result(Runtime::RowLayout::Dense(values.size(), 1));
            {
                auto write = result.Write();
                const auto view = write.View<Runtime::FixedRows<1>>();
                for (index iRow = 0; iRow < index(values.size()); ++iRow)
                    view(iRow, 0) = values[iRow];
            }
            return result;
        }
        std::vector<index> Copy(const Runtime::Array<index> &values)
        {
            const auto read = values.Read();
            const auto view = read.View<Runtime::FixedRows<1>>();
            std::vector<index> result(view.RowCount());
            for (index iRow = 0; iRow < view.RowCount(); ++iRow)
                result[iRow] = view(iRow, 0);
            return result;
        }
    }
    LinearPatternStorage::LinearPatternStorage(
        const std::vector<std::vector<index>> &columns,
        const std::vector<index> &order, const std::vector<index> &position,
        const std::vector<index> &core, const std::vector<index> &dependent)
        : _order(Publish(order)), _position(Publish(position)),
          _core(Publish(core)), _dependent(Publish(dependent))
    {
        std::vector<size_t> lengths;
        lengths.reserve(columns.size());
        for (const auto &row : columns)
            lengths.push_back(row.size());
        _columns = Runtime::Array<index>(Runtime::RowLayout::Ragged(lengths));
        auto write = _columns.Write();
        const auto view = write.View<Runtime::CSRRows>();
        for (index iRow = 0; iRow < index(columns.size()); ++iRow)
            for (index j = 0; j < index(columns[iRow].size()); ++j)
                view(iRow, j) = columns[iRow][j];
    }
    std::vector<std::vector<index>> LinearPatternStorage::CopyColumns() const
    {
        const auto read = _columns.Read();
        const auto view = read.View<Runtime::CSRRows>();
        std::vector<std::vector<index>> result(view.RowCount());
        for (index iRow = 0; iRow < view.RowCount(); ++iRow)
            for (const auto column : view[iRow])
                result[iRow].push_back(column);
        return result;
    }
    std::vector<index> LinearPatternStorage::CopyOrder() const { return Copy(_order); }
    std::vector<index> LinearPatternStorage::CopyCore() const { return Copy(_core); }
    std::vector<index> LinearPatternStorage::CopyDependent() const { return Copy(_dependent); }
}
