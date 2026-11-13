#include "Cauchy/Interface/InterfacePacketOperator.hpp"
#include "Cauchy/Interface/CPU/InterfacePacketKernels.hpp"
#include "DNDS/Runtime/Execution/EagerArrayBatch.hpp"
#include "DNDS/Algebra/DofAlgebra.hpp"
#ifdef DNDS_USE_CUDA
#    include "Cauchy/Interface/CUDA/InterfacePacketKernels.hpp"
#endif
#include <algorithm>

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    struct InterfacePacketOperator::Data
    {
        R::Array<InterfacePacketRow> rows;
        std::vector<R::MetadataBuffer<index>> components;
        R::MetadataBuffer<index> vectorStarts;
        index width, quadratureRows, faceRows;
        R::Device device;
    };
    InterfacePacketOperator::InterfacePacketOperator(const R::Array<InterfacePacketRow> &hostRows,
                                                     std::vector<R::MetadataBuffer<index>> components, R::MetadataBuffer<index> vectors,
                                                     index width, index quadratureRows, index faceRows, R::ExecutionContext context)
    {
        DNDS_check_throw_info(hostRows.GetDevice() == R::Device::Host() && width > 0 && quadratureRows >= 0 && faceRows >= 0 &&
                                  !components.empty() && vectors.GetDevice() == R::Device::Host(),
                              "invalid interface packet setup inputs");
        std::vector<index> ordinals;
        for (const auto &field : components)
        {
            DNDS_check_throw_info(field.GetDevice() == R::Device::Host() && !field.empty(), "interface component setup requires host metadata");
            ordinals.insert(ordinals.end(), field.begin(), field.end());
        }
        std::sort(ordinals.begin(), ordinals.end());
        DNDS_check_throw_info(index(ordinals.size()) == width, "interface mapping must supply every component");
        for (index j = 0; j < width; ++j)
            DNDS_check_throw_info(ordinals[size_t(j)] == j, "interface component mapping is not a permutation");
        std::vector<index> vectorComponents;
        for (const index first : vectors)
        {
            DNDS_check_throw_info(width >= 3 && first >= 0 && first <= width - 3, "interface vector block is out of bounds");
            for (int d = 0; d < 3; ++d)
                vectorComponents.push_back(first + d);
        }
        std::sort(vectorComponents.begin(), vectorComponents.end());
        DNDS_check_throw_info(std::adjacent_find(vectorComponents.begin(), vectorComponents.end()) == vectorComponents.end(), "interface vector blocks overlap");
        std::vector<std::pair<bool, index>> destinations;
        {
            const auto read = hostRows.Read();
            const auto rows = read.View<R::FixedRows<1>>();
            destinations.reserve(size_t(rows.RowCount()));
            for (index iRow = 0; iRow < rows.RowCount(); ++iRow)
            {
                const auto row = rows(iRow, 0);
                DNDS_check_throw_info(row.row >= 0 && row.row < (row.faceCenter ? faceRows : quadratureRows), "interface destination row is out of bounds");
                destinations.emplace_back(row.faceCenter, row.row);
            }
        }
        std::sort(destinations.begin(), destinations.end());
        DNDS_check_throw_info(std::adjacent_find(destinations.begin(), destinations.end()) == destinations.end(), "interface packet has duplicate destinations");
        DNDS_check_throw_info(context.device == R::Device::Host() || (context.device.space == R::MemorySpace::CUDA && components.size() == 1), "CUDA interface packets require combined donor storage");
        // Snapshot mapping rows, rather than retaining mutable caller aliases.
        R::Array<InterfacePacketRow> rows(hostRows.Layout(), context.device);
        rows.CopyFrom(hostRows, context).Wait();
        for (auto &field : components)
            field = field.CopyTo(context.device, context);
        vectors = vectors.CopyTo(context.device, context);
        _data = std::make_shared<Data>(Data{std::move(rows), std::move(components), std::move(vectors), width, quadratureRows, faceRows, context.device});
    }
    void InterfacePacketOperator::Submit(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                         const std::vector<R::Array<real>> &values, R::Array<real> &quadrature, R::Array<real> &faceCenters,
                                         InterfacePacketRotation rotation) const
    {
        DNDS_check_throw_info(stream.Context().device == _data->device && values.size() == _data->components.size() &&
                                  quadrature.RowCount() == _data->quadratureRows && faceCenters.RowCount() == _data->faceRows,
                              "interface packet operands or stream differ from preparation");
        std::vector<R::ArrayOperand<real, false, R::UniformTableRows>> inputs;
        for (index iField = 0; iField < index(values.size()); ++iField)
        {
            const auto &field = values[size_t(iField)];
            DNDS_check_throw_info(field.GetDevice() == _data->device && field.RowCount() == _data->rows.RowCount() &&
                                      !field.Layout().IsVariable() && field.Layout().Width() == index(_data->components[size_t(iField)].size()) &&
                                      (_data->device == R::Device::Host() || field.Layout().Kind() == R::RowLayoutKind::Table),
                                  "interface donor field shape/placement differs");
            inputs.push_back(R::ReadAs<R::UniformTableRows>(field));
        }
        for (const auto &output : {quadrature, faceCenters})
        {
            DNDS_check_throw_info(output.GetDevice() == _data->device && !output.Layout().IsVariable() && output.Layout().Width() == _data->width &&
                                      (_data->device == R::Device::Host() || output.Layout().Kind() == R::RowLayoutKind::Table),
                                  "interface receiver shape/placement differs");
            for (const auto &field : values)
                DNDS_check_throw_info(!Algebra::Detail::Overlaps(output.Storage(), field.Storage()), "interface input/output overlap");
        }
        DNDS_check_throw_info(!Algebra::Detail::Overlaps(quadrature.Storage(), faceCenters.Storage()), "interface packet outputs overlap");
        auto rows = std::make_shared<R::Array<InterfacePacketRow>::ReadLease>(_data->rows.Read(stream.Context()));
        auto metadata = std::make_shared<std::vector<R::ReadAccess>>();
        for (const auto &field : _data->components)
            metadata->push_back(field.Read(stream.Context()));
        metadata->push_back(_data->vectorStarts.Read(stream.Context()));
        // SubmitArrayBatch retains the captured scopes and Data until this
        // operation retires. No per-iteration engine-wide retention list.
        std::vector<R::ArrayOperand<real, true, R::UniformTableRows>> outputs{
            R::ReadWriteAs<R::UniformTableRows>(quadrature), R::ReadWriteAs<R::UniformTableRows>(faceCenters)};
        R::SubmitArrayBatch(engine, stream, [data = _data, rows, metadata, rotation](R::ExecutionContext context, auto inputs, auto outputs)
                            {
                                if (context.device == R::Device::Host())
                                {
                                    std::vector<CPU::InterfacePacketFieldView> fields;
                                    for (index iField = 0; iField < inputs.size(); ++iField)
                                        fields.push_back({inputs[iField], data->components[size_t(iField)].HostView()});
                                    std::vector<Algebra::DofComponentColumn<real>> quadrature, faces;
                                    for (index j = 0; j < data->width; ++j)
                                    {
                                        quadrature.push_back({outputs[0], {}, j});
                                        faces.push_back({outputs[1], {}, j});
                                    }
                                    CPU::InterfacePacketRequest request;
                                    request.rows = rows->View<R::FixedRows<1>>();
                                    request.fields = {fields.data(), index(fields.size())};
                                    request.vectorStarts = data->vectorStarts.HostView();
                                    request.quadrature = {{quadrature.data(), index(quadrature.size())}, outputs[0].RowCount(), 0, data->width};
                                    request.faceCenters = {{faces.data(), index(faces.size())}, outputs[1].RowCount(), 0, data->width};
                                    request.rotation = rotation;
                                    CPU::FillInterfacePacket(request);
                                }
#ifdef DNDS_USE_CUDA
                                else
                                {
                                    InterfacePacketTableView request;
                                    request.rows = rows->View<R::FixedRows<1>>();
                                    request.values = inputs[0];
                                    request.quadrature = outputs[0];
                                    request.faceCenters = outputs[1];
                                    request.receiverComponents = {metadata->front().View<index>().data, index(data->components[0].size())};
                                    request.vectorStarts = {metadata->back().View<index>().data, index(data->vectorStarts.size())};
                                    request.rotation = rotation;
                                    CUDA::FillInterfacePacket(context, request);
                                }
#else
                                else
                                    DNDS_check_throw_info(false, "CUDA interface packet kernel is unavailable");
#endif
                            },
                            std::move(inputs), std::move(outputs));
    }
}
