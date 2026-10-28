#pragma once

#include "DofField.hpp"
#include "DNDS/Runtime/EagerArray.hpp"
#include <cmath>
#include <array>
#include <functional>
#include <limits>
#include <optional>

namespace DNDS::Algebra
{
    namespace Detail
    {
        enum class Pointwise
        {
            Fill,
            Copy,
            Scale,
            Axpy,
            Combination
        };
        enum class Reduction
        {
            Dot,
            Sum,
            Min,
            Max,
            Squares
        };
        void ApplyCUDA(Runtime::ExecutionContext, Pointwise, Runtime::ArrayView<float>, Runtime::ArrayView<const float>, Runtime::ArrayView<const float>, double, double);
        void ApplyCUDA(Runtime::ExecutionContext, Pointwise, Runtime::ArrayView<double>, Runtime::ArrayView<const double>, Runtime::ArrayView<const double>, double, double);
        void ReduceCUDA(Runtime::ExecutionContext, Reduction, Runtime::ArrayView<const float>, Runtime::ArrayView<const float>, Runtime::ArrayView<double>);
        void ReduceCUDA(Runtime::ExecutionContext, Reduction, Runtime::ArrayView<const double>, Runtime::ArrayView<const double>, Runtime::ArrayView<double>);

        template <class T>
        void Apply(Runtime::ExecutionContext context, Pointwise operation, Runtime::ArrayView<T> output,
                   Runtime::ArrayView<const T> x = {}, Runtime::ArrayView<const T> y = {}, double a = 0, double b = 0)
        {
            if (context.device.space == Runtime::MemorySpace::CUDA)
            {
                ApplyCUDA(context, operation, output, x, y, a, b);
                return;
            }
            for (size_t row = 0; row < output.Rows(); ++row)
                for (size_t component = 0; component < output.RowSize(row); ++component)
                {
                    switch (operation)
                    {
                    case Pointwise::Fill:
                        output(row, component) = T(a);
                        break;
                    case Pointwise::Copy:
                        output(row, component) = x(row, component);
                        break;
                    case Pointwise::Scale:
                        output(row, component) *= T(a);
                        break;
                    case Pointwise::Axpy:
                        output(row, component) += T(a) * x(row, component);
                        break;
                    case Pointwise::Combination:
                    {
                        const auto left = x(row, component), right = y(row, component);
                        output(row, component) = T(a) * left + T(b) * right;
                        break;
                    }
                    }
                }
        }
        template <class T>
        void Reduce(Runtime::ExecutionContext context, Reduction operation, Runtime::ArrayView<const T> x,
                    Runtime::ArrayView<const T> y, Runtime::ArrayView<double> result)
        {
            if (context.device.space == Runtime::MemorySpace::CUDA)
            {
                ReduceCUDA(context, operation, x, y, result);
                return;
            }
            for (size_t component = 0; component < result.width; ++component)
                result(0, component) = operation == Reduction::Min ? std::numeric_limits<double>::infinity() : operation == Reduction::Max ? -std::numeric_limits<double>::infinity()
                                                                                                                                           : 0;
            for (size_t row = 0; row < x.Rows(); ++row)
                for (size_t component = 0; component < x.width; ++component)
                {
                    const double value = x(row, component);
                    if (operation == Reduction::Dot)
                        result(0, 0) += value * double(y(row, component));
                    else if (operation == Reduction::Sum)
                        result(0, component) += value;
                    else if (operation == Reduction::Squares)
                        result(0, component) += value * value;
                    else if (operation == Reduction::Min)
                        result(0, component) = std::fmin(result(0, component), value);
                    else
                        result(0, component) = std::fmax(result(0, component), value);
                }
        }
        template <class Field>
        void Check(const Field &field, const Runtime::ExecutionStream &stream)
        {
            field.Validate();
            DNDS_check_throw_info(field.Device() == stream.Context().device, "algebra stream and field placement differ");
        }
        template <class A, class B>
        void Pair(const A &a, const B &b, bool writing = false)
        {
            a.Validate();
            b.Validate();
            DNDS_check_throw_info(a.Space().Same(b.Space()) && a.Width() == b.Width() && a.Device() == b.Device(),
                                  "algebra operand distribution, component width or placement differs");
            if (writing)
                DNDS_check_throw_info(!Overlaps(a.Owned().Storage(), b.Ghost().Storage()), "algebra output overlaps an operand's ghost storage");
            if (writing && Overlaps(a.Owned().Storage(), b.Owned().Storage()))
                DNDS_check_throw_info(a.Owned().Storage().LegacyPointer() == b.Owned().Storage().LegacyPointer() &&
                                          a.Owned().Layout() == b.Owned().Layout(),
                                      "algebra partial overlap is unsupported");
        }
        template <class F>
        void Immediate(F &&run)
        {
            Runtime::EagerEngine engine;
            const auto stream = engine.CreateCPUStream();
            run(engine, stream);
            engine.Synchronize();
        }
        template <class Field>
        void RequireIdle(const Field &field)
        {
            field.Validate();
            DNDS_check_throw_info(field.Owned().Storage().IsIdle(), "immediate algebra requires idle owned storage; finish pending accesses first");
        }
        struct GlobalSchema
        {
            size_t width, fieldWidth, scalarBytes;
            int operation, protocol;
        };
        std::vector<double> Aggregate(const DofSpace &, const std::function<GlobalSchema()> &schema,
                                      const std::function<std::vector<double>()> &local);
    }

    /// Explicit result owner. Get waits this submitted prefix before exposing
    /// host values; it is separate from ordinary mutation command submission.
    class ComponentResult
    {
        Runtime::Array<double, Runtime::TableRows> _values;
        Runtime::StreamEvent _ready;

    public:
        ComponentResult(Runtime::Array<double, Runtime::TableRows> values, Runtime::StreamEvent ready)
            : _values(std::move(values)), _ready(std::move(ready)) {}
        std::vector<double> Get() const
        {
            _ready.WaitHost();
            const auto host = _values.GetDevice() == Runtime::Device::Host() ? _values : _values.To(Runtime::Device::Host());
            const auto read = host.Read();
            return {read.View().Row(0), read.View().Row(0) + host.Layout().Width()};
        }
    };
    class ScalarResult
    {
        ComponentResult _values;
        bool _sqrt;

    public:
        ScalarResult(ComponentResult values, bool squareRoot = false) : _values(std::move(values)), _sqrt(squareRoot) {}
        double Get() const
        {
            const auto value = _values.Get()[0];
            return _sqrt ? std::sqrt(value) : value;
        }
        ScalarResult SquareRoot() const
        {
            auto result = *this;
            result._sqrt = true;
            return result;
        }
    };

    template <class T, size_t N>
    void Fill(Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, DofField<T, N> &output, T value)
    {
        Detail::Check(output, stream);
        Runtime::SubmitArrays(engine, stream, [partition = output.Space().Pin(), value](Runtime::ExecutionContext context, auto out)
                              { Detail::Apply(context, Detail::Pointwise::Fill, out.DynamicView(), {}, {}, value); }, Runtime::Write(output.Owned()));
    }
    template <class T, size_t N>
    void Scale(Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, DofField<T, N> &output, T value)
    {
        Detail::Check(output, stream);
        Runtime::SubmitArrays(engine, stream, [partition = output.Space().Pin(), value](Runtime::ExecutionContext context, auto out)
                              { Detail::Apply(context, Detail::Pointwise::Scale, out.DynamicView(), {}, {}, value); }, Runtime::ReadWrite(output.Owned()));
    }
    template <class T, size_t A, size_t B>
    void Copy(Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, DofField<T, A> &output, const DofField<T, B> &input)
    {
        Detail::Check(output, stream);
        Detail::Pair(output, input, true);
        Runtime::SubmitArrays(engine, stream, [partition = output.Space().Pin()](Runtime::ExecutionContext context, auto x, auto out)
                              { Detail::Apply(context, Detail::Pointwise::Copy, out.DynamicView(), x.DynamicView()); }, Runtime::Read(input.Owned()), Runtime::Write(output.Owned()));
    }
    template <class T, size_t A, size_t B>
    void Axpy(Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, DofField<T, A> &output, T alpha, const DofField<T, B> &input)
    {
        Detail::Check(output, stream);
        Detail::Pair(output, input, true);
        Runtime::SubmitArrays(engine, stream, [partition = output.Space().Pin(), alpha](Runtime::ExecutionContext context, auto x, auto out)
                              { Detail::Apply(context, Detail::Pointwise::Axpy, out.DynamicView(), x.DynamicView(), {}, alpha); }, Runtime::Read(input.Owned()), Runtime::ReadWrite(output.Owned()));
    }
    template <class T, size_t A, size_t B, size_t C>
    void LinearCombination(Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, DofField<T, A> &output,
                           T alpha, const DofField<T, B> &x, T beta, const DofField<T, C> &y)
    {
        Detail::Check(output, stream);
        Detail::Pair(output, x, true);
        Detail::Pair(output, y, true);
        Runtime::SubmitArrays(engine, stream, [partition = output.Space().Pin(), alpha, beta](Runtime::ExecutionContext context, auto a, auto b, auto out)
                              { Detail::Apply(context, Detail::Pointwise::Combination, out.DynamicView(), a.DynamicView(), b.DynamicView(), alpha, beta); }, Runtime::Read(x.Owned()), Runtime::Read(y.Owned()), Runtime::Write(output.Owned()));
    }

    template <class T, size_t N>
    void Fill(DofField<T, N> &out, T value)
    {
        Detail::RequireIdle(out);
        Detail::Immediate([&](auto &e, const auto &s)
                          { Fill(e, s, out, value); });
    }
    template <class T, size_t N>
    void Scale(DofField<T, N> &out, T value)
    {
        Detail::RequireIdle(out);
        Detail::Immediate([&](auto &e, const auto &s)
                          { Scale(e, s, out, value); });
    }
    template <class T, size_t A, size_t B>
    void Copy(DofField<T, A> &out, const DofField<T, B> &in)
    {
        Detail::RequireIdle(out);
        Detail::RequireIdle(in);
        Detail::Immediate([&](auto &e, const auto &s)
                          { Copy(e, s, out, in); });
    }
    template <class T, size_t A, size_t B>
    void Axpy(DofField<T, A> &out, T alpha, const DofField<T, B> &in)
    {
        Detail::RequireIdle(out);
        Detail::RequireIdle(in);
        Detail::Immediate([&](auto &e, const auto &s)
                          { Axpy(e, s, out, alpha, in); });
    }
    template <class T, size_t A, size_t B, size_t C>
    void LinearCombination(DofField<T, A> &out, T a, const DofField<T, B> &x, T b, const DofField<T, C> &y)
    {
        Detail::RequireIdle(out);
        Detail::RequireIdle(x);
        Detail::RequireIdle(y);
        Detail::Immediate([&](auto &e, const auto &s)
                          { LinearCombination(e, s, out, a, x, b, y); });
    }

    template <class T, size_t A, size_t B>
    ScalarResult DotLocal(Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, const DofField<T, A> &x, const DofField<T, B> &y)
    {
        Detail::Check(x, stream);
        Detail::Pair(x, y);
        Runtime::Array<double, Runtime::TableRows> result(Runtime::RowLayout::Dense(1, 1), x.Device());
        Runtime::SubmitArrays(engine, stream, [partition = x.Space().Pin()](Runtime::ExecutionContext context, auto a, auto b, auto out)
                              { Detail::Reduce(context, Detail::Reduction::Dot, a.DynamicView(), b.DynamicView(), out.DynamicView()); }, Runtime::Read(x.Owned()), Runtime::Read(y.Owned()), Runtime::Write(result));
        return ScalarResult(ComponentResult(std::move(result), stream.Record()));
    }
    template <class T, size_t N>
    ScalarResult NormLocal(Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, const DofField<T, N> &x)
    {
        auto dot = DotLocal(engine, stream, x, x);
        // Retain the dot result through the scalar transformation's Get call.
        return dot.SquareRoot();
    }
    template <class T, size_t N>
    ComponentResult ComponentsLocal(Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream,
                                    const DofField<T, N> &x, Detail::Reduction operation)
    {
        Detail::Check(x, stream);
        Runtime::Array<double, Runtime::TableRows> result(Runtime::RowLayout::Dense(1, x.Width()), x.Device());
        Runtime::SubmitArrays(engine, stream, [partition = x.Space().Pin(), operation](Runtime::ExecutionContext context, auto a, auto out)
                              { Detail::Reduce(context, operation, a.DynamicView(), {}, out.DynamicView()); }, Runtime::Read(x.Owned()), Runtime::Write(result));
        return ComponentResult(std::move(result), stream.Record());
    }
    template <class T, size_t N>
    ComponentResult ComponentSumLocal(Runtime::EagerEngine &e, const Runtime::ExecutionStream &s, const DofField<T, N> &x) { return ComponentsLocal(e, s, x, Detail::Reduction::Sum); }
    template <class T, size_t N>
    ComponentResult ComponentMinLocal(Runtime::EagerEngine &e, const Runtime::ExecutionStream &s, const DofField<T, N> &x) { return ComponentsLocal(e, s, x, Detail::Reduction::Min); }
    template <class T, size_t N>
    ComponentResult ComponentMaxLocal(Runtime::EagerEngine &e, const Runtime::ExecutionStream &s, const DofField<T, N> &x) { return ComponentsLocal(e, s, x, Detail::Reduction::Max); }
    template <class T, size_t A, size_t B>
    double DotLocal(const DofField<T, A> &x, const DofField<T, B> &y)
    {
        Detail::RequireIdle(x);
        Detail::RequireIdle(y);
        double value = 0;
        Detail::Immediate([&](auto &e, const auto &s)
                          { value = DotLocal(e, s, x, y).Get(); });
        return value;
    }
    template <class T, size_t N>
    double NormLocal(const DofField<T, N> &x) { return std::sqrt(DotLocal(x, x)); }
    template <class T, size_t N>
    std::vector<double> ComponentSumLocal(const DofField<T, N> &x)
    {
        Detail::RequireIdle(x);
        std::vector<double> value;
        Detail::Immediate([&](auto &e, const auto &s)
                          { value = ComponentSumLocal(e, s, x).Get(); });
        return value;
    }
    template <class T, size_t N>
    std::vector<double> ComponentMinLocal(const DofField<T, N> &x)
    {
        Detail::RequireIdle(x);
        std::vector<double> value;
        Detail::Immediate([&](auto &e, const auto &s)
                          { value = ComponentMinLocal(e, s, x).Get(); });
        return value;
    }
    template <class T, size_t N>
    std::vector<double> ComponentMaxLocal(const DofField<T, N> &x)
    {
        Detail::RequireIdle(x);
        std::vector<double> value;
        Detail::Immediate([&](auto &e, const auto &s)
                          { value = ComponentMaxLocal(e, s, x).Get(); });
        return value;
    }

    template <class T, size_t A, size_t B>
    double DotGlobal(Runtime::EagerEngine &e, const Runtime::ExecutionStream &s, const DofField<T, A> &x, const DofField<T, B> &y, bool immediate = false)
    {
        return Detail::Aggregate(x.Space(), [&]
                                 { Detail::Check(x, s); Detail::Pair(x, y); if (immediate) { Detail::RequireIdle(x); Detail::RequireIdle(y); } return Detail::GlobalSchema{1, x.Width(), sizeof(T), 0, 0}; }, [&]
                                 { return std::vector<double>{DotLocal(e, s, x, y).Get()}; })[0];
    }
    template <class T, size_t N>
    double NormGlobal(Runtime::EagerEngine &e, const Runtime::ExecutionStream &s, const DofField<T, N> &x, bool immediate = false)
    {
        return std::sqrt(Detail::Aggregate(x.Space(), [&]
                                           { Detail::Check(x, s); if (immediate) Detail::RequireIdle(x); return Detail::GlobalSchema{1, x.Width(), sizeof(T), 0, 1}; }, [&]
                                           { return std::vector<double>{DotLocal(e, s, x, x).Get()}; })[0]);
    }
    template <class T, size_t N>
    std::vector<double> ComponentSumGlobal(Runtime::EagerEngine &e, const Runtime::ExecutionStream &s, const DofField<T, N> &x, bool immediate = false)
    {
        return Detail::Aggregate(x.Space(), [&]
                                 { Detail::Check(x, s); if (immediate) Detail::RequireIdle(x); return Detail::GlobalSchema{x.Width(), x.Width(), sizeof(T), 0, 2}; }, [&]
                                 { return ComponentSumLocal(e, s, x).Get(); });
    }
    template <class T, size_t N>
    std::vector<double> ComponentMinGlobal(Runtime::EagerEngine &e, const Runtime::ExecutionStream &s, const DofField<T, N> &x, bool immediate = false)
    {
        return Detail::Aggregate(x.Space(), [&]
                                 { Detail::Check(x, s); if (immediate) Detail::RequireIdle(x); return Detail::GlobalSchema{x.Width(), x.Width(), sizeof(T), 1, 3}; }, [&]
                                 { return ComponentMinLocal(e, s, x).Get(); });
    }
    template <class T, size_t N>
    std::vector<double> ComponentMaxGlobal(Runtime::EagerEngine &e, const Runtime::ExecutionStream &s, const DofField<T, N> &x, bool immediate = false)
    {
        return Detail::Aggregate(x.Space(), [&]
                                 { Detail::Check(x, s); if (immediate) Detail::RequireIdle(x); return Detail::GlobalSchema{x.Width(), x.Width(), sizeof(T), 2, 4}; }, [&]
                                 { return ComponentMaxLocal(e, s, x).Get(); });
    }
    template <class T, size_t A, size_t B>
    double DotGlobal(const DofField<T, A> &x, const DofField<T, B> &y)
    {
        double value = 0;
        Detail::Immediate([&](auto &e, const auto &s)
                          { value = DotGlobal(e, s, x, y, true); });
        return value;
    }
    template <class T, size_t N>
    double NormGlobal(const DofField<T, N> &x)
    {
        double value = 0;
        Detail::Immediate([&](auto &e, const auto &s)
                          { value = NormGlobal(e, s, x, true); });
        return value;
    }
    template <class T, size_t N>
    std::vector<double> ComponentSumGlobal(const DofField<T, N> &x)
    {
        std::vector<double> value;
        Detail::Immediate([&](auto &e, const auto &s)
                          { value = ComponentSumGlobal(e, s, x, true); });
        return value;
    }
    template <class T, size_t N>
    std::vector<double> ComponentMinGlobal(const DofField<T, N> &x)
    {
        std::vector<double> value;
        Detail::Immediate([&](auto &e, const auto &s)
                          { value = ComponentMinGlobal(e, s, x, true); });
        return value;
    }
    template <class T, size_t N>
    std::vector<double> ComponentMaxGlobal(const DofField<T, N> &x)
    {
        std::vector<double> value;
        Detail::Immediate([&](auto &e, const auto &s)
                          { value = ComponentMaxGlobal(e, s, x, true); });
        return value;
    }
}
