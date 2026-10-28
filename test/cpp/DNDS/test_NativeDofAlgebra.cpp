#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "DNDS/Algebra/DofAlgebra.hpp"
#include <atomic>
#include <thread>

namespace A = DNDS::Algebra;
namespace R = DNDS::Runtime;

template <class Field>
static void Sentinels(Field &field)
{
    using T = typename Field::value_type;
    auto owned = field.Owned().Storage().Write(), ghost = field.Ghost().Storage().Write();
    std::fill_n(owned.template View<T>().data, field.Owned().Layout().Elements(), T(-919));
    std::fill_n(ghost.template View<T>().data, field.Ghost().Layout().Elements(), T(-727));
}

template <class Field>
static void CheckSentinels(const Field &field)
{
    using T = typename Field::value_type;
    const auto owned = field.Owned().Storage().Read(), ghost = field.Ghost().Storage().Read();
    const auto shape = field.Owned().Layout();
    for (size_t row = 0; row < shape.Rows(); ++row)
        for (size_t component = shape.Width(); component < shape.Stride(); ++component)
            CHECK(owned.template View<T>().data[row * shape.Stride() + component] == T(-919));
    for (size_t entry = 0; entry < field.Ghost().Layout().Elements(); ++entry)
        CHECK(ghost.template View<T>().data[entry] == T(-727));
}

TEST_CASE("Native DOF algebra operates on logical owned entries for arbitrary table widths")
{
    for (size_t workers : {size_t(0), size_t(2)})
        for (size_t width : {size_t(5), size_t(6), size_t(7), size_t(114)})
            for (size_t rows : {size_t(0), size_t(19)})
            {
                CAPTURE(workers);
                CAPTURE(width);
                CAPTURE(rows);
                A::DofSpace space(rows, 3);
                A::DofField<> x(space, width, width + 3), y(space, width, width + 1), z(space, width, width + 5);
                Sentinels(x);
                Sentinels(y);
                Sentinels(z);
                {
                    auto write = x.Owned().Write();
                    for (size_t row = 0; row < rows; ++row)
                        for (size_t column = 0; column < width; ++column)
                            write.At(row, column) = 1 + 2 * row + double(column) / 4;
                }
                R::EagerEngine engine(workers);
                auto stream = engine.CreateCPUStream();
                A::Fill(engine, stream, y, 3.);
                A::Copy(engine, stream, z, x);
                A::Scale(engine, stream, z, 2.);
                A::Axpy(engine, stream, z, -0.5, x);
                const auto intermediate = A::DotLocal(engine, stream, z, y);
                const double sumX = width * (double(rows) + double(rows) * (double(rows) - 1)) +
                                    double(rows) * double(width) * double(width - 1) / 8;
                CHECK(intermediate.Get() == doctest::Approx(4.5 * sumX));
                A::LinearCombination(engine, stream, z, 2., x, -3., y);
                A::Axpy(engine, stream, z, 1., z);
                A::LinearCombination(engine, stream, z, 0.25, z, 0.75, z);
                const auto sums = A::ComponentSumLocal(engine, stream, z).Get();
                const auto minimum = A::ComponentMinLocal(engine, stream, z).Get();
                const auto maximum = A::ComponentMaxLocal(engine, stream, z).Get();
                const auto norm = A::NormLocal(engine, stream, z).Get();
                engine.Synchronize();
                double expectedSquare = 0;
                auto read = z.Owned().Read();
                for (size_t row = 0; row < rows; ++row)
                    for (size_t column = 0; column < width; ++column)
                    {
                        const double expected = 4 * (1 + 2 * row + double(column) / 4) - 18;
                        CHECK(read.At(row, column) == expected);
                        expectedSquare += expected * expected;
                    }
                CHECK(norm == doctest::Approx(std::sqrt(expectedSquare)));
                for (size_t column = 0; column < width; ++column)
                {
                    CHECK(sums[column] == double(rows) * (-14 + double(column)) + 4 * double(rows) * double(rows ? rows - 1 : 0));
                    if (rows)
                    {
                        CHECK(minimum[column] == -14 + double(column));
                        CHECK(maximum[column] == -14 + double(column) + 8 * double(rows - 1));
                    }
                    else
                    {
                        CHECK(minimum[column] == std::numeric_limits<double>::infinity());
                        CHECK(maximum[column] == -std::numeric_limits<double>::infinity());
                    }
                }
                CheckSentinels(x);
                CheckSentinels(y);
                CheckSentinels(z);
            }
}

TEST_CASE("Native DOF fixed widths share handles and numerical copies are explicit")
{
    A::DofSpace space(11, 2);
    A::DofField<double, 5> fixed(space, 5, 8);
    A::DofField<> dynamic(space, 5, 7);
    Sentinels(fixed);
    Sentinels(dynamic);
    auto shared = fixed;
    CHECK(shared.Owned().Identity() == fixed.Owned().Identity());
    A::Fill(shared, 2.);
    CHECK(A::DotLocal(fixed, fixed) == 220);
    A::Copy(dynamic, fixed);
    CHECK(dynamic.Owned().Identity() != fixed.Owned().Identity());
    A::Scale(fixed, 3.);
    CHECK(A::DotLocal(dynamic, dynamic) == 220);
    CHECK(A::DotLocal(fixed, dynamic) == 660);
    CheckSentinels(fixed);
    CheckSentinels(dynamic);
    A::DofField<float, 6> six(space, 6, 9);
    A::DofField<float, 7> seven(space, 7, 10);
    A::Fill(six, 2.F);
    A::Fill(seven, 3.F);
    CHECK(A::NormLocal(six) == doctest::Approx(std::sqrt(11 * 6 * 4)));
    CHECK(A::NormLocal(seven) == doctest::Approx(std::sqrt(11 * 7 * 9)));
}

TEST_CASE("Native DOF algebra validates semantic spaces, immutable shape and overlap")
{
    A::DofSpace space(4), other(4);
    A::DofField<> x(space, 5), unrelated(other, 5), wrongWidth(space, 6);
    CHECK_THROWS(A::Copy(x, unrelated));
    CHECK_THROWS(A::Axpy(x, 1., wrongWidth));
    CHECK_THROWS(A::DotLocal(x, unrelated));
    CHECK_THROWS(A::DofField<>(space, 0));
    using Array = A::DofField<>::Array;
    CHECK_THROWS(A::DofField<>(space, Array(R::RowLayout::Dense(3, 5)), Array(R::RowLayout::Dense(0, 5))));
    R::Array<double> ragged(R::RowLayout::Ragged({5, 5, 5, 5}));
    CHECK_THROWS(Array::FromSharedStorage(ragged.SharedStorage()));
    auto bytes = R::Buffer::Allocate((4 * 7 + 1) * sizeof(double));
    auto a = Array::FromStorage(R::RowLayout::Dense(4, 5, 7), bytes.Subspan(0, 4 * 7 * sizeof(double)));
    auto b = Array::FromStorage(R::RowLayout::Dense(4, 5, 7), bytes.Subspan(sizeof(double), 4 * 7 * sizeof(double)));
    A::DofField<> left(space, a, Array(R::RowLayout::Dense(0, 5))), right(space, b, Array(R::RowLayout::Dense(0, 5)));
    CHECK_THROWS(A::Copy(left, right));
    CHECK_THROWS(A::LinearCombination(left, 1., left, 1., right));
    A::Copy(left, left);
    A::DofSpace ghostSpace(4, 4);
    auto overlap = Array(R::RowLayout::Dense(4, 5));
    A::DofField<> out(ghostSpace, overlap, Array(R::RowLayout::Dense(4, 5)));
    A::DofField<> in(ghostSpace, Array(R::RowLayout::Dense(4, 5)), overlap);
    CHECK_THROWS(A::Copy(out, in));
    x.Owned().Reset(R::RowLayout::Dense(3, 5));
    CHECK_THROWS(A::Fill(x, 1.));
    CHECK_THROWS(A::NormGlobal(unrelated));
}

struct AlgebraGate : R::StreamEvent::State
{
    std::atomic<bool> open{false};
    bool Ready() const override { return open.load(); }
    void Wait() const override
    {
        while (!open.load())
            std::this_thread::yield();
    }
};

TEST_CASE("Immediate DOF convenience rejects pending or active access before using an inline lane")
{
    R::EagerEngine engine(1);
    const auto stream = engine.CreateCPUStream();
    auto gate = std::make_shared<AlgebraGate>();
    stream.WaitFor(R::StreamEvent(gate));
    A::DofField<> value(A::DofSpace(17), 5);
    A::Fill(engine, stream, value, 3.);
    CHECK_THROWS(A::Fill(value, 7.));
    CHECK_THROWS(A::DotLocal(value, value));
    gate->open = true;
    stream.Record().WaitHost();
    CHECK(A::DotLocal(value, value) == 17 * 5 * 9);
    A::Fill(value, 7.);
    CHECK(A::DotLocal(value, value) == 17 * 5 * 49);
    {
        auto active = value.Owned().Read();
        CHECK_THROWS(A::Scale(value, 2.));
        CHECK_THROWS(A::DotLocal(value, value));
    }
    CHECK(A::NormLocal(value) == doctest::Approx(7 * std::sqrt(17 * 5)));
}

TEST_CASE("Native DOF commands retain values and honor explicit cross-stream joins")
{
    R::EagerEngine engine(2);
    auto producer = engine.CreateCPUStream(), consumer = engine.CreateCPUStream();
    auto gate = std::make_shared<AlgebraGate>();
    producer.WaitFor(R::StreamEvent(gate));
    A::DofSpace space(37, 2);
    auto input = std::make_unique<A::DofField<>>(space, 7, 10);
    A::DofField<> output(space, 7, 8);
    std::weak_ptr<const R::ArrayStorage> retained = input->Owned().SharedStorage();
    A::Fill(engine, producer, *input, 4.);
    CHECK_THROWS(input->Owned().Read());
    consumer.WaitFor(producer.Record());
    A::Copy(engine, consumer, output, *input);
    input.reset();
    CHECK_FALSE(retained.expired());
    gate->open = true;
    CHECK(A::NormLocal(engine, consumer, output).Get() == doctest::Approx(4 * std::sqrt(37 * 7)));
    engine.Synchronize();
    CHECK(retained.expired());
    // Failed upstream commands invalidate the output and reduction Get reports
    // the failed stream instead of manufacturing a usable host scalar.
    R::EagerEngine failed(1);
    const auto stream = failed.CreateCPUStream();
    auto failureGate = std::make_shared<AlgebraGate>();
    stream.WaitFor(R::StreamEvent(failureGate));
    R::SubmitArrays(failed, stream, [](R::ExecutionContext, auto)
                    { throw std::runtime_error("algebra upstream failure"); }, R::Write(output.Owned()));
    auto result = A::DotLocal(failed, stream, output, output);
    failureGate->open = true;
    CHECK_THROWS(result.Get());
    CHECK_THROWS(failed.Synchronize());
    CHECK_FALSE(output.Owned().IsValid());
}

TEST_CASE("Native DOF global reductions include uneven and empty owned ranks with collective failure")
{
    int rank = 0, ranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const size_t rows = ranks > 1 && rank + 1 == ranks ? 0 : size_t(rank + 2);
    auto context = std::make_shared<R::CommunicationContext>(MPI_COMM_WORLD);
    A::DofSpace space(context, rows, 3);
    A::DofField<> x(space, 7, 10), y(space, 7, 9);
    A::Fill(x, double(rank + 1));
    A::Fill(y, 2.);
    double expectedDot = 0, expectedSquare = 0, expectedComponent = 0, expectedConstantDot = 0;
    for (int peer = 0; peer < ranks; ++peer)
    {
        const auto count = ranks > 1 && peer + 1 == ranks ? 0 : peer + 2;
        expectedDot += count * 7 * (peer + 1) * 2;
        expectedSquare += count * 7 * (peer + 1) * (peer + 1);
        expectedComponent += count * (peer + 1);
        expectedConstantDot += count * 7 * 4;
    }
    CHECK(A::DotGlobal(x, y) == expectedDot);
    CHECK(A::NormGlobal(x) == doctest::Approx(std::sqrt(expectedSquare)));
    for (double value : A::ComponentSumGlobal(x))
        CHECK(value == expectedComponent);
    for (double value : A::ComponentMinGlobal(x))
        CHECK(value == 1);
    for (double value : A::ComponentMaxGlobal(x))
        CHECK(value == std::max(1, ranks - 1));
    A::DofField<> mismatch(rank == 0 ? A::DofSpace(rows, 3) : space, 7);
    CHECK_THROWS(A::DotGlobal(x, mismatch));
    CHECK(A::DotGlobal(x, y) == expectedDot);
    A::DofField<> invalid(space, 7);
    if (rank == 0)
        invalid.Owned().Reset(R::RowLayout::Dense(rows + 1, 7));
    CHECK_THROWS(A::ComponentSumGlobal(invalid));
    CHECK(A::DotGlobal(x, y) == expectedDot);
    A::DofField<> pending(space, 7);
    R::EagerEngine pendingEngine(1);
    const auto pendingStream = pendingEngine.CreateCPUStream();
    auto pendingGate = std::make_shared<AlgebraGate>();
    if (rank == 0)
    {
        pendingStream.WaitFor(R::StreamEvent(pendingGate));
        A::Fill(pendingEngine, pendingStream, pending, 3.);
    }
    CHECK_THROWS(A::NormGlobal(pending));
    pendingGate->open = true;
    pendingStream.Record().WaitHost();
    A::Fill(pending, 2.);
    CHECK(A::DotGlobal(pending, y) == expectedConstantDot);
    if (ranks > 1)
    {
        A::DofField<> scalar(space, 1);
        A::Fill(scalar, 2.);
        if (rank == 0)
            CHECK_THROWS(A::DotGlobal(scalar, scalar));
        else
            CHECK_THROWS(A::ComponentSumGlobal(scalar));
        if (rank == 0)
            CHECK_THROWS(A::NormGlobal(scalar));
        else
            CHECK_THROWS(A::DotGlobal(scalar, scalar));
        CHECK(A::DotGlobal(x, y) == expectedDot);
    }
}

TEST_CASE("Native DOF operators support a manufactured diagonal Krylov solve")
{
    auto communication = std::make_shared<R::CommunicationContext>(MPI_COMM_WORLD);
    A::DofSpace space(communication, 17, 2);
    A::DofField<double, 7> x(space, 7, 10), residual(space), direction(space), product(space);
    A::Fill(x, 0.);
    {
        auto write = residual.Owned().Write();
        for (size_t row = 0; row < 17; ++row)
            for (size_t component = 0; component < 7; ++component)
                write.At(row, component) = double(component + 1) * (1 + double(row) / 2 + double(component) / 4);
    }
    A::Copy(direction, residual);
    R::EagerEngine engine(2);
    const auto stream = engine.CreateCPUStream();
    double square = A::DotGlobal(engine, stream, residual, residual);
    size_t iterations = 0;
    for (; iterations < 8 && square > 1e-20; ++iterations)
    {
        R::SubmitArrays(engine, stream, [](R::ExecutionContext, auto input, auto output)
                        {
                            for (size_t row = 0; row < input.Rows(); ++row)
                                for (size_t component = 0; component < 7; ++component)
                                    output(row, component) = double(component + 1) * input(row, component); }, R::Read(direction.Owned()), R::Write(product.Owned()));
        const double alpha = square / A::DotGlobal(engine, stream, direction, product);
        A::Axpy(engine, stream, x, alpha, direction);
        A::Axpy(engine, stream, residual, -alpha, product);
        const double next = A::DotGlobal(engine, stream, residual, residual);
        if (next > 1e-20)
            A::LinearCombination(engine, stream, direction, 1., residual, next / square, direction);
        square = next;
    }
    engine.Synchronize();
    CHECK(iterations <= 7);
    CHECK(std::sqrt(square) < 1e-10);
    const auto read = x.Owned().Read();
    for (size_t row = 0; row < 17; ++row)
        for (size_t component = 0; component < 7; ++component)
            CHECK(read.At(row, component) == doctest::Approx(1 + double(row) / 2 + double(component) / 4).epsilon(1e-10));
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    const auto result = context.run();
    MPI_Finalize();
    return result;
}
