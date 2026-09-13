// Trusted single-process HYPRE policy driver.
// Candidate creation, setup, and solve are timed. Assembly, destruction,
// extraction, mutation checks, and verification are outside the interval.

#include <HYPRE.h>
#include <HYPRE_IJ_mv.h>
#include <HYPRE_parcsr_ls.h>

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/time.h>
#include <unistd.h>

extern "C" HYPRE_Int solver_create(HYPRE_Solver* solver,
                                   HYPRE_Real relative_tolerance,
                                   HYPRE_Real absolute_tolerance,
                                   HYPRE_Int maximum_iterations);

namespace {

constexpr char kInputMagic[8] = {'L', 'S', 'B', 'I', 'N', '0', '0', '1'};
constexpr char kOutputMagic[8] = {'L', 'S', 'B', 'O', 'U', 'T', '0', '1'};
constexpr std::uint32_t kProtocolVersion = 1;

volatile std::sig_atomic_t g_phase = 0;

void fatal_signal_handler(int signal_number) {
    if (signal_number == SIGALRM) {
        constexpr char message[] = "LSB_TIMING_TIMEOUT\n";
        static_cast<void>(::write(STDERR_FILENO, message, sizeof(message) - 1));
        ::_exit(124);
    }
    const char* message = "LSB_FATAL_PHASE=unknown\n";
    switch (g_phase) {
        case 1: message = "LSB_FATAL_PHASE=read_input\n"; break;
        case 2: message = "LSB_FATAL_PHASE=hypre_initialize\n"; break;
        case 3: message = "LSB_FATAL_PHASE=create_objects\n"; break;
        case 4: message = "LSB_FATAL_PHASE=solver_create\n"; break;
        case 5: message = "LSB_FATAL_PHASE=solver_setup\n"; break;
        case 6: message = "LSB_FATAL_PHASE=solver_solve\n"; break;
        case 7: message = "LSB_FATAL_PHASE=solver_destroy\n"; break;
        case 8: message = "LSB_FATAL_PHASE=solution_read\n"; break;
        case 9: message = "LSB_FATAL_PHASE=write_output\n"; break;
        case 10: message = "LSB_FATAL_PHASE=destroy_objects\n"; break;
        case 11: message = "LSB_FATAL_PHASE=hypre_finalize\n"; break;
    }
    static_cast<void>(::write(STDERR_FILENO, message, std::strlen(message)));
    ::_exit(128 + signal_number);
}

#pragma pack(push, 1)
struct InputHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t reserved;
    std::uint64_t n;
    std::uint64_t nnz;
    double tolerance;
};

struct OutputHeader {
    char magic[8];
    std::uint32_t version;
    std::int32_t status;
    std::uint64_t n;
    std::uint64_t elapsed_ns;
    std::uint64_t create_ns;
    std::uint64_t setup_ns;
    std::uint64_t solve_ns;
    std::uint8_t input_mutated;
    std::uint8_t padding[7];
};
#pragma pack(pop)

static_assert(sizeof(InputHeader) == 40, "input ABI changed");
static_assert(sizeof(OutputHeader) == 64, "output ABI changed");

struct Input {
    std::vector<std::uint64_t> rows;
    std::vector<std::uint32_t> columns;
    std::vector<double> values;
    std::vector<double> rhs;
    std::vector<double> initial_guess;
    double tolerance = 0.0;
};

template <typename T>
void read_exact(std::ifstream& stream, T* destination, std::size_t count) {
    const auto bytes = static_cast<std::streamsize>(sizeof(T) * count);
    stream.read(reinterpret_cast<char*>(destination), bytes);
    if (stream.gcount() != bytes) {
        throw std::runtime_error("truncated input");
    }
}

Input read_input(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open input");
    }
    InputHeader header{};
    read_exact(stream, &header, 1);
    if (std::memcmp(header.magic, kInputMagic, 8) != 0 ||
        header.version != kProtocolVersion || header.reserved != 0 ||
        header.n == 0 || !std::isfinite(header.tolerance) ||
        header.tolerance <= 0.0 || header.tolerance >= 1.0 ||
        header.n > static_cast<std::uint64_t>(std::numeric_limits<HYPRE_Int>::max()) ||
        header.nnz > static_cast<std::uint64_t>(std::numeric_limits<HYPRE_Int>::max())) {
        throw std::runtime_error("invalid input header");
    }
    Input input;
    input.rows.resize(header.n + 1);
    input.columns.resize(header.nnz);
    input.values.resize(header.nnz);
    input.rhs.resize(header.n);
    input.initial_guess.resize(header.n);
    input.tolerance = header.tolerance;
    read_exact(stream, input.rows.data(), input.rows.size());
    read_exact(stream, input.columns.data(), input.columns.size());
    read_exact(stream, input.values.data(), input.values.size());
    read_exact(stream, input.rhs.data(), input.rhs.size());
    read_exact(stream, input.initial_guess.data(), input.initial_guess.size());
    char trailing = 0;
    if (stream.read(&trailing, 1)) {
        throw std::runtime_error("input has trailing bytes");
    }
    if (input.rows.front() != 0 || input.rows.back() != header.nnz) {
        throw std::runtime_error("invalid CSR offsets");
    }
    for (std::size_t row = 0; row < header.n; ++row) {
        if (input.rows[row] > input.rows[row + 1]) {
            throw std::runtime_error("nonmonotone CSR offsets");
        }
    }
    for (std::size_t index = 0; index < header.nnz; ++index) {
        if (input.columns[index] >= header.n || !std::isfinite(input.values[index])) {
            throw std::runtime_error("invalid CSR entry");
        }
    }
    const auto finite = [](double value) { return std::isfinite(value); };
    if (!std::all_of(input.rhs.begin(), input.rhs.end(), finite) ||
        !std::all_of(input.initial_guess.begin(), input.initial_guess.end(), finite)) {
        throw std::runtime_error("nonfinite input vector");
    }
    return input;
}

int positive_int(const char* text) {
    std::size_t consumed = 0;
    const long value = std::stol(text, &consumed);
    if (text[consumed] != '\0' || value <= 0 ||
        value > std::numeric_limits<HYPRE_Int>::max()) {
        throw std::runtime_error("invalid maximum iteration count");
    }
    return static_cast<int>(value);
}

double positive_double(const char* text) {
    std::size_t consumed = 0;
    const double value = std::stod(text, &consumed);
    if (text[consumed] != '\0' || !std::isfinite(value) || value <= 0.0) {
        throw std::runtime_error("invalid timing limit");
    }
    return value;
}

void set_timing_limit(double seconds) {
    struct itimerval timer {};
    const auto microseconds = static_cast<std::uint64_t>(std::ceil(seconds * 1e6));
    timer.it_value.tv_sec = static_cast<time_t>(microseconds / 1000000U);
    timer.it_value.tv_usec = static_cast<suseconds_t>(microseconds % 1000000U);
    if (::setitimer(ITIMER_REAL, &timer, nullptr) != 0) {
        throw std::runtime_error("cannot arm timing limit");
    }
}

void clear_timing_limit() {
    struct itimerval timer {};
    if (::setitimer(ITIMER_REAL, &timer, nullptr) != 0) {
        throw std::runtime_error("cannot clear timing limit");
    }
}

void check(HYPRE_Int status, const char* operation) {
    if (status != 0) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 std::to_string(status));
    }
}

struct HypreObjects {
    HYPRE_IJMatrix matrix = nullptr;
    HYPRE_IJVector rhs = nullptr;
    HYPRE_IJVector solution = nullptr;
    HYPRE_ParCSRMatrix par_matrix = nullptr;
    HYPRE_ParVector par_rhs = nullptr;
    HYPRE_ParVector par_solution = nullptr;
    std::vector<HYPRE_BigInt> indices;
    std::vector<HYPRE_BigInt> columns;
    std::vector<HYPRE_Int> row_sizes;
};

HypreObjects create_objects(const Input& input) {
    HypreObjects objects;
    const std::size_t n = input.rhs.size();
    const HYPRE_BigInt upper = static_cast<HYPRE_BigInt>(n - 1);
    const MPI_Comm communicator = 0;
    objects.indices.resize(n);
    objects.row_sizes.resize(n);
    objects.columns.resize(input.columns.size());
    for (std::size_t row = 0; row < n; ++row) {
        objects.indices[row] = static_cast<HYPRE_BigInt>(row);
        objects.row_sizes[row] = static_cast<HYPRE_Int>(input.rows[row + 1] - input.rows[row]);
    }
    for (std::size_t index = 0; index < input.columns.size(); ++index) {
        objects.columns[index] = static_cast<HYPRE_BigInt>(input.columns[index]);
    }

    check(HYPRE_IJMatrixCreate(communicator, 0, upper, 0, upper, &objects.matrix),
          "HYPRE_IJMatrixCreate");
    check(HYPRE_IJMatrixSetObjectType(objects.matrix, HYPRE_PARCSR),
          "HYPRE_IJMatrixSetObjectType");
    check(HYPRE_IJMatrixSetRowSizes(objects.matrix, objects.row_sizes.data()),
          "HYPRE_IJMatrixSetRowSizes");
    check(HYPRE_IJMatrixInitialize(objects.matrix), "HYPRE_IJMatrixInitialize");
    for (std::size_t row = 0; row < n; ++row) {
        const std::size_t begin = input.rows[row];
        HYPRE_Int count = objects.row_sizes[row];
        const HYPRE_BigInt row_index = static_cast<HYPRE_BigInt>(row);
        check(HYPRE_IJMatrixSetValues(objects.matrix, 1, &count, &row_index,
                                      objects.columns.data() + begin,
                                      input.values.data() + begin),
              "HYPRE_IJMatrixSetValues");
    }
    check(HYPRE_IJMatrixAssemble(objects.matrix), "HYPRE_IJMatrixAssemble");
    check(HYPRE_IJMatrixGetObject(objects.matrix,
                                  reinterpret_cast<void**>(&objects.par_matrix)),
          "HYPRE_IJMatrixGetObject");

    check(HYPRE_IJVectorCreate(communicator, 0, upper, &objects.rhs),
          "HYPRE_IJVectorCreate(rhs)");
    check(HYPRE_IJVectorSetObjectType(objects.rhs, HYPRE_PARCSR),
          "HYPRE_IJVectorSetObjectType(rhs)");
    check(HYPRE_IJVectorInitialize(objects.rhs), "HYPRE_IJVectorInitialize(rhs)");
    check(HYPRE_IJVectorSetValues(objects.rhs, static_cast<HYPRE_Int>(n),
                                  objects.indices.data(), input.rhs.data()),
          "HYPRE_IJVectorSetValues(rhs)");
    check(HYPRE_IJVectorAssemble(objects.rhs), "HYPRE_IJVectorAssemble(rhs)");
    check(HYPRE_IJVectorGetObject(objects.rhs,
                                  reinterpret_cast<void**>(&objects.par_rhs)),
          "HYPRE_IJVectorGetObject(rhs)");

    check(HYPRE_IJVectorCreate(communicator, 0, upper, &objects.solution),
          "HYPRE_IJVectorCreate(solution)");
    check(HYPRE_IJVectorSetObjectType(objects.solution, HYPRE_PARCSR),
          "HYPRE_IJVectorSetObjectType(solution)");
    check(HYPRE_IJVectorInitialize(objects.solution),
          "HYPRE_IJVectorInitialize(solution)");
    check(HYPRE_IJVectorSetValues(objects.solution, static_cast<HYPRE_Int>(n),
                                  objects.indices.data(), input.initial_guess.data()),
          "HYPRE_IJVectorSetValues(solution)");
    check(HYPRE_IJVectorAssemble(objects.solution),
          "HYPRE_IJVectorAssemble(solution)");
    check(HYPRE_IJVectorGetObject(objects.solution,
                                  reinterpret_cast<void**>(&objects.par_solution)),
          "HYPRE_IJVectorGetObject(solution)");
    return objects;
}

void destroy_objects(HypreObjects& objects) {
    if (objects.solution) {
        HYPRE_IJVectorDestroy(objects.solution);
        objects.solution = nullptr;
    }
    if (objects.rhs) {
        HYPRE_IJVectorDestroy(objects.rhs);
        objects.rhs = nullptr;
    }
    if (objects.matrix) {
        HYPRE_IJMatrixDestroy(objects.matrix);
        objects.matrix = nullptr;
    }
}

bool input_mutated(const Input& input, HypreObjects& objects) {
    std::vector<double> matrix_values(input.values.size());
    const HYPRE_Int matrix_status = HYPRE_IJMatrixGetValues(
        objects.matrix,
        static_cast<HYPRE_Int>(objects.indices.size()),
        objects.row_sizes.data(),
        objects.indices.data(),
        objects.columns.data(),
        matrix_values.data());
    std::vector<double> rhs_values(input.rhs.size());
    const HYPRE_Int rhs_status = HYPRE_IJVectorGetValues(
        objects.rhs,
        static_cast<HYPRE_Int>(objects.indices.size()),
        objects.indices.data(),
        rhs_values.data());
    return matrix_status != 0 || rhs_status != 0 ||
           matrix_values != input.values || rhs_values != input.rhs;
}

struct SolveResult {
    std::vector<double> solution;
    std::uint64_t elapsed_ns = 0;
    std::uint64_t create_ns = 0;
    std::uint64_t setup_ns = 0;
    std::uint64_t solve_ns = 0;
    std::int32_t status = 0;
    bool input_mutated = false;
};

SolveResult solve(const Input& input, HypreObjects& objects,
                  HYPRE_Int maximum_iterations, double timing_limit_s) {
    HYPRE_Solver solver = nullptr;
    HYPRE_Int status = 0;
    set_timing_limit(timing_limit_s);
    const auto start = std::chrono::steady_clock::now();
    g_phase = 4;
    status |= solver_create(&solver, input.tolerance, 0.0, maximum_iterations);
    const auto create_stop = std::chrono::steady_clock::now();
    auto setup_stop = create_stop;
    if (status == 0 && solver != nullptr) {
        g_phase = 5;
        status |= HYPRE_SolverSetup(
            solver, reinterpret_cast<HYPRE_Matrix>(objects.par_matrix),
            reinterpret_cast<HYPRE_Vector>(objects.par_rhs),
            reinterpret_cast<HYPRE_Vector>(objects.par_solution));
        setup_stop = std::chrono::steady_clock::now();
        if (status == 0) {
            g_phase = 6;
            status |= HYPRE_SolverSolve(
                solver, reinterpret_cast<HYPRE_Matrix>(objects.par_matrix),
                reinterpret_cast<HYPRE_Vector>(objects.par_rhs),
                reinterpret_cast<HYPRE_Vector>(objects.par_solution));
        }
    }
    const auto stop = std::chrono::steady_clock::now();
    clear_timing_limit();

    SolveResult result;
    result.elapsed_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count());
    result.create_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(create_stop - start).count());
    result.setup_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(setup_stop - create_stop).count());
    result.solve_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(stop - setup_stop).count());
    result.status = static_cast<std::int32_t>(status);

    g_phase = 7;
    if (solver != nullptr) {
        result.status |= static_cast<std::int32_t>(HYPRE_SolverDestroy(solver));
    }
    g_phase = 8;
    result.solution.resize(input.rhs.size());
    result.status |= static_cast<std::int32_t>(HYPRE_IJVectorGetValues(
        objects.solution,
        static_cast<HYPRE_Int>(objects.indices.size()),
        objects.indices.data(),
        result.solution.data()));
    result.input_mutated = input_mutated(input, objects);
    return result;
}

void write_output(const std::string& path, const SolveResult& result) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("cannot open output");
    }
    OutputHeader header{};
    std::memcpy(header.magic, kOutputMagic, 8);
    header.version = kProtocolVersion;
    header.status = result.status;
    header.n = result.solution.size();
    header.elapsed_ns = result.elapsed_ns;
    header.create_ns = result.create_ns;
    header.setup_ns = result.setup_ns;
    header.solve_ns = result.solve_ns;
    header.input_mutated = result.input_mutated ? 1 : 0;
    stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
    stream.write(reinterpret_cast<const char*>(result.solution.data()),
                 static_cast<std::streamsize>(result.solution.size() * sizeof(double)));
    if (!stream) {
        throw std::runtime_error("failed writing output");
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: lsb_driver INPUT OUTPUT MAX_ITERATIONS TIME_LIMIT_S\n";
        return 2;
    }
    HypreObjects objects;
    bool initialized = false;
    std::signal(SIGSEGV, fatal_signal_handler);
    std::signal(SIGABRT, fatal_signal_handler);
    std::signal(SIGALRM, fatal_signal_handler);
    try {
        const int maximum_iterations = positive_int(argv[3]);
        const double timing_limit_s = positive_double(argv[4]);
        g_phase = 1;
        const Input input = read_input(argv[1]);
        g_phase = 2;
        check(HYPRE_Initialize(), "HYPRE_Initialize");
        initialized = true;
        g_phase = 3;
        objects = create_objects(input);
        const SolveResult result = solve(input, objects, maximum_iterations, timing_limit_s);
        g_phase = 9;
        write_output(argv[2], result);
        g_phase = 10;
        destroy_objects(objects);
        check(HYPRE_ClearAllErrors(), "HYPRE_ClearAllErrors");
        g_phase = 11;
        const HYPRE_Int finalize_status = HYPRE_Finalize();
        initialized = false;
        check(finalize_status, "HYPRE_Finalize");
        return 0;
    } catch (const std::exception& error) {
        destroy_objects(objects);
        if (initialized) {
            HYPRE_ClearAllErrors();
            HYPRE_Finalize();
        }
        std::cerr << error.what() << '\n';
        return 1;
    }
}
