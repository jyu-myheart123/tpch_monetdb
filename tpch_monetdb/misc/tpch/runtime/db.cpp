#include "builder_api.hpp"
#include "loader_api.hpp"
#include "query_api.hpp"
#ifdef __APPLE__
#include "utils/plugin.hpp"
#endif
#ifndef __APPLE__
#include "utils/pipeline.hpp"
#endif

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <signal.h>
#include <stdexcept>

struct State {
    std::string data_path;
    RawData* raw_data;
    Engine* engine;
    double load_ms = 0.0;
};

static State state;

static void cleanup_reload_dir_on_start() {
    std::error_code ec;
    std::filesystem::remove_all("./build/.reload", ec);
    std::filesystem::create_directories("./build/.reload", ec);
}

#ifdef __APPLE__
static int execute_once() {
    Plugin loader("./build/libloader.so");
    auto loader_api = loader.get<LoaderApi>();
    std::cerr << "loader start\n";
    const auto load_t0 = std::chrono::steady_clock::now();
    state.raw_data = loader_api.load(state.data_path);
    const auto load_t1 = std::chrono::steady_clock::now();
    std::cerr << "loader done\n";
    const double load_ms = std::chrono::duration<double, std::milli>(load_t1 - load_t0).count();
    std::printf("Load ms: %.3f\n", load_ms);

    Plugin builder("./build/libbuilder.so");
    auto builder_api = builder.get<BuilderApi>();
    std::cerr << "builder start\n";
    const auto build_t0 = std::chrono::steady_clock::now();
    state.engine = builder_api.build(state.raw_data);
    const auto build_t1 = std::chrono::steady_clock::now();
    std::cerr << "builder done\n";
    const double build_ms = std::chrono::duration<double, std::milli>(build_t1 - build_t0).count();
    const double ingest_ms = load_ms + build_ms;
    std::printf("Build ms: %.3f\n", build_ms);
    std::printf("Ingest ms: %.3f\n", ingest_ms);

    Plugin query("./build/libquery.so");
    auto query_api = query.get<QueryApi>();
    std::cerr << "query start\n";
    query_api.query(state.engine);
    std::cerr << "query done\n";
    return 0;
}
#else
static auto build_pipeline() {
    return make_pipeline(
        stage<RunPolicy::OnChange>("./build/libloader.so", [](Plugin& plugin) {
            auto api = plugin.get<LoaderApi>();
            std::cerr << "loader start\n";
            const auto load_t0 = std::chrono::steady_clock::now();
            state.raw_data = api.load(state.data_path);
            const auto load_t1 = std::chrono::steady_clock::now();
            std::cerr << "loader done\n";
            const double load_ms = std::chrono::duration<double, std::milli>(load_t1 - load_t0).count();
            state.load_ms = load_ms;
            std::printf("Load ms: %.3f\n", load_ms);
            return 0;
        }),
        stage<RunPolicy::OnChange>("./build/libbuilder.so", [](Plugin& plugin, int) {
            auto api = plugin.get<BuilderApi>();
            std::cerr << "builder start\n";
            const auto build_t0 = std::chrono::steady_clock::now();
            state.engine = api.build(state.raw_data);
            const auto build_t1 = std::chrono::steady_clock::now();
            std::cerr << "builder done\n";
            const double build_ms = std::chrono::duration<double, std::milli>(build_t1 - build_t0).count();
            const double ingest_ms = state.load_ms + build_ms;
            std::printf("Build ms: %.3f\n", build_ms);
            std::printf("Ingest ms: %.3f\n", ingest_ms);
            return 0;
        }),
        stage<RunPolicy::Always>("./build/libquery.so", [](Plugin& plugin, int) {
            auto api = plugin.get<QueryApi>();
            std::cerr << "query start\n";
            api.query(state.engine);
            std::cerr << "query done\n";
            return 0;
        }));
}

static void run_child(int read_fd, int done_fd) {
    auto pipeline = build_pipeline();
    pipeline.run(read_fd, done_fd, false);
}
#endif

static int getenv_fd(const char* name) {
    const char* v = std::getenv(name);
    if (!v) {
        throw std::runtime_error(std::string(name) + " not supplied");
    }
    return std::atoi(v);
}

static FILE* open_fd_stream(int fd, const char* mode, const char* name) {
    FILE* stream = fdopen(fd, mode);
    if (stream == nullptr) {
        throw std::runtime_error(std::string("open ") + name + " failed");
    }
    return stream;
}

static bool read_command(FILE* stream, std::string& cmd) {
    char* line = nullptr;
    size_t cap = 0;
    const ssize_t nread = getline(&line, &cap, stream);
    if (nread < 0) {
        std::free(line);
        return false;
    }
    if (nread > 0 && line[nread - 1] == '\n') {
        line[nread - 1] = '\0';
    }
    cmd = line;
    std::free(line);
    return true;
}

#ifdef __APPLE__
static void run_parent_direct() {
    int in_fd = getenv_fd("P2C_FD");
    int out_fd = getenv_fd("C2P_FD");

    FILE* in = open_fd_stream(in_fd, "r", "P2C_FD");
    FILE* out = open_fd_stream(out_fd, "w", "C2P_FD");

    std::string cmd;
    while (read_command(in, cmd)) {
        std::cout << "got: " << cmd << "\n";

        if (cmd == "stop") {
            break;
        }
        if (cmd != "run") {
            throw std::runtime_error("invalid command");
        }

        try {
            const int exit_code = execute_once();
            std::fprintf(out, "exit_code: %d signal: 0\n", exit_code);
            std::fflush(out);
        } catch (const std::exception& exc) {
            std::cerr << exc.what() << "\n";
            std::fprintf(out, "exit_code: 1 signal: 0\n");
            std::fflush(out);
        }
    }

    std::fclose(in);
    std::fclose(out);
}
#endif

#ifndef __APPLE__
static void run_parent(PipelineControl& control) {
    int in_fd = getenv_fd("P2C_FD");  // read from parent
    int out_fd = getenv_fd("C2P_FD"); // write to parent

    FILE* in = open_fd_stream(in_fd, "r", "P2C_FD");
    FILE* out = open_fd_stream(out_fd, "w", "C2P_FD");

    std::string cmd;
    while (read_command(in, cmd)) {
        std::cout << "got: " << cmd << "\n";

        if (cmd == "stop") {
            break;
        }
        if (cmd != "run") {
            throw std::runtime_error("invalid command");
        }

        control.send_run();
        DoneToken token = control.read_done();
        std::cerr << "exit_code: " << token.exit_code << " signal: " << token.term_signal
                  << "\n";
        std::fprintf(out, "exit_code: %d signal: %d\n", token.exit_code, token.term_signal);
        std::fflush(out);
    }

    control.send_terminate();
    std::fclose(in);
    std::fclose(out);
}
#endif


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <DATA_PATH>\n";
        return 1;
    }
    std::string base_data = argv[1];
    state.data_path = base_data;

    cleanup_reload_dir_on_start();

    signal(SIGPIPE, SIG_IGN);
#ifdef __APPLE__
    run_parent_direct();
    return 0;
#else
    int p2c[2];
    int done_pipe[2];
    if (pipe(p2c) == -1) {
        perror("pipe");
        return 1;
    }
    if (pipe(done_pipe) == -1) {
        perror("pipe");
        close(p2c[0]);
        close(p2c[1]);
        return 1;
    }

    pid_t pid = fork();
    if (pid == 0) {
        close(p2c[1]);
        close(done_pipe[0]);
        run_child(p2c[0], done_pipe[1]);
        _exit(0);
    }
    if (pid < 0) {
        std::cerr << "[ERROR:INFRA_BLOCKED] fork failed: "
                  << std::strerror(errno) << "\n";
        close(p2c[0]);
        close(p2c[1]);
        close(done_pipe[0]);
        close(done_pipe[1]);
        return 1;
    }

    close(p2c[0]);
    close(done_pipe[1]);
    PipelineControl control(p2c[1], done_pipe[0], true);
    run_parent(control);
    waitpid(pid, nullptr, 0);
    return 0;
#endif
}
