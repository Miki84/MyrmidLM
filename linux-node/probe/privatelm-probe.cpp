// privatelm-probe: lists this machine's ggml devices as JSON, with free
// memory and a measured memory bandwidth for each.
//
// The phone master splits a model by free memory and by memory bandwidth
// (token speed follows it closely), so a PC worker advertises both, just as
// worker phones do. Measuring beats a lookup table: it covers every CUDA,
// ROCm, Vulkan and SYCL device, including iGPUs and odd clocks.
//
//   privatelm-probe            all devices, with bandwidth
//   privatelm-probe --no-bench memory only (instant)
//   privatelm-probe --mb 512   size of the test buffers

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-rpc.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

static std::string json_escape(const char* s) {
    std::string out;
    for (; s && *s; s++) {
        const unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') {
            out += '\\';
            out += (char)c;
        } else if (c < 0x20) {
            char buf[8];
            snprintf(buf, sizeof(buf), "\\u%04x", c);
            out += buf;
        } else {
            out += (char)c;
        }
    }
    return out;
}

static const char* type_name(enum ggml_backend_dev_type t) {
    switch (t) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:  return "cpu";
        case GGML_BACKEND_DEVICE_TYPE_GPU:  return "gpu";
        case GGML_BACKEND_DEVICE_TYPE_IGPU: return "igpu";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "accel";
        default: return "other";
    }
}

static void quiet_log(ggml_log_level level, const char* text, void*) {
    if (level >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
}

/// Copies one buffer into another on the device; returns GB/s (read + write),
/// or 0 if the device can't run the test.
static double measure_bandwidth(ggml_backend_dev_t dev, size_t want_bytes) {
    size_t free_mem = 0, total_mem = 0;
    ggml_backend_dev_memory(dev, &free_mem, &total_mem);
    size_t bytes = want_bytes;
    if (free_mem > 0) bytes = std::min(bytes, free_mem / 4);
    bytes = std::max<size_t>(bytes, 16u << 20) & ~size_t(255);

    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) return 0;

    if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto set_threads = (ggml_backend_set_n_threads_t)ggml_backend_reg_get_proc_address(
            reg, "ggml_backend_set_n_threads");
        if (set_threads) set_threads(backend, (int)std::max(1u, std::thread::hardware_concurrency()));
    }

    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * 8 + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context* ctx = ggml_init(params);
    const int64_t n = (int64_t)(bytes / sizeof(float));
    ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_set_input(a);
    ggml_set_output(b);
    ggml_tensor* c = ggml_cpy(ctx, a, b);
    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, c);

    double gbps = 0;
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf && ggml_backend_supports_op(backend, c)) {
        ggml_backend_buffer_clear(buf, 0);
        // Warm-up (kernel compile, clocks ramping up).
        for (int i = 0; i < 3; i++) ggml_backend_graph_compute(backend, graph);
        ggml_backend_synchronize(backend);

        const int iters = 20;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; i++) ggml_backend_graph_compute(backend, graph);
        ggml_backend_synchronize(backend);
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (secs > 0) gbps = 2.0 * (double)bytes * iters / secs / 1e9;
    }
    if (buf) ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return gbps;
}

int main(int argc, char** argv) {
    bool bench = true;
    size_t mb = 256;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--no-bench")) {
            bench = false;
        } else if (!strcmp(argv[i], "--mb") && i + 1 < argc) {
            mb = (size_t)std::max(16, atoi(argv[++i]));
        } else {
            fprintf(stderr, "usage: %s [--no-bench] [--mb N]\n", argv[0]);
            return 2;
        }
    }
    ggml_log_set(quiet_log, nullptr);
    ggml_backend_load_all();

    printf("{\n  \"rpc_proto\": \"%d.%d.%d\",\n  \"devices\": [",
           RPC_PROTO_MAJOR_VERSION, RPC_PROTO_MINOR_VERSION, RPC_PROTO_PATCH_VERSION);
    bool first = true;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        if (type == GGML_BACKEND_DEVICE_TYPE_ACCEL) continue;  // BLAS, AMX: not placeable
        size_t free_mem = 0, total_mem = 0;
        ggml_backend_dev_memory(dev, &free_mem, &total_mem);
        const double gbps = bench ? measure_bandwidth(dev, mb << 20) : 0.0;
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        printf("%s\n    {\"name\": \"%s\", \"description\": \"%s\", \"backend\": \"%s\", "
               "\"type\": \"%s\", \"free_mb\": %zu, \"total_mb\": %zu, \"bandwidth_gbps\": %.1f}",
               first ? "" : ",",
               json_escape(ggml_backend_dev_name(dev)).c_str(),
               json_escape(ggml_backend_dev_description(dev)).c_str(),
               json_escape(reg ? ggml_backend_reg_name(reg) : "").c_str(),
               type_name(type), free_mem >> 20, total_mem >> 20, gbps);
        fflush(stdout);
        first = false;
    }
    printf("\n  ]\n}\n");
    return 0;
}
