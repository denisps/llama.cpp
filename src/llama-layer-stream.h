#pragma once

#include "llama.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

struct llama_model;

// Layer streaming: double-buffered layer-by-layer GPU execution for models
// that don't fit entirely in GPU memory.
//
// Overview:
//   - Model weights are loaded to CPU (mmap) with n_gpu_layers=0
//   - All layer tensor `buffer` pointers are swapped to a GPU pool buffer
//     so the scheduler routes layer computation to GPU
//   - Two GPU buffer pools (A and B) each hold one layer's worth of weights
//   - Before graph compute, the graph is pre-analyzed to find layer boundaries
//   - During execution, the eval callback returns TRUE at layer boundaries
//     to create sync points; between batches, layer data is double-buffered
//
// Pipeline: Execute layer N (pool A) | Upload layer N+1 (pool B) | ...

struct llama_layer_stream {
    llama_layer_stream(
        const llama_model & model,
        ggml_backend_dev_t gpu_dev);

    ~llama_layer_stream();

    // Pre-analyze a compute graph to identify per-layer node boundaries.
    // Must be called after graph build, before graph compute.
    void pre_analyze_graph(ggml_cgraph * gf);

    // Prepare for a new inference pass.
    // For multi-token batches (prompt processing), uploads layer 0 synchronously
    // and kicks off async upload of layer 1.
    // For single-token batches (generation), restores tensors to CPU for direct
    // CPU computation (faster on Apple Silicon unified memory).
    void begin_pass(int n_tokens);

    // Called when computation for layer `il` is about to begin.
    // Waits for async upload to complete, swaps data pointers to GPU pool.
    void on_layer_begin(int il);

    // Called when computation for layer `il` is complete.
    void on_layer_end(int il);

    // Signal end of inference pass.
    void end_pass();

    int n_layers() const;
    bool is_active() const;
    bool is_pass_active() const;
    bool is_layer_boundary(ggml_tensor * t) const;

private:
    struct tensor_info {
        ggml_tensor * tensor;
        void        * cpu_data;
        ggml_backend_buffer_t cpu_buffer;
        size_t        size;
        size_t        gpu_offset;
    };

    struct gpu_pool {
        ggml_backend_buffer_t buffer = nullptr;
        size_t                size   = 0;
        int                   layer  = -1;
        bool                  ready  = false;
    };

    void upload_layer_sync(int il, int pool_idx);
    void upload_layer_async_start(int il, int pool_idx);
    void upload_thread_fn();
    void swap_layer_tensors_to_gpu(int il, int pool_idx);
    void restore_layer_tensors(int il);
    void build_layer_info(int il);

    const llama_model & model;
    ggml_backend_buffer_type_t buft;
    ggml_backend_t      gpu_backend = nullptr;

    std::vector<std::vector<tensor_info>> layer_tensors;

    gpu_pool pools[2];
    int      active_pool = 0;

    std::thread              upload_thread;
    std::mutex               upload_mutex;
    std::condition_variable  upload_cv;
    int                      upload_request_layer = -1;
    int                      upload_request_pool  = -1;
    std::atomic<bool>        upload_done{true};
    std::atomic<bool>        shutdown{false};

    int  current_layer   = -1;
    bool active          = false;
    bool pass_active     = false;  // whether the current pass uses GPU layer streaming
    int  min_batch_for_gpu = 2;    // minimum n_tokens to use GPU layer streaming
    size_t max_layer_size = 0;

    // Pre-analyzed graph layer boundaries (set of node pointers that are
    // the last node of each layer — returning TRUE for these in the ask
    // callback creates the per-layer sync points)
    std::unordered_set<ggml_tensor *> layer_boundary_nodes;
};

// Eval callback wrapper that drives the layer streaming pipeline.
struct llama_layer_stream_eval_cb {
    llama_layer_stream * streamer = nullptr;

    ggml_backend_sched_eval_callback original_cb = nullptr;
    void * original_ud = nullptr;

    int last_layer_seen = -1;

    static bool callback(ggml_tensor * t, bool ask, void * user_data);

    // Parse layer index from tensor name (e.g., "blk.5.attn_norm" -> 5, "attn_norm-5" -> 5)
    static int parse_layer_from_name(const char * name);
};
