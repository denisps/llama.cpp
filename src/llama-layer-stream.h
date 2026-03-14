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
#include <vector>

struct llama_model;
struct llama_model_loader;

// Layer streaming: double-buffered layer-by-layer execution for models
// that don't fit entirely in GPU memory.
//
// The idea:
//   - All model weights live on the CPU (mmap or RAM)
//   - Two GPU buffer pools (A and B) each hold one layer's worth of weights
//   - During inference, layer N executes from pool A while layer N+1 is
//     being uploaded to pool B via a background thread
//   - After layer N finishes, pools are swapped
//
// Step n: | Execute layer n (pool A) | Upload layer n+1 (pool B) | (pool from n-1 is now free)

struct llama_layer_stream {
    // Initialize streaming for a model. The model must have its weights
    // accessible on the CPU (via mmap or loaded into RAM).
    // gpu_dev: the GPU device to stream to
    // model_loader: used to load tensor data from the GGUF file
    llama_layer_stream(
        const llama_model & model,
        ggml_backend_dev_t gpu_dev);

    ~llama_layer_stream();

    // Prepare for a new inference pass. Must be called before graph computation.
    // This pre-loads layer 0 into the active GPU buffer synchronously.
    void begin_pass();

    // Called during graph computation (from the eval callback) when we detect
    // that computation for a specific layer is about to begin.
    // Ensures that layer `il` weights are in GPU memory (blocking if the
    // background upload hasn't finished yet) and kicks off upload of layer `il+1`.
    void on_layer_begin(int il);

    // Called when computation for layer `il` is complete.
    // The GPU buffer holding that layer's weights can be reused.
    void on_layer_end(int il);

    // Signal end of inference pass.
    void end_pass();

    // Returns how many layers the model has
    int n_layers() const;

    // Get the GPU buffer type used for streaming
    ggml_backend_buffer_type_t gpu_buft() const;

    // Check if a tensor belongs to layer il and swap its data pointer
    // to the GPU-buffered copy. Returns true if the tensor was swapped.
    bool swap_tensor_to_gpu(ggml_tensor * tensor, int il);

    // Restore a tensor's data pointer back to its CPU source.
    bool restore_tensor_to_cpu(ggml_tensor * tensor, int il);

    // Returns true if streaming is active and operational
    bool is_active() const;

private:
    // Per-layer tensor info: remembers the CPU source for each tensor
    struct tensor_info {
        ggml_tensor * tensor;     // the model's tensor pointer
        void        * cpu_data;   // original CPU data pointer
        ggml_backend_buffer_t cpu_buffer; // original CPU buffer
        size_t        size;       // tensor data size in bytes
        size_t        gpu_offset; // offset within the GPU pool buffer
    };

    // One GPU buffer pool that can hold a single layer's worth of weights
    struct gpu_pool {
        ggml_backend_buffer_t buffer = nullptr;
        size_t                size   = 0;
        int                   layer  = -1; // which layer is currently loaded (-1 = none)
        bool                  ready  = false;
    };

    void upload_layer_sync(int il, int pool_idx);
    void upload_layer_async_start(int il, int pool_idx);
    void upload_thread_fn();
    void swap_layer_tensors_to_gpu(int il, int pool_idx);
    void restore_layer_tensors(int il);

    // Collect all non-null tensor pointers from a layer and compute layout
    void build_layer_info(int il);

    const llama_model & model;
    ggml_backend_dev_t  gpu_dev;
    ggml_backend_buffer_type_t buft;
    ggml_backend_t      gpu_backend = nullptr;

    // Per-layer tensor mappings
    std::vector<std::vector<tensor_info>> layer_tensors; // [layer_idx][tensor_idx]

    // Double buffer pools
    gpu_pool pools[2];
    int      active_pool = 0; // which pool index the current layer uses

    // Background upload thread
    std::thread              upload_thread;
    std::mutex               upload_mutex;
    std::condition_variable  upload_cv;
    int                      upload_request_layer = -1;
    int                      upload_request_pool  = -1;
    std::atomic<bool>        upload_done{true};
    std::atomic<bool>        shutdown{false};

    // State tracking
    int  current_layer   = -1;
    bool active          = false;
    size_t max_layer_size = 0; // largest layer in bytes (for buffer allocation)
};

// Eval callback wrapper that drives the layer streaming pipeline.
// This wraps any existing eval callback the user may have set.
struct llama_layer_stream_eval_cb {
    llama_layer_stream * streamer = nullptr;

    // The user's original eval callback (if any)
    ggml_backend_sched_eval_callback original_cb = nullptr;
    void * original_ud = nullptr;

    int last_layer_seen = -1;

    static bool callback(ggml_tensor * t, bool ask, void * user_data);
};
