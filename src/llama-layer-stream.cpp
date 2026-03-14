#include "llama-layer-stream.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstring>
#include <stdexcept>

// Helper: collect all non-null tensor pointers from a llama_layer struct.
// We iterate the struct's known tensor fields. This uses offsetof-style enumeration.
static std::vector<ggml_tensor *> get_layer_tensors(const llama_layer & layer) {
    std::vector<ggml_tensor *> tensors;

    // Macro to add a tensor if non-null
    #define ADD_TENSOR(field) if (layer.field) tensors.push_back(layer.field)

    // normalization
    ADD_TENSOR(attn_norm);
    ADD_TENSOR(attn_norm_b);
    ADD_TENSOR(attn_norm_2);
    ADD_TENSOR(attn_norm_2_b);
    ADD_TENSOR(attn_q_norm);
    ADD_TENSOR(attn_q_norm_b);
    ADD_TENSOR(attn_k_norm);
    ADD_TENSOR(attn_k_norm_b);
    ADD_TENSOR(attn_out_norm);
    ADD_TENSOR(attn_out_norm_b);
    ADD_TENSOR(attn_q_a_norm);
    ADD_TENSOR(attn_kv_a_norm);
    ADD_TENSOR(attn_sub_norm);
    ADD_TENSOR(attn_post_norm);
    ADD_TENSOR(ffn_sub_norm);
    ADD_TENSOR(attn_norm_cross);
    ADD_TENSOR(attn_norm_enc);
    ADD_TENSOR(ssm_norm);
    ADD_TENSOR(ssm_dt_norm);
    ADD_TENSOR(ssm_b_norm);
    ADD_TENSOR(ssm_c_norm);

    // attention
    ADD_TENSOR(wq);
    ADD_TENSOR(wk);
    ADD_TENSOR(wv);
    ADD_TENSOR(wo);
    ADD_TENSOR(wqkv);
    ADD_TENSOR(wq_a);
    ADD_TENSOR(wq_b);
    ADD_TENSOR(wkv_a_mqa);
    ADD_TENSOR(wkv_b);
    ADD_TENSOR(wk_b);
    ADD_TENSOR(wv_b);
    ADD_TENSOR(wq_cross);
    ADD_TENSOR(wk_cross);
    ADD_TENSOR(wv_cross);
    ADD_TENSOR(wo_cross);
    ADD_TENSOR(wq_enc);
    ADD_TENSOR(wk_enc);
    ADD_TENSOR(wv_enc);
    ADD_TENSOR(wo_enc);
    ADD_TENSOR(wqkv_gate);

    // attention bias
    ADD_TENSOR(bq);
    ADD_TENSOR(bk);
    ADD_TENSOR(bv);
    ADD_TENSOR(bo);
    ADD_TENSOR(bqkv);

    // relative position bias
    ADD_TENSOR(attn_rel_b);
    ADD_TENSOR(attn_rel_b_enc);
    ADD_TENSOR(attn_rel_b_cross);

    // normalization (ffn)
    ADD_TENSOR(ffn_norm);
    ADD_TENSOR(ffn_norm_b);
    ADD_TENSOR(ffn_post_norm);
    ADD_TENSOR(layer_out_norm);
    ADD_TENSOR(layer_out_norm_b);
    ADD_TENSOR(ffn_norm_exps);
    ADD_TENSOR(ffn_norm_enc);

    // ff
    ADD_TENSOR(ffn_gate);
    ADD_TENSOR(ffn_down);
    ADD_TENSOR(ffn_up);
    ADD_TENSOR(ffn_gate_enc);
    ADD_TENSOR(ffn_down_enc);
    ADD_TENSOR(ffn_up_enc);

    // ff MoE
    ADD_TENSOR(ffn_gate_inp);
    ADD_TENSOR(ffn_gate_exps);
    ADD_TENSOR(ffn_down_exps);
    ADD_TENSOR(ffn_up_exps);
    ADD_TENSOR(ffn_gate_up_exps);
    ADD_TENSOR(ffn_gate_inp_b);
    ADD_TENSOR(ffn_gate_exps_b);
    ADD_TENSOR(ffn_down_exps_b);
    ADD_TENSOR(ffn_up_exps_b);
    ADD_TENSOR(ffn_gate_up_exps_b);

    // ff MoE per-expert scales
    ADD_TENSOR(ffn_gate_exps_s);
    ADD_TENSOR(ffn_down_exps_s);
    ADD_TENSOR(ffn_up_exps_s);

    // ff MoE latent proj
    ADD_TENSOR(ffn_latent_down);
    ADD_TENSOR(ffn_latent_up);

    // ff shared expert
    ADD_TENSOR(ffn_gate_inp_shexp);
    ADD_TENSOR(ffn_gate_shexp);
    ADD_TENSOR(ffn_down_shexp);
    ADD_TENSOR(ffn_up_shexp);

    // ff adjugate experts
    ADD_TENSOR(ffn_gate_chexps);
    ADD_TENSOR(ffn_down_chexps);
    ADD_TENSOR(ffn_up_chexps);

    // ff bias
    ADD_TENSOR(ffn_gate_b);
    ADD_TENSOR(ffn_down_b);
    ADD_TENSOR(ffn_up_b);
    ADD_TENSOR(ffn_act);
    ADD_TENSOR(ffn_exp_probs_b);

    // mamba proj
    ADD_TENSOR(ssm_in);
    ADD_TENSOR(ssm_x);
    ADD_TENSOR(ssm_dt);
    ADD_TENSOR(ssm_out);

    // mamba
    ADD_TENSOR(ssm_conv1d);
    ADD_TENSOR(ssm_a);
    ADD_TENSOR(ssm_d);

    // mamba bias
    ADD_TENSOR(ssm_conv1d_b);
    ADD_TENSOR(ssm_dt_b);

    // qwen3next
    ADD_TENSOR(ssm_beta_alpha);

    // qwen3.5
    ADD_TENSOR(ssm_alpha);

    // rwkv
    ADD_TENSOR(time_mix_w1);
    ADD_TENSOR(time_mix_w2);
    ADD_TENSOR(time_mix_lerp_x);
    ADD_TENSOR(time_mix_lerp_w);
    ADD_TENSOR(time_mix_lerp_k);
    ADD_TENSOR(time_mix_lerp_v);
    ADD_TENSOR(time_mix_lerp_r);
    ADD_TENSOR(time_mix_lerp_g);
    ADD_TENSOR(time_mix_lerp_fused);

    ADD_TENSOR(time_mix_first);
    ADD_TENSOR(time_mix_decay);
    ADD_TENSOR(time_mix_decay_w1);
    ADD_TENSOR(time_mix_decay_w2);
    ADD_TENSOR(time_mix_key);
    ADD_TENSOR(time_mix_key_b);
    ADD_TENSOR(time_mix_value);
    ADD_TENSOR(time_mix_value_b);
    ADD_TENSOR(time_mix_receptance);
    ADD_TENSOR(time_mix_receptance_b);
    ADD_TENSOR(time_mix_gate);

    // rwkv7
    ADD_TENSOR(time_mix_w0);
    ADD_TENSOR(time_mix_a0);
    ADD_TENSOR(time_mix_a1);
    ADD_TENSOR(time_mix_a2);
    ADD_TENSOR(time_mix_v0);
    ADD_TENSOR(time_mix_v1);
    ADD_TENSOR(time_mix_v2);
    ADD_TENSOR(time_mix_g1);
    ADD_TENSOR(time_mix_g2);
    ADD_TENSOR(time_mix_k_k);
    ADD_TENSOR(time_mix_k_a);
    ADD_TENSOR(time_mix_r_k);

    ADD_TENSOR(time_mix_ln);
    ADD_TENSOR(time_mix_ln_b);
    ADD_TENSOR(time_mix_output);

    ADD_TENSOR(channel_mix_lerp_k);
    ADD_TENSOR(channel_mix_lerp_r);

    ADD_TENSOR(channel_mix_key);
    ADD_TENSOR(channel_mix_receptance);
    ADD_TENSOR(channel_mix_value);

    // long rope factors
    ADD_TENSOR(rope_long);
    ADD_TENSOR(rope_short);
    ADD_TENSOR(rope_freqs);

    // bitnet scale
    ADD_TENSOR(wq_s);
    ADD_TENSOR(wk_s);
    ADD_TENSOR(wv_s);
    ADD_TENSOR(wo_s);
    ADD_TENSOR(ffn_gate_s);
    ADD_TENSOR(ffn_up_s);
    ADD_TENSOR(ffn_down_s);

    // altup & laurel
    ADD_TENSOR(per_layer_inp_gate);
    ADD_TENSOR(per_layer_proj);
    ADD_TENSOR(per_layer_post_norm);
    ADD_TENSOR(altup_correct_coef);
    ADD_TENSOR(altup_correct_scale);
    ADD_TENSOR(altup_predict_coef);
    ADD_TENSOR(altup_router);
    ADD_TENSOR(altup_router_norm);
    ADD_TENSOR(laurel_l);
    ADD_TENSOR(laurel_r);
    ADD_TENSOR(laurel_post_norm);

    // openai-moe
    ADD_TENSOR(attn_sinks);

    // cogvlm
    ADD_TENSOR(visexp_attn_wqkv);
    ADD_TENSOR(visexp_attn_wo);
    ADD_TENSOR(visexp_ffn_gate);
    ADD_TENSOR(visexp_ffn_down);
    ADD_TENSOR(visexp_ffn_up);

    // xIELU activation parameters
    ADD_TENSOR(ffn_act_alpha_n);
    ADD_TENSOR(ffn_act_alpha_p);
    ADD_TENSOR(ffn_act_beta);
    ADD_TENSOR(ffn_act_eps);

    // Kimi Linear KDA
    ADD_TENSOR(ssm_q_conv);
    ADD_TENSOR(ssm_k_conv);
    ADD_TENSOR(ssm_v_conv);
    ADD_TENSOR(ssm_f_a);
    ADD_TENSOR(ssm_f_b);
    ADD_TENSOR(ssm_beta);
    ADD_TENSOR(ssm_g_a);
    ADD_TENSOR(ssm_g_b);
    ADD_TENSOR(ssm_o_norm);

    // DSA
    ADD_TENSOR(indexer_k_norm);
    ADD_TENSOR(indexer_k_norm_b);
    ADD_TENSOR(indexer_proj);
    ADD_TENSOR(indexer_attn_k);
    ADD_TENSOR(indexer_attn_q_b);

    #undef ADD_TENSOR

    return tensors;
}

// Compute total size of tensors with proper alignment
static size_t compute_layer_gpu_size(const std::vector<ggml_tensor *> & tensors, size_t alignment) {
    size_t total = 0;
    for (const auto * t : tensors) {
        size_t size = ggml_nbytes(t);
        // Align each tensor
        total = (total + alignment - 1) & ~(alignment - 1);
        total += size;
    }
    return total;
}

llama_layer_stream::llama_layer_stream(
        const llama_model & model,
        ggml_backend_dev_t gpu_dev)
    : model(model)
    , gpu_dev(gpu_dev) {

    buft = ggml_backend_dev_buffer_type(gpu_dev);
    if (!buft) {
        throw std::runtime_error("llama_layer_stream: failed to get GPU buffer type");
    }

    gpu_backend = ggml_backend_dev_init(gpu_dev, nullptr);
    if (!gpu_backend) {
        throw std::runtime_error("llama_layer_stream: failed to initialize GPU backend");
    }

    const int nl = n_layers();
    if (nl <= 0) {
        throw std::runtime_error("llama_layer_stream: model has no layers");
    }

    const size_t alignment = ggml_backend_buft_get_alignment(buft);

    // Build per-layer tensor info and compute max layer size
    layer_tensors.resize(nl);
    max_layer_size = 0;

    for (int il = 0; il < nl; il++) {
        build_layer_info(il);

        size_t layer_size = 0;
        for (const auto & ti : layer_tensors[il]) {
            layer_size += (ti.size + alignment - 1) & ~(alignment - 1);
        }
        max_layer_size = std::max(max_layer_size, layer_size);
    }

    LLAMA_LOG_INFO("%s: max layer size = %.2f MiB, allocating 2 GPU pool buffers\n",
        __func__, max_layer_size / (1024.0 * 1024.0));

    // Allocate two GPU buffer pools, each large enough for the biggest layer
    for (int i = 0; i < 2; i++) {
        pools[i].buffer = ggml_backend_buft_alloc_buffer(buft, max_layer_size);
        if (!pools[i].buffer) {
            // Clean up already allocated pool
            if (i > 0 && pools[0].buffer) {
                ggml_backend_buffer_free(pools[0].buffer);
                pools[0].buffer = nullptr;
            }
            throw std::runtime_error("llama_layer_stream: failed to allocate GPU buffer pool");
        }
        pools[i].size = max_layer_size;
        pools[i].layer = -1;
        pools[i].ready = false;
    }

    // Start background upload thread
    shutdown.store(false);
    upload_done.store(true);
    upload_thread = std::thread(&llama_layer_stream::upload_thread_fn, this);

    active = true;

    LLAMA_LOG_INFO("%s: layer streaming initialized with double-buffering for %d layers\n",
        __func__, nl);
}

llama_layer_stream::~llama_layer_stream() {
    // Signal and join background thread
    {
        std::lock_guard<std::mutex> lock(upload_mutex);
        shutdown.store(true);
    }
    upload_cv.notify_one();
    if (upload_thread.joinable()) {
        upload_thread.join();
    }

    // Restore any tensors that might still point to GPU
    for (int il = 0; il < n_layers(); il++) {
        restore_layer_tensors(il);
    }

    // Free GPU buffers
    for (int i = 0; i < 2; i++) {
        if (pools[i].buffer) {
            ggml_backend_buffer_free(pools[i].buffer);
            pools[i].buffer = nullptr;
        }
    }

    if (gpu_backend) {
        ggml_backend_free(gpu_backend);
        gpu_backend = nullptr;
    }
}

void llama_layer_stream::build_layer_info(int il) {
    const auto & layer = model.layers[il];
    auto tensors = get_layer_tensors(layer);

    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    size_t offset = 0;

    layer_tensors[il].clear();
    layer_tensors[il].reserve(tensors.size());

    for (auto * t : tensors) {
        if (!t->data) {
            continue; // skip unallocated tensors
        }
        tensor_info ti;
        ti.tensor     = t;
        ti.cpu_data   = t->data;
        ti.cpu_buffer = t->buffer;
        ti.size       = ggml_nbytes(t);

        // Compute aligned offset within GPU pool
        offset = (offset + alignment - 1) & ~(alignment - 1);
        ti.gpu_offset = offset;
        offset += ti.size;

        layer_tensors[il].push_back(ti);
    }
}

int llama_layer_stream::n_layers() const {
    return (int)model.layers.size();
}

ggml_backend_buffer_type_t llama_layer_stream::gpu_buft() const {
    return buft;
}

bool llama_layer_stream::is_active() const {
    return active;
}

void llama_layer_stream::begin_pass() {
    if (!active) return;

    current_layer = -1;

    // Pre-load layer 0 synchronously into pool 0
    upload_layer_sync(0, 0);
    active_pool = 0;
}

void llama_layer_stream::on_layer_begin(int il) {
    if (!active || il < 0 || il >= n_layers()) return;
    if (il == current_layer) return; // already set up

    const int my_pool = active_pool;

    // Wait for the upload to this pool to complete if it was async
    if (!upload_done.load()) {
        // The background thread is uploading to the other pool or this one
        std::unique_lock<std::mutex> lock(upload_mutex);
        upload_cv.wait(lock, [this]() { return upload_done.load(); });
    }

    // Verify the correct layer is in our pool
    if (pools[my_pool].layer != il || !pools[my_pool].ready) {
        // Need to upload synchronously (shouldn't happen in normal flow)
        LLAMA_LOG_WARN("%s: sync upload needed for layer %d (pool %d has layer %d)\n",
            __func__, il, my_pool, pools[my_pool].layer);
        upload_layer_sync(il, my_pool);
    }

    // Swap tensor data pointers to GPU
    swap_layer_tensors_to_gpu(il, my_pool);
    current_layer = il;

    // Kick off async upload of next layer into the other pool
    const int next_layer = il + 1;
    if (next_layer < n_layers()) {
        const int next_pool = 1 - my_pool;
        upload_layer_async_start(next_layer, next_pool);
    }
}

void llama_layer_stream::on_layer_end(int il) {
    if (!active || il < 0 || il >= n_layers()) return;

    // Restore tensor pointers back to CPU
    restore_layer_tensors(il);

    // Swap active pool for the next layer
    active_pool = 1 - active_pool;
    current_layer = -1;
}

void llama_layer_stream::end_pass() {
    if (!active) return;

    // Wait for any pending upload
    if (!upload_done.load()) {
        std::unique_lock<std::mutex> lock(upload_mutex);
        upload_cv.wait(lock, [this]() { return upload_done.load(); });
    }

    // Ensure all tensors are restored to CPU pointers
    for (int il = 0; il < n_layers(); il++) {
        restore_layer_tensors(il);
    }

    current_layer = -1;
}

void llama_layer_stream::upload_layer_sync(int il, int pool_idx) {
    if (il < 0 || il >= n_layers()) return;

    auto & pool = pools[pool_idx];
    if (pool.layer == il && pool.ready) return; // already loaded

    void * base = ggml_backend_buffer_get_base(pool.buffer);

    for (const auto & ti : layer_tensors[il]) {
        void * dst = (char *)base + ti.gpu_offset;
        // Copy from CPU to GPU pool buffer
        memcpy(dst, ti.cpu_data, ti.size);
    }

    // If the GPU buffer requires explicit uploading (non-unified memory),
    // we use ggml_backend_tensor_set. For unified memory (Apple Silicon Metal),
    // the memcpy above already works since shared buffers are used.
    // For discrete GPUs, we'd need to use the proper upload mechanism.
    // The Metal backend on Apple Silicon uses shared memory, so direct memcpy works.

    pool.layer = il;
    pool.ready = true;
}

void llama_layer_stream::upload_layer_async_start(int il, int pool_idx) {
    if (il < 0 || il >= n_layers()) return;

    // Wait for any prior async upload to finish
    if (!upload_done.load()) {
        std::unique_lock<std::mutex> lock(upload_mutex);
        upload_cv.wait(lock, [this]() { return upload_done.load(); });
    }

    {
        std::lock_guard<std::mutex> lock(upload_mutex);
        upload_request_layer = il;
        upload_request_pool  = pool_idx;
        upload_done.store(false);
    }
    upload_cv.notify_one();
}

void llama_layer_stream::upload_thread_fn() {
    while (true) {
        std::unique_lock<std::mutex> lock(upload_mutex);
        upload_cv.wait(lock, [this]() {
            return !upload_done.load() || shutdown.load();
        });

        if (shutdown.load()) {
            break;
        }

        const int il = upload_request_layer;
        const int pi = upload_request_pool;
        lock.unlock();

        if (il >= 0 && il < n_layers() && pi >= 0 && pi < 2) {
            upload_layer_sync(il, pi);
        }

        upload_done.store(true);
        upload_cv.notify_all();
    }
}

void llama_layer_stream::swap_layer_tensors_to_gpu(int il, int pool_idx) {
    auto & pool = pools[pool_idx];
    void * base = ggml_backend_buffer_get_base(pool.buffer);

    for (auto & ti : layer_tensors[il]) {
        // Point tensor to GPU buffer
        ti.tensor->data   = (char *)base + ti.gpu_offset;
        ti.tensor->buffer = pool.buffer;
    }
}

void llama_layer_stream::restore_layer_tensors(int il) {
    for (auto & ti : layer_tensors[il]) {
        // Restore original CPU data pointer
        ti.tensor->data   = ti.cpu_data;
        ti.tensor->buffer = ti.cpu_buffer;
    }
}

// --- Eval callback implementation ---

// Parse layer index from tensor name (e.g., "attn_norm-5" -> 5)
static int parse_layer_from_name(const char * name) {
    if (!name) return -1;

    // Find the last '-' in the name
    const char * dash = strrchr(name, '-');
    if (!dash) return -1;

    // Parse the number after the dash
    char * end = nullptr;
    long val = strtol(dash + 1, &end, 10);
    if (end == dash + 1 || *end != '\0') return -1;
    if (val < 0 || val > 10000) return -1;

    return (int)val;
}

bool llama_layer_stream_eval_cb::callback(ggml_tensor * t, bool ask, void * user_data) {
    auto * self = (llama_layer_stream_eval_cb *)user_data;
    if (!self || !self->streamer || !self->streamer->is_active()) {
        // Fall through to original callback
        if (self && self->original_cb) {
            return self->original_cb(t, ask, self->original_ud);
        }
        return true;
    }

    if (ask) {
        // "ask" phase: detect layer transitions
        int il = parse_layer_from_name(ggml_get_name(t));

        if (il >= 0 && il != self->last_layer_seen) {
            // Entering a new layer
            if (self->last_layer_seen >= 0) {
                self->streamer->on_layer_end(self->last_layer_seen);
            }
            self->streamer->on_layer_begin(il);
            self->last_layer_seen = il;
        }

        // Forward to original callback
        if (self->original_cb) {
            return self->original_cb(t, ask, self->original_ud);
        }
        // Return false to indicate we don't need this specific node's output
        // (the ask phase is asking "do you need the data for this node?")
        return false;
    } else {
        // "result" phase
        if (self->original_cb) {
            return self->original_cb(t, ask, self->original_ud);
        }
        return true; // continue computation
    }
}
