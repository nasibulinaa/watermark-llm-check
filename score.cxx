// Exact per-token logprob scorer for watermark LLR detection.
// For a text T (optionally continued from a prompt P):
//   L = sum over text tokens j=1..k-1 of log p(T_j | P, T_0..T_{j-1})
// Matches the server's generation-logprob convention (logprobs[1:]).
// Strictly llama.cpp: links the same build's libllama/libggml.
//
// n_ctx is computed from the input sizes (not fixed):
//   auto: n_ctx = max over files of (|prompt| + |text|) + 1024, floor 2048.
//   override: --ctx N (N > 0).
// n_batch is clamped to n_ctx (a batch larger than the context is an
// invalid configuration in this build).
//
// Usage: score model.gguf [--prompt promptfile] [--ctx N] file1 [file2 ...]
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

static std::vector<llama_token> load_tokens(const llama_vocab * vocab, const char * path,
                                            bool * ok = nullptr) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "error: cannot open %s\n", path);
        if (ok) *ok = false;
        return {};
    }
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> tokens(text.size() * 2 + 16);
    const int N = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), tokens.data(),
                                 (int32_t) tokens.size(), add_bos, false);
    tokens.resize(N);
    if (ok) *ok = true;
    return tokens;
}

static int score_tokens(llama_context * ctx, const char * path, int n_vocab,
                        const std::vector<llama_token> & text_tokens,
                        const std::vector<llama_token> * prompt_tokens) {
    llama_memory_clear(llama_get_memory(ctx), true);

    const int k = (int) text_tokens.size();
    if (k < 2) {
        std::fprintf(stderr, "warn: %s: only %d tokens, skipped\n", path, k);
        return 0;
    }

    const int n_prompt = prompt_tokens ? (int) prompt_tokens->size() : 0;
    std::vector<llama_token> tokens;
    if (n_prompt > 0) {
        tokens.insert(tokens.end(), prompt_tokens->begin(), prompt_tokens->end());
    }
    tokens.insert(tokens.end(), text_tokens.begin(), text_tokens.end());
    const int N = (int) tokens.size();

    llama_batch batch = llama_batch_init(N, 0, 1);
    batch.n_tokens = N;
    batch.embd = nullptr;
    for (int i = 0; i < N; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1; // logits на всех позициях
    }
    if (llama_decode(ctx, batch)) {
        std::fprintf(stderr, "error: llama_decode failed for %s\n", path);
        llama_batch_free(batch);
        return 1;
    }

    const float * logits = llama_get_logits(ctx); // (N-1) vectors: position 1..N-1
    if (!logits) {
        std::fprintf(stderr, "error: no logits for %s\n", path);
        llama_batch_free(batch);
        return 1;
    }

    double L = 0.0;
    for (int j = 1; j < k; ++j) { // skip first text token, like server logprobs[1:]
        const int pos = n_prompt + j;
        const float * row = logits + (size_t)(pos - 1) * n_vocab;
        const llama_token tok = text_tokens[j];
        float m = -1e30f;
        for (int v = 0; v < n_vocab; ++v) m = std::max(m, row[v]);
        float Z = 0.0f;
        for (int v = 0; v < n_vocab; ++v) Z += std::exp(row[v] - m);
        L += (double) row[tok] - (double) m - std::log((double) Z);
    }

    llama_batch_free(batch);
    std::printf("L=%.6f N=%d %s\n", L, k, path);
    std::fflush(stdout);
    return 0;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: %s model.gguf [--prompt promptfile] [--ctx N] file1 [file2 ...]\n",
                     argv[0]);
        return 1;
    }

    std::string prompt_path;
    int ctx_arg = 0; // 0 = auto
    int first_file = 2;
    for (int a = 2; a < argc; ++a) {
        const std::string s = argv[a];
        if (s == "--prompt" && a + 1 < argc) {
            prompt_path = argv[++a];
        } else if (s == "--ctx" && a + 1 < argc) {
            ctx_arg = std::atoi(argv[++a]);
        } else {
            first_file = a;
            break;
        }
    }
    if (argc <= first_file) {
        std::fprintf(stderr, "error: no text files given\n");
        return 1;
    }

    llama_model_params mp = {};
    mp.n_gpu_layers = 9999; // all on GPU, like the server
    llama_backend_init();
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        std::fprintf(stderr, "error: failed to load model %s\n", argv[1]);
        llama_backend_free();
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // Pre-tokenize all inputs: n_ctx must cover the longest (prompt + text).
    std::vector<std::vector<llama_token> > all_tokens;
    all_tokens.reserve(argc - first_file);
    int max_text_len = 0;
    for (int a = first_file; a < argc; ++a) {
        bool ok = true;
        std::vector<llama_token> t = load_tokens(vocab, argv[a], &ok);
        if (!ok) {
            std::fprintf(stderr, "error: failed to load %s\n", argv[a]);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        max_text_len = std::max(max_text_len, (int) t.size());
        all_tokens.push_back(std::move(t));
    }

    std::vector<llama_token> prompt_tokens;
    const std::vector<llama_token> * pt = nullptr;
    if (!prompt_path.empty()) {
        bool pok = true;
        prompt_tokens = load_tokens(vocab, prompt_path.c_str(), &pok);
        if (!pok) {
            std::fprintf(stderr, "error: failed to load prompt %s\n", prompt_path.c_str());
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        pt = &prompt_tokens;
    }

    // Context: enough for the longest input + margin, floored at 2048.
    const int max_len = max_text_len + (pt ? (int) prompt_tokens.size() : 0);
    const int auto_ctx = std::max(2048, max_len + 1024);
    const int n_ctx = ctx_arg > 0 ? ctx_arg : auto_ctx;
    std::printf("ctx: n_ctx=%d (max input=%d, auto=%d)\n", n_ctx, max_len, auto_ctx);
    std::fflush(stdout);

    llama_context_params cp = {};
    cp.n_ctx = n_ctx;
    cp.n_batch = std::min(4096, n_ctx);
    cp.offload_kqv = false; // KV-кэш в RAM (как --cache-ram у сервера),
                            // экономим VRAM для весов модели
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        std::fprintf(stderr, "error: failed to init context\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    int rc = 0;
    for (size_t a = 0; a < all_tokens.size(); ++a) {
        rc |= score_tokens(ctx, argv[first_file + a], n_vocab, all_tokens[a], pt);
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return rc;
}
