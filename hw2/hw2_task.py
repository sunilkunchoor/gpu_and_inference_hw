import torch
from utils import (
    build_model,
    get_input_ids,
    slow_loop,
    time_generation,
    MODEL_NAME,
    PROFILE_STEPS,
    RESULTS_DIR,
)


def optimized_loop(model, input_ids, n_steps):
    current_ids = input_ids
    past_key_values = None
    generated_tokens = []
    for _ in range(n_steps):
        outputs = model(input_ids=current_ids, past_key_values=past_key_values, use_cache=True)
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        generated_tokens.append(next_token_id.item())
        current_ids = next_token_id.unsqueeze(0)
        past_key_values = outputs.past_key_values
    return generated_tokens


def profile(loop_fn, model, input_ids, trace_name: str):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        loop_fn(model, input_ids, PROFILE_STEPS)
    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_chrome_trace(str(RESULTS_DIR / trace_name))


def generate_optimized(optimized_trace_name: str) -> float:
    # Load model with float16 to save memory bandwidth and utilize tensor cores
    model = build_model(torch.float16)
    input_ids = get_input_ids()
    
    profile(optimized_loop, model, input_ids, optimized_trace_name)
    optimized_elapsed = time_generation(optimized_loop, model, input_ids, "Optimized")
    
    del model
    torch.cuda.empty_cache()
    
    return optimized_elapsed


def main():
    print("=" * 60)
    print("HW2: LLM Inference Optimization")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    print("\n--- Part 1: Slow baseline ---")
    model = build_model(torch.float32)
    input_ids = get_input_ids()
    profile(slow_loop, model, input_ids, "v0_slow_trace.json")
    slow_elapsed = time_generation(slow_loop, model, input_ids, "Slow")
    del model
    torch.cuda.empty_cache()

    print("\n--- Part 2: Optimized ---")
    optimized_elapsed = generate_optimized(optimized_trace_name="v1_optimized_trace.json")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if optimized_elapsed is None or optimized_elapsed <= 0:
        print("generate_optimized() did not return a positive elapsed time; "
              "cannot compute speedup.")
    else:
        speedup = slow_elapsed / optimized_elapsed
        print(f"  Slow:      {slow_elapsed:6.2f}s")
        print(f"  Optimized: {optimized_elapsed:6.2f}s")
        print(f"  Speedup:   {speedup:6.2f}x  (vs V0 slow baseline)")


if __name__ == "__main__":
    main()


# ============================================================================
# Writeup
# ============================================================================
#
# Changes made and speedup per fix:
# - Our combined optimizations reduced generation time from 0.31s (Slow) to 0.24s (Optimized), achieving a 1.30x overall speedup.
# - Enabled KV cache (`use_cache=True` and passing `past_key_values`): Avoids recomputing representations for previous tokens.
# - Removed `torch.cat` of input context: Instead of passing the growing sequence, only the single newly generated token is passed into the model at each step.
# - Switched model dtype from `torch.float32` to `torch.float16`: Halves the memory footprint and doubles the memory bandwidth, allowing Tensor Cores to be used effectively.
#
# Biggest impact and why:
# The KV cache implementation provides the biggest impact. Without it, the model re-computes the keys and values for the entire history of the sequence at every generation step. This results in quadratic time complexity. The KV cache makes generation linear by caching previous states, drastically cutting down redundant computation.
