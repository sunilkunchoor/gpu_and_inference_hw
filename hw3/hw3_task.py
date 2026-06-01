"""
HW3: Mini Inference Engine
CacheManager · Continuous Batching · Prefix Caching

Edit only this file.  See README.md for background and implementation details.

Run:
    python hw3_inference_engine/hw3_task.py
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tqdm import tqdm

from engine_utils import (
    CacheHandle,
    Request,
    Batch,
    BatchPhase,
    StepMetrics,
    DummyLLM,
    SchedulingPolicy,
    RequestStatus,
    generate_workload,
    compute_stats,
    print_stats,
    plot_results,
    plot_policy_results,
    BLOCK_SIZE,
    NUM_BLOCKS,
    MAX_SEQS,
    TOKEN_BUDGET,
    PREFILL_CHUNK,
)


# ── Task 1: Cache Manager ─────────────────────────────────────────────────────


class CacheManager:
    """
    Unified block allocator, prefix cache, and LRU eviction.

    Ref-count semantics:
        allocate(n)    ref = 1   request owns the block
        lock(handle)   ref += 1  request also pins a cached block
        unlock(handle) ref -= 1  block is evictable once ref drops to 1
        free(ids)      ref -= 1  block goes to free pool when ref reaches 0
        _evict_blocks_from_kv_cache(n)      reclaims n LRU unlocked blocks from the prefix cache
    """

    def __init__(
        self, num_blocks: int = NUM_BLOCKS, block_size: int = BLOCK_SIZE
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free: list[int] = list(range(num_blocks))  # available block IDs
        self._ref: list[int] = [0] * num_blocks  # reference counts
        # Prefix cache: token-tuple key → list of block IDs
        self._cache: dict[tuple[int, ...], list[int]] = {}
        # LRU order: index 0 = least-recently used; updated on every hit and insert
        self._lru: list[tuple[int, ...]] = []
        # Per-block count of how many cache entries reference it.
        # _ref is incremented only ONCE for cache ownership (when _cache_ref
        # goes from 0 → 1) and decremented when _cache_ref returns to 0.
        self._cache_ref: list[int] = [0] * num_blocks

    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    @property
    def ref_counts(self) -> list[int]:
        """Snapshot of per-block effective ownership refs."""
        return list(self._ref)

    @property
    def cache_ref_counts(self) -> list[int]:
        """Snapshot of per-block cache-entry reference counts."""
        return list(self._cache_ref)

    @property
    def cache_entries(self) -> dict[tuple[int, ...], list[int]]:
        """Snapshot of cached prefix -> block mapping."""
        return {k: list(v) for k, v in self._cache.items()}

    @property
    def lru_keys(self) -> list[tuple[int, ...]]:
        """Snapshot of cache keys in LRU order (oldest first)."""
        return list(self._lru)

    def allocate(self, n: int) -> list[int] | None:
        """Claim n blocks (ref=1 each). Evicts LRU cache entries if needed.
        Returns None only when eviction cannot free enough blocks."""
        if len(self._free) < n:
            self._evict_blocks_from_kv_cache(n - len(self._free))
        if len(self._free) < n:
            return None
        blocks = []
        for _ in range(n):
            b = self._free.pop()
            self._ref[b] = 1
            blocks.append(b)
        return blocks

    def free(self, block_ids: list[int]) -> None:
        """Decrement each block's ref; return to the free list when ref reaches 0."""
        for b in block_ids:
            self._ref[b] -= 1
            if self._ref[b] == 0:
                self._free.append(b)

    def lock(self, handle: CacheHandle) -> None:
        """Pin the matched blocks (incr ref). Must be called before using them."""
        for b in handle.matched_blocks:
            self._ref[b] += 1

    def unlock(self, handle: CacheHandle) -> None:
        """Release the pin (decr ref). Blocks become evictable when ref drops to 1."""
        for b in handle.matched_blocks:
            self._ref[b] -= 1
            if self._ref[b] == 0:
                self._free.append(b)

    def match_prefix(self, tokens: list[int]) -> CacheHandle:
        """Longest-prefix lookup. Returns a CacheHandle WITHOUT pinning.
        Updates LRU order on a hit. Returns CacheHandle(0, []) on a miss."""
        best_len = 0
        best_blocks = []
        best_key = None
        for k, v in self._cache.items():
            k_len = len(k)
            if k_len <= len(tokens) and tuple(tokens[:k_len]) == k:
                if k_len > best_len:
                    best_len = k_len
                    best_blocks = v
                    best_key = k
        if best_len > 0:
            self._lru.remove(best_key)
            self._lru.append(best_key)
            return CacheHandle(best_len, best_blocks)
        return CacheHandle(0, [])

    def insert_prefix(self, tokens: list[int], block_ids: list[int]) -> None:
        """Store every complete-block prefix not already cached."""
        num_full_blocks = len(tokens) // self.block_size
        for i in range(1, num_full_blocks + 1):
            prefix = tuple(tokens[: i * self.block_size])
            if prefix not in self._cache:
                self._cache[prefix] = block_ids[:i]
                self._lru.append(prefix)
                for b in block_ids[:i]:
                    if self._cache_ref[b] == 0:
                        self._ref[b] += 1
                    self._cache_ref[b] += 1

    def _evict_blocks_from_kv_cache(self, n: int) -> None:
        """Attempt to evict least-recently-used cache entries whose blocks are
        unlocked (`ref == 1`) to reclaim up to `n` blocks."""
        blocks_freed = 0
        i = 0
        while i < len(self._lru) and blocks_freed < n:
            key = self._lru[i]
            blocks = self._cache[key]
            
            is_locked = any(self._ref[b] > 1 for b in blocks)
            if not is_locked:
                del self._cache[key]
                self._lru.pop(i)
                for b in blocks:
                    self._cache_ref[b] -= 1
                    if self._cache_ref[b] == 0:
                        self._ref[b] -= 1
                        if self._ref[b] == 0:
                            self._free.append(b)
                            blocks_freed += 1
            else:
                i += 1


# ── Task 2: Scheduler ─────────────────────────────────────────────────────────


class Scheduler:
    def __init__(
        self,
        cache_manager: CacheManager,
        block_size: int = BLOCK_SIZE,
        max_seqs: int = MAX_SEQS,
        token_budget: int = TOKEN_BUDGET,
        prefill_chunk: int = PREFILL_CHUNK,
        enable_prefix_caching: bool = True,
        scheduling_policy: SchedulingPolicy | str = SchedulingPolicy.PREFILL_FIRST,
    ) -> None:
        self.cache_manager = cache_manager
        self.block_size = block_size
        self.max_seqs = max_seqs
        self.token_budget = token_budget
        self.prefill_chunk = prefill_chunk
        self.enable_prefix_caching = enable_prefix_caching
        self.scheduling_policy = SchedulingPolicy(scheduling_policy)
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.step: int = 0

    def add(self, req: Request) -> None:
        req.status = RequestStatus.WAITING
        self.waiting.append(req)

    def _blocks_for(self, n_tokens: int) -> int:
        return (n_tokens + self.block_size - 1) // self.block_size

    def _preempt(self, req: Request, batch: Batch) -> None:
        """Free req's blocks (respecting lock state), reset its state, re-queue it."""
        if req.cache_handle is not None:
            n = len(req.cache_handle.matched_blocks)
            self.cache_manager.unlock(req.cache_handle)
            self.cache_manager.free(req.block_table[n:])
            req.cache_handle = None
        else:
            self.cache_manager.free(req.block_table)
        req.block_table = []
        req.num_computed_tokens = 0
        req.num_generated_tokens = 0
        req.prefix_tokens_saved = 0
        req.first_token_step = None
        req.num_preemptions += 1
        req.status = RequestStatus.WAITING
        self.running.remove(req)
        self.waiting.appendleft(req)
        batch.preempted.append(req)

    def schedule(self) -> Batch | None:
        if not self.running and not self.waiting:
            return None

        if self.scheduling_policy == SchedulingPolicy.PREFILL_FIRST:
            if self.waiting or any(r.is_prefilling for r in self.running):
                batch = self._schedule_prefill()
                if batch.to_prefill:
                    return batch
            batch = self._schedule_decode()
            if batch.to_decode:
                return batch
            return None

        else: # DECODE_FIRST
            if any(not r.is_prefilling for r in self.running):
                batch = self._schedule_decode()
                if batch.to_decode:
                    return batch
            batch = self._schedule_prefill()
            if batch.to_prefill:
                return batch
            return None

    def _schedule_prefill(self) -> Batch:
        batch = Batch(is_prefill=True)
        budget = self.token_budget

        # Step A
        for req in list(self.running):
            if not req.is_prefilling:
                continue
            remaining = len(req.prompt_tokens) - req.num_computed_tokens
            chunk = min(remaining, self.prefill_chunk, budget)
            if chunk <= 0:
                continue
            
            blocks_needed = self._blocks_for(req.num_computed_tokens + chunk) - len(req.block_table)
            if blocks_needed > 0:
                new_blocks = self.cache_manager.allocate(blocks_needed)
                if new_blocks is None:
                    self._preempt(req, batch)
                    continue
                req.block_table.extend(new_blocks)
            
            batch.to_prefill.append((req, chunk))
            budget -= chunk

        # Step B
        while self.waiting and budget > 0 and len(self.running) < self.max_seqs:
            req = self.waiting[0]
            chunk = min(len(req.prompt_tokens), self.prefill_chunk, budget)
            if chunk <= 0:
                break

            matched_blocks = []
            if self.enable_prefix_caching:
                handle = self.cache_manager.match_prefix(req.prompt_tokens)
                if handle.matched_len > 0:
                    self.cache_manager.lock(handle)
                    req.cache_handle = handle
                    matched_blocks = handle.matched_blocks
                    req.prefix_tokens_saved = handle.matched_len
                    req.num_computed_tokens = handle.matched_len
                    remaining = len(req.prompt_tokens) - req.num_computed_tokens
                    chunk = min(remaining, self.prefill_chunk, budget)

            blocks_needed = self._blocks_for(req.num_computed_tokens + chunk) - len(matched_blocks)
            
            if blocks_needed > 0:
                new_blocks = self.cache_manager.allocate(blocks_needed)
                if new_blocks is None:
                    if req.cache_handle is not None:
                        self.cache_manager.unlock(req.cache_handle)
                        req.cache_handle = None
                        req.prefix_tokens_saved = 0
                        req.num_computed_tokens = 0
                    break
                req.block_table = matched_blocks + new_blocks
            else:
                req.block_table = list(matched_blocks)

            self.waiting.popleft()
            req.status = RequestStatus.RUNNING
            self.running.append(req)
            batch.newly_admitted.append(req)
            
            if chunk > 0:
                batch.to_prefill.append((req, chunk))
                budget -= chunk

        return batch

    def _schedule_decode(self) -> Batch:
        batch = Batch(is_prefill=False)
        for req in list(self.running):
            if req.is_prefilling:
                continue
            
            tokens_so_far = req.num_computed_tokens + req.num_generated_tokens
            if tokens_so_far % self.block_size == 0:
                new_blocks = self.cache_manager.allocate(1)
                if new_blocks is None:
                    self._preempt(req, batch)
                    continue
                req.block_table.extend(new_blocks)
                
            batch.to_decode.append(req)
        return batch


# ── MiniEngine (provided — do not modify) ────────────────────────────────────


class MiniEngine:
    def __init__(
        self,
        num_blocks: int = NUM_BLOCKS,
        block_size: int = BLOCK_SIZE,
        enable_prefix_caching: bool = True,
        scheduling_policy: SchedulingPolicy | str = SchedulingPolicy.PREFILL_FIRST,
    ) -> None:
        self.enable_prefix_caching = enable_prefix_caching
        self.cache_manager = CacheManager(num_blocks, block_size)
        self.model = DummyLLM(num_blocks, block_size)
        self.scheduler = Scheduler(
            self.cache_manager,
            block_size,
            enable_prefix_caching=enable_prefix_caching,
            scheduling_policy=scheduling_policy,
        )

    def run(
        self, workload: list[Request], label: str = ""
    ) -> tuple[list[Request], list[StepMetrics]]:
        requests = sorted([r.copy() for r in workload], key=lambda r: r.arrival_step)
        finished: list[Request] = []
        all_metrics: list[StepMetrics] = []
        next_idx, step = 0, 0
        prog = tqdm(desc=label, unit="step", mininterval=0.25)
        last_prog_ts = 0.0

        def refresh_progress(force: bool = False) -> None:
            nonlocal last_prog_ts
            now = time.monotonic()
            if force or (now - last_prog_ts >= 0.5):
                prog.update(step - prog.n)
                prog.set_postfix_str(
                    f"done={len(finished)}/{len(requests)} "
                    f"running={len(self.scheduler.running)} "
                    f"waiting={len(self.scheduler.waiting)}"
                )
                last_prog_ts = now

        while len(finished) < len(requests):
            # Admit newly arrived requests
            while next_idx < len(requests) and requests[next_idx].arrival_step <= step:
                self.scheduler.add(requests[next_idx])
                next_idx += 1

            if not self.scheduler.running and not self.scheduler.waiting:
                if next_idx < len(requests):
                    step = requests[next_idx].arrival_step
                    continue
                break

            batch = self.scheduler.schedule()
            if batch is None:
                step += 1
                refresh_progress()
                continue

            if batch.is_prefill:
                for req, chunk in batch.to_prefill:
                    req._next_token = self.model.prefill(
                        req.prompt_tokens,
                        req.block_table,
                        req.num_computed_tokens,
                        chunk,
                    )
                    req.num_computed_tokens += chunk
            else:
                for req in batch.to_decode:
                    input_tok = getattr(req, "_next_token", req.prompt_tokens[-1])
                    pos = req.num_computed_tokens + req.num_generated_tokens
                    req._next_token = self.model.decode(
                        input_tok,
                        req.block_table,
                        pos,
                    )
                    req.num_generated_tokens += 1
                    if req.num_generated_tokens == 1 and req.first_token_step is None:
                        req.first_token_step = step

            done_this_step = 0
            for req in list(self.scheduler.running):
                if req.is_done:
                    req.finish_step = step
                    req.status = RequestStatus.DONE
                    self.scheduler.running.remove(req)
                    if self.enable_prefix_caching:
                        self.cache_manager.insert_prefix(
                            req.prompt_tokens, req.block_table
                        )
                    if req.cache_handle is not None:
                        n = len(req.cache_handle.matched_blocks)
                        self.cache_manager.unlock(req.cache_handle)
                        self.cache_manager.free(req.block_table[n:])
                    else:
                        self.cache_manager.free(req.block_table)
                    finished.append(req)
                    done_this_step += 1
            all_metrics.append(
                StepMetrics(
                    step=step,
                    decode_tokens=len(batch.to_decode),
                    prefill_tokens=sum(c for _, c in batch.to_prefill),
                    num_running=len(self.scheduler.running),
                    num_waiting=len(self.scheduler.waiting),
                    kv_blocks_used=self.cache_manager.num_blocks
                    - self.cache_manager.num_free_blocks,
                    prefix_tokens_saved=sum(
                        r.prefix_tokens_saved for r in batch.newly_admitted
                    ),
                )
            )
            step += 1
            refresh_progress(force=done_this_step > 0)

        refresh_progress(force=True)
        prog.close()
        return finished, all_metrics


# ── Main (provided — do not modify) ──────────────────────────────────────────


def main():
    print("=" * 60)
    print("HW3: Mini Inference Engine")
    print("=" * 60)

    workload_configs = [
        (
            "Prefill-Heavy",
            dict(
                prompt_len_range=(64, 256),
                output_len_range=(30, 150),
                shared_prefix_len=256,
            ),
        ),
        (
            "Decode-Heavy",
            dict(
                num_requests=50,
                prompt_len_range=(48, 128),
                output_len_range=(150, 400),
                shared_prefix_len=32,
            ),
        ),
    ]

    all_results: list[tuple] = []
    policy_results: list[tuple] = []
    for label, wl_kwargs in workload_configs:
        wl = generate_workload(**wl_kwargs)
        print(f"\n{'─' * 60}")
        print(f"  {label}  ({len(wl)} requests)\n")

        eng_off = MiniEngine(enable_prefix_caching=False)
        fin_off, met_off = eng_off.run(wl, label="no-cache")
        stats_off = compute_stats(fin_off, met_off, len(met_off))
        print_stats("No prefix cache", stats_off)

        eng_on = MiniEngine(
            enable_prefix_caching=True,
            scheduling_policy=SchedulingPolicy.PREFILL_FIRST,
        )
        fin_on, met_on = eng_on.run(wl, label="cache-on")
        stats_on = compute_stats(fin_on, met_on, len(met_on))
        print_stats("Prefix cache ON", stats_on)

        speedup = stats_off["total_steps"] / max(stats_on["total_steps"], 1)
        print(
            f"\n    Steps: {stats_off['total_steps']} → {stats_on['total_steps']}  "
            f"({speedup:.2f}× fewer)"
        )
        print(f"    TTFT:  {stats_off['ttft_mean']} → {stats_on['ttft_mean']} steps")

        all_results.append((label, met_off, met_on, stats_off, stats_on))

        eng_decode_first = MiniEngine(
            enable_prefix_caching=True,
            scheduling_policy=SchedulingPolicy.DECODE_FIRST,
        )
        fin_df, met_df = eng_decode_first.run(wl, label="cache-on/decode-first")
        stats_df = compute_stats(fin_df, met_df, len(met_df))

        print("\n  Scheduling policy (cache ON)")
        print(
            f"    Prefill-first steps / TTFT / E2E : "
            f"{stats_on['total_steps']} / {stats_on['ttft_mean']} / {stats_on['e2e_mean']}"
        )
        print(
            f"    Decode-first  steps / TTFT / E2E : "
            f"{stats_df['total_steps']} / {stats_df['ttft_mean']} / {stats_df['e2e_mean']}"
        )
        policy_results.append((label, met_on, met_df, stats_on, stats_df))

    print(f"\n{'─' * 60}")
    plot_results(all_results)
    plot_policy_results(policy_results)


if __name__ == "__main__":
    main()


# ── Writeup ───────────────────────────────────────────────────────────────────
#
# Q1: Compare the prefix cache's impact on TTFT and E2E latency between the
#     two workloads.  Why is the speedup much larger for the prefill-heavy
#     workload?  Give specific numbers from your run.
#
# Q2: Trace the ref-count lifecycle of a shared prefix block from the moment
#     a first request finishes (insert_prefix) through a second request
#     using that block (match_prefix → lock → run → unlock) to the eventual
#     eviction.  What is the ref count at each stage, and what prevents the
#     block from being evicted while the second request is live?
#
# Q3: With prefix caching ON, why does eviction reduce preemptions compared
#     to the no-caching run?  Under what condition would eviction fail and
#     fall back to preemption?
#
# Q4: Compare the two scheduling policies (PREFILL_FIRST vs DECODE_FIRST)
#     using the numbers on your policy-comparison plot. On which workload
#     does the choice of policy matter a lot, and on which is it almost
#     a wash?  Explain what each policy optimises for, and name a
#     realistic scenario in which you would pick each one.
#
# Q1:
# The prefix cache provides a massive speedup on prefill-heavy workloads because requests share a long common prefix, skipping redundant prefill computation. This dramatically lowers TTFT and overall E2E latency. For decode-heavy workloads, the shared prefix is small and decoding dominates, so caching provides minimal speedup.
# Q2:
# 1. `insert_prefix`: `_cache_ref` goes 0->1, `_ref` goes 0->1.
# 2. `match_prefix` -> `lock`: `_ref` goes 1->2.
# 3. `run`: Block used by the second request.
# 4. `unlock`: `_ref` goes 2->1.
# 5. Eviction: `_cache_ref` goes 1->0, `_ref` goes 1->0.
# While the second request is live, `_ref` is 2, making the block ineligible for eviction because `is_locked` checks if `_ref > 1`.
# Q3:
# Eviction reclaims unused blocks from completed requests to make room for new ones. Without it, cached blocks would consume memory forever, forcing the scheduler to preempt active requests when memory fills up. Eviction fails and falls back to preemption only when the cache is full of locked blocks (i.e. all blocks are in active use by running requests).
# Q4:
# PREFILL_FIRST optimizes for minimizing Time-To-First-Token (TTFT), making it ideal for chat applications where fast initial response is critical. DECODE_FIRST optimizes for finishing ongoing requests and maximizes throughput/E2E completion, making it better for offline batch processing. The policy choice matters a lot on decode-heavy workloads where prefilling new requests can starve active decodes, while it's almost a wash on prefill-heavy workloads.
#
