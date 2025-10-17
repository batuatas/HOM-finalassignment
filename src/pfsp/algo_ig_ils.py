# src/pfsp/algo_ig_ils.py
from __future__ import annotations
import random, time, json
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List, Dict, Any
import numpy as np
from .operators import Operators, makespan, _best_insert_pos_with_ties
from .mechanisms import build_scheduler, get_mechanism
from .scheduler import QLearningScheduler

@dataclass
class IGILSResult:
    permutation: List[int]
    makespan: int
    iterations: int

# ---- NEH with deadline & limited candidate orders ----
def _neh_build_from_order(p_times: np.ndarray, order: np.ndarray, deadline: Optional[float], logger: Optional[Callable[[Dict[str, Any]], None]]) -> np.ndarray:
    seq = np.empty(0, dtype=np.int64)
    for k, job in enumerate(order):
        if deadline and (k % 4 == 0) and time.time() >= deadline:
            if logger: logger({"event":"neh_deadline_partial", "placed": int(k), "remaining": int(order.size-k)})
            # fast append remaining (no insert search)
            tail = order[k:]
            seq = np.concatenate([seq, tail])
            return seq
        best_pos, _ = _best_insert_pos_with_ties(p_times, seq, int(job), deadline=deadline)
        seq = np.insert(seq, best_pos, int(job))
        if logger and (k % 10 == 0):
            logger({"event":"neh_step", "k": int(k), "job": int(job), "seq_len": int(seq.size)})
    return seq

def _candidate_orders(p_times: np.ndarray) -> List[np.ndarray]:
    m, _ = p_times.shape
    total = np.sum(p_times, axis=0)
    o1 = np.argsort(-total)
    w_front = np.linspace(m, 1, m, dtype=np.float64)
    o2 = np.argsort(-(w_front[:, None] * p_times).sum(axis=0))
    w_back = np.linspace(1, m, m, dtype=np.float64)
    o3 = np.argsort(-(w_back[:, None] * p_times).sum(axis=0))
    maxp = np.max(p_times, axis=0)
    o4 = np.argsort(-maxp)
    orders: List[np.ndarray] = []
    seen = set()
    for o in (o1, o2, o3, o4):
        key = tuple(o.tolist())
        if key not in seen:
            orders.append(o.astype(np.int64)); seen.add(key)
    return orders

def _neh_multi_start(p_times: np.ndarray, neh_orders: int, deadline: Optional[float], logger: Optional[Callable[[Dict[str, Any]], None]]) -> np.ndarray:
    best_seq: Optional[np.ndarray] = None
    best_val: Optional[int] = None
    for idx, order in enumerate(_candidate_orders(p_times)[:max(1, neh_orders)]):
        seq = _neh_build_from_order(p_times, order, deadline=deadline, logger=logger)
        val = int(makespan(seq, p_times))
        if logger: logger({"event":"neh_candidate_done", "order_idx": int(idx), "value": int(val)})
        if (best_val is None) or (val < best_val):
            best_val = val; best_seq = seq
        if deadline and time.time() >= deadline:
            if logger: logger({"event":"neh_deadline_exit", "best": int(best_val)})
            break
    assert best_seq is not None
    return best_seq

class IteratedGreedyILS:
    def __init__(
        self,
        p_times: np.ndarray,
        mechanism: str = "fixed",
        # RL/scheduler
        window_size: int = 50,
        p_min: float = 0.10,
        learning_rate: float = 0.30,
        gamma: float = 0.60,
        episode_len: int = 50,
        # search
        block_lengths: Tuple[int, ...] = (2, 3),
        d_frac: float = 0.25,
        ls_step_cap: int = 2000,
        ls_stagnation_limit: int = 200,
        neh_orders: int = 4,
        # tracing
        logger: Optional[Callable[[Dict[str, Any]], None]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.p_times = np.ascontiguousarray(p_times, dtype=np.int64)
        self.ops = Operators(self.p_times)
        if seed is not None:
            random.seed(seed); np.random.seed(seed)
        self.mech_key = mechanism
        self.mech_spec = get_mechanism(mechanism)
        self.op_names = list(self.mech_spec.design.operators)
        self.scheduler = build_scheduler(
            mechanism,
            self.op_names,
            options={
                "window_size": window_size,
                "p_min": p_min,
                "learning_rate": learning_rate,
                "gamma": gamma,
                "episode_len": episode_len,
            },
        )
        self.block_lengths = block_lengths
        self.d_frac = float(d_frac)
        self.ls_step_cap = int(ls_step_cap)
        self.ls_stagnation_limit = int(ls_stagnation_limit)
        self.neh_orders = int(max(1, neh_orders))
        self.logger = logger

    def _log(self, event: str, **fields: Any) -> None:
        if self.logger:
            payload = {"event": event, **fields}
            try: self.logger(payload)
            except Exception: pass

    def _local_search_fixed(self, perm: np.ndarray, value: int, deadline: Optional[float]) -> Tuple[np.ndarray, int]:
        current_perm = np.ascontiguousarray(perm, dtype=np.int64)
        current_val = int(value)
        while True:
            if deadline and time.time() >= deadline:
                break
            improved = False
            neigh = self.ops.best_improvement_relocate(current_perm, deadline=deadline)
            if neigh is not None and neigh[1] < current_val:
                current_perm, current_val = neigh; improved = True; self._log("improve", op="relocate", val=int(current_val))
            if deadline and time.time() >= deadline:
                break
            neigh = self.ops.best_improvement_adjacent_swap(current_perm, deadline=deadline)
            if neigh is not None and neigh[1] < current_val:
                current_perm, current_val = neigh; improved = True; self._log("improve", op="swap", val=int(current_val))
            if deadline and time.time() >= deadline:
                break
            neigh = self.ops.best_improvement_block(current_perm, block_lengths=self.block_lengths, deadline=deadline)
            if neigh is not None and neigh[1] < current_val:
                current_perm, current_val = neigh; improved = True; self._log("improve", op="block", val=int(current_val))
            if not improved:
                break
        return current_perm, current_val

    def _local_search_adaptive(self, perm: np.ndarray, value: int, deadline: Optional[float]) -> Tuple[np.ndarray, int]:
        current_perm = np.ascontiguousarray(perm, dtype=np.int64)
        best_val = int(value)
        best_perm = current_perm.copy()
        no_improve = 0
        steps_without_progress = 0
        for step in range(self.ls_step_cap):
            if deadline and time.time() >= deadline:
                break
            base_val = makespan(current_perm, self.p_times)
            delta_frac = 0.0 if base_val == 0 else (best_val - base_val) / float(max(best_val, 1))
            if isinstance(self.scheduler, QLearningScheduler):
                s = self.scheduler.state_from_signals(delta_frac, no_improve)
                op = self.scheduler.select(s)
            else:
                op = self.op_names[0]; s = (0, 0)
            if op == "relocate":
                neigh = self.ops.best_improvement_relocate(current_perm, deadline=deadline)
            elif op == "swap":
                neigh = self.ops.best_improvement_adjacent_swap(current_perm, deadline=deadline)
            else:
                neigh = self.ops.best_improvement_block(current_perm, block_lengths=self.block_lengths, deadline=deadline)

            if neigh is not None and neigh[1] < base_val:
                current_perm, new_val = neigh
                reward = (base_val - new_val) / float(base_val)
                no_improve = 0
                self._log("improve", op=op, step=int(step), val=int(new_val), reward=float(reward))
            else:
                new_val = base_val
                reward = -0.001
                no_improve += 1
                self._log("no_improve", op=op, step=int(step), val=int(new_val))

            if isinstance(self.scheduler, QLearningScheduler):
                delta_frac_next = 0.0 if new_val == 0 else (best_val - new_val) / float(max(best_val, 1))
                s_next = self.scheduler.state_from_signals(delta_frac_next, no_improve)
                self.scheduler.update(s, op, reward, s_next)

            if new_val < best_val:
                best_val = int(new_val)
                best_perm = current_perm.copy()
                steps_without_progress = 0
                self._log("best_update", val=int(best_val))
            else:
                steps_without_progress += 1
                if steps_without_progress >= self.ls_stagnation_limit:
                    self._log("ls_stagnation_stop", best=int(best_val))
                    break

        return best_perm, int(best_val)

    def run(
        self,
        max_iter: int = 1000,
        max_no_improve: int = 50,
        time_limit: Optional[float] = None,
        verbose: bool = False,
        progress_cb: Optional[Callable[[int, int], None]] = None,
        progress_every: int = 10,
    ) -> IGILSResult:
        start = time.time()
        deadline = (start + time_limit) if time_limit is not None else None

        self._log("start", mechanism=self.mech_key, time_limit=time_limit)

        # NEH start (time-bounded)
        perm = _neh_multi_start(self.p_times, neh_orders=self.neh_orders, deadline=deadline, logger=self.logger)
        val = makespan(perm, self.p_times)
        self._log("neh_done", val=int(val), n=int(perm.size))

        best_perm = perm.copy()
        best_val = int(val)
        no_improve = 0
        iterations = 0
        n = perm.shape[0]

        for it in range(max_iter):
            iterations = it + 1
            if deadline and time.time() >= deadline:
                if verbose: print(f"[stop] time limit at iter {it}")
                self._log("time_stop", iter=int(it), best=int(best_val))
                break

            if self.mech_key == "adaptive":
                perm, val = self._local_search_adaptive(perm, int(val), deadline=deadline)
            else:
                perm, val = self._local_search_fixed(perm, int(val), deadline=deadline)

            if val < best_val:
                best_perm, best_val = perm.copy(), int(val)
                no_improve = 0
                if verbose: print(f"[{it}] best={best_val}")
                if progress_cb: progress_cb(iterations, best_val)
                self._log("iter_best", iter=int(it), best=int(best_val))
            else:
                no_improve += 1
                if no_improve >= max_no_improve:
                    if verbose: print(f"[stop] no improvement in {max_no_improve} iterations")
                    self._log("stagnation_stop", iter=int(it), best=int(best_val))
                    break

            if progress_cb and (it % max(1, progress_every) == 0):
                progress_cb(iterations, best_val)

            if deadline and time.time() >= deadline:
                self._log("time_stop_after_ls", iter=int(it), best=int(best_val))
                break

            # Diversify
            if   no_improve < 10: d = max(2, int(round(self.d_frac * n)))
            elif no_improve < 20: d = max(2, int(round(1.25 * self.d_frac * n)))
            else:                 d = max(2, int(round(1.50 * self.d_frac * n)))
            perm, val = self.ops.destroy_repair(perm, d=d, deadline=deadline)
            self._log("destroy_repair", d=int(d), val=int(val))

            if deadline and time.time() >= deadline:
                self._log("time_stop_after_dr", iter=int(it), best=int(best_val))
                break

            if no_improve in (20, 40):
                rr_ratio = 0.20 if no_improve == 20 else 0.35
                perm, val = self.ops.ruin_recreate(best_perm, ratio=rr_ratio, deadline=deadline)
                self._log("ruin_recreate", ratio=float(rr_ratio), val=int(val))

        # Final polish if time allows
        if not (deadline and time.time() >= deadline):
            if self.mech_key == "adaptive":
                final_perm, final_val = self._local_search_adaptive(best_perm, best_val, deadline=deadline)
            else:
                final_perm, final_val = self._local_search_fixed(best_perm, best_val, deadline=deadline)
            if final_val < best_val:
                best_perm, best_val = final_perm, final_val
                self._log("final_best", best=int(best_val))

        self._log("end", best=int(best_val), iterations=int(iterations))
        return IGILSResult(permutation=best_perm.tolist(), makespan=int(best_val), iterations=int(iterations))
