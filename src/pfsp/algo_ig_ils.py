# src/pfsp/algo_ig_ils.py
"""IG/ILS for PFSP with:
- Mechanism 1A: deterministic VND w/ granular neighborhoods + IG diversification
- Mechanism 2B: Q-learning operator choice (richer state), same neighborhoods
All phases are deadline aware; optional tracing via logger(ev: dict).
"""
from __future__ import annotations
import random, time
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

# ---- NEH (deadline aware) ----
def _neh_orders(p_times: np.ndarray) -> List[np.ndarray]:
    m, _ = p_times.shape
    total = np.sum(p_times, axis=0)
    o1 = np.argsort(-total)
    w_front = np.linspace(m, 1, m, dtype=np.float64)
    o2 = np.argsort(-(w_front[:, None] * p_times).sum(axis=0))
    w_back = np.linspace(1, m, m, dtype=np.float64)
    o3 = np.argsort(-(w_back[:, None] * p_times).sum(axis=0))
    maxp = np.max(p_times, axis=0)
    o4 = np.argsort(-maxp)
    out: List[np.ndarray] = []
    seen = set()
    for o in (o1, o2, o3, o4):
        key = tuple(o.tolist())
        if key not in seen:
            out.append(o.astype(np.int64)); seen.add(key)
    return out

def _neh_build(p_times: np.ndarray, order: np.ndarray, deadline: Optional[float], log: Optional[Callable[[Dict[str, Any]], None]]) -> np.ndarray:
    seq = np.empty(0, dtype=np.int64)
    for k, job in enumerate(order):
        if deadline and (k % 4 == 0) and time.time() >= deadline:
            if log: log({"event":"neh_deadline_partial","placed":int(k)})
            seq = np.concatenate([seq, order[k:]])
            return seq
        pos, _ = _best_insert_pos_with_ties(p_times, seq, int(job), deadline=deadline)
        seq = np.insert(seq, pos, int(job))
        if log and (k % 10 == 0): log({"event":"neh_step","k":int(k)})
    return seq

def _neh_multi(p_times: np.ndarray, max_orders: int, deadline: Optional[float], log: Optional[Callable[[Dict[str, Any]], None]]) -> np.ndarray:
    best_seq: Optional[np.ndarray] = None
    best_val: Optional[int] = None
    for idx, order in enumerate(_neh_orders(p_times)[:max(1, max_orders)]):
        seq = _neh_build(p_times, order, deadline, log)
        val = int(makespan(seq, p_times))
        if log: log({"event":"neh_candidate","idx":int(idx),"val":int(val)})
        if (best_val is None) or (val < best_val):
            best_val = val; best_seq = seq
        if deadline and time.time() >= deadline:
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
        tau: float = 0.10,
        optimistic_init: float = 0.01,
        # neighborhoods
        block_lengths: Tuple[int, ...] = (2, 3),
        granular_window: int = 10,
        critical_only: bool = True,
        critical_take_frac: float = 0.25,
        d_frac: float = 0.25,
        # LS bounds
        ls_step_cap: int = 2000,
        ls_stagnation_limit: int = 200,
        # NEH options
        neh_orders: int = 4,
        # logging
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
                "tau": tau,
                "optimistic_init": optimistic_init,
            },
        )
        self.block_lengths = block_lengths
        self.granular_window = int(granular_window)
        self.critical_only = bool(critical_only)
        self.critical_take_frac = float(critical_take_frac)
        self.d_frac = float(d_frac)
        self.ls_step_cap = int(ls_step_cap)
        self.ls_stagnation_limit = int(ls_stagnation_limit)
        self.neh_orders = int(max(1, neh_orders))
        self.log = logger

    def _log(self, event: str, **fields: Any) -> None:
        if self.log:
            try: self.log({"event": event, **fields})
            except Exception: pass

    # ---- Deterministic VND (1A) with granular neighborhoods ----
    def _ls_fixed(self, perm: np.ndarray, value: int, deadline: Optional[float]) -> Tuple[np.ndarray, int]:
        curp = np.ascontiguousarray(perm, dtype=np.int64)
        curv = int(value)
        while True:
            if deadline and time.time() >= deadline:
                break
            improved = False
            # relocate
            neigh = self.ops.best_improvement_relocate(
                curp, deadline=deadline,
                granular_window=self.granular_window,
                critical_only=self.critical_only,
                critical_take_frac=self.critical_take_frac,
            )
            if neigh is not None and neigh[1] < curv:
                curp, curv = neigh; improved = True; self._log("improve", op="relocate", val=int(curv))
            if deadline and time.time() >= deadline: break
            # adj swap
            neigh = self.ops.best_improvement_adjacent_swap(
                curp, deadline=deadline,
                critical_only=self.critical_only,
                critical_take_frac=self.critical_take_frac,
            )
            if neigh is not None and neigh[1] < curv:
                curp, curv = neigh; improved = True; self._log("improve", op="swap", val=int(curv))
            if deadline and time.time() >= deadline: break
            # block
            neigh = self.ops.best_improvement_block(
                curp, block_lengths=self.block_lengths, deadline=deadline,
                granular_window=self.granular_window,
                critical_only=self.critical_only,
                critical_take_frac=self.critical_take_frac,
            )
            if neigh is not None and neigh[1] < curv:
                curp, curv = neigh; improved = True; self._log("improve", op="block", val=int(curv))
            if not improved:
                break
        return curp, curv

    # ---- Q-learning guided LS (2B) ----
    def _ls_adaptive(self, perm: np.ndarray, value: int, deadline: Optional[float], iter_frac_provider: Callable[[], float]) -> Tuple[np.ndarray, int]:
        curp = np.ascontiguousarray(perm, dtype=np.int64)
        best_val = int(value)
        best_perm = curp.copy()
        no_improve = 0
        stagn_steps = 0
        for step in range(self.ls_step_cap):
            if deadline and time.time() >= deadline:
                break
            base_val = makespan(curp, self.p_times)
            delta_frac = 0.0 if base_val == 0 else (best_val - base_val) / float(max(best_val, 1))
            iter_frac = float(max(0.0, min(1.0, iter_frac_provider())))
            if isinstance(self.scheduler, QLearningScheduler):
                s = self.scheduler.state_from_signals(delta_frac, no_improve, iter_frac)
                op = self.scheduler.select(s)
            else:
                s = (0, 0, 0); op = self.op_names[0]

            if op == "relocate":
                neigh = self.ops.best_improvement_relocate(
                    curp, deadline=deadline,
                    granular_window=self.granular_window,
                    critical_only=self.critical_only,
                    critical_take_frac=self.critical_take_frac,
                )
            elif op == "swap":
                neigh = self.ops.best_improvement_adjacent_swap(
                    curp, deadline=deadline,
                    critical_only=self.critical_only,
                    critical_take_frac=self.critical_take_frac,
                )
            else:
                neigh = self.ops.best_improvement_block(
                    curp, block_lengths=self.block_lengths, deadline=deadline,
                    granular_window=self.granular_window,
                    critical_only=self.critical_only,
                    critical_take_frac=self.critical_take_frac,
                )

            if neigh is not None and neigh[1] < base_val:
                curp, new_val = neigh
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
                s_next = self.scheduler.state_from_signals(delta_frac_next, no_improve, iter_frac)
                self.scheduler.update(s, op, reward, s_next)

            if new_val < best_val:
                best_val = int(new_val)
                best_perm = curp.copy()
                stagn_steps = 0
                self._log("best_update", val=int(best_val))
            else:
                stagn_steps += 1
                if stagn_steps >= self.ls_stagnation_limit:
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
        t0 = time.time()
        deadline = (t0 + time_limit) if time_limit is not None else None

        # NEH init (deadline aware)
        if verbose: print("[neh] building initial solution…")
        perm = _neh_multi(self.p_times, max_orders=self.neh_orders, deadline=deadline, log=self.log)
        val = makespan(perm, self.p_times)
        self._log("neh_done", val=int(val), n=int(perm.size))

        best_perm = perm.copy()
        best_val = int(val)
        no_improve = 0
        iterations = 0
        n = perm.shape[0]

        def iter_frac_provider() -> float:
            return iterations / float(max(1, max_iter))

        for it in range(max_iter):
            iterations = it + 1
            if deadline and time.time() >= deadline:
                if verbose: print(f"[stop] time limit at iter {it}")
                self._log("time_stop", iter=int(it), best=int(best_val))
                break

            # Intensification
            if self.mech_key == "adaptive":
                perm, val = self._ls_adaptive(perm, int(val), deadline=deadline, iter_frac_provider=iter_frac_provider)
            else:
                perm, val = self._ls_fixed(perm, int(val), deadline=deadline)

            # Update best
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

            # Diversification: destroy–repair intensity grows with stagnation
            if   no_improve < 10: d = max(2, int(round(self.d_frac * n)))
            elif no_improve < 20: d = max(2, int(round(1.25 * self.d_frac * n)))
            else:                 d = max(2, int(round(1.50 * self.d_frac * n)))
            perm, val = self.ops.destroy_repair(perm, d=d, deadline=deadline)
            self._log("destroy_repair", d=int(d), val=int(val))

            if deadline and time.time() >= deadline:
                self._log("time_stop_after_dr", iter=int(it), best=int(best_val))
                break

            # Occasional ruin–recreate around incumbent
            if no_improve in (20, 40):
                rr_ratio = 0.20 if no_improve == 20 else 0.35
                perm, val = self.ops.ruin_recreate(best_perm, ratio=rr_ratio, deadline=deadline)
                self._log("ruin_recreate", ratio=float(rr_ratio), val=int(val))

        # Final polish if time left
        if not (deadline and time.time() >= deadline):
            if self.mech_key == "adaptive":
                final_perm, final_val = self._ls_adaptive(best_perm, best_val, deadline=deadline, iter_frac_provider=iter_frac_provider)
            else:
                final_perm, final_val = self._ls_fixed(best_perm, best_val, deadline=deadline)
            if final_val < best_val:
                best_perm, best_val = final_perm, final_val
                self._log("final_best", best=int(best_val))

        return IGILSResult(permutation=best_perm.tolist(), makespan=int(best_val), iterations=int(iterations))
