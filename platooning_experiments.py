#!/usr/bin/env python3
"""
Numerical experiments for the multi-company truck platooning paper (Section 6).

Pipeline
  1. Read the road network and the truck trips.
  2. Build the itineraries of every truck (route + departure slot)        [Section 3, Appendix A.3]
  3. Compute v'(S) for every coalition S with the compact MIP             [Appendix A.1-A.2]
  4. From the table of v'(S): system optimum, Shapley benchmark,
     Shapley-envelope Moulin mechanism, Greedy mechanism                  [Sections 4-5]
  5. Report efficiency, participation, surplus, stability, bounds, runtime.

Usage
  python platooning_experiments.py --config config.yaml
  python platooning_experiments.py --config config.yaml --scenarios 3 4 --instances 2

Requires Python >= 3.8, numpy, scipy >= 1.9, networkx, pyyaml.
The MIPs are solved with HiGHS (through scipy.optimize.milp).
"""
import argparse
import csv
import itertools
import json
import math
import os
import random
import time
from collections import defaultdict

import networkx as nx
import numpy as np
import yaml
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix

EPS = 1e-9


# ----------------------------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------------------------
def load_network(path):
    """CSV with header from,to,miles. Every row is a two-way road segment."""
    G = nx.DiGraph()
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            u, v, miles = row["from"].strip(), row["to"].strip(), float(row["miles"])
            G.add_edge(u, v, miles=miles)
            G.add_edge(v, u, miles=miles)
    return G


def load_trips(path, cfg, rng):
    """CSV with header truck_id,origin,destination[,pref_departure_min].
    If pref_departure_min is missing, preferred departure times are drawn uniformly
    from [day_start_min, day_end_min] (config), with the given random generator."""
    trips = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            pref = row.get("pref_departure_min")
            if pref is None or str(pref).strip() == "":
                pref = rng.uniform(cfg["day_start_min"], cfg["day_end_min"])
            trips.append({"id": row["truck_id"].strip(), "o": row["origin"].strip(),
                          "d": row["destination"].strip(), "pref": float(pref)})
    return trips


# ----------------------------------------------------------------------------------------
# 2. Itineraries
# ----------------------------------------------------------------------------------------
class Itinerary(object):
    __slots__ = ("truck", "cost", "pairs")

    def __init__(self, truck, cost, pairs):
        self.truck = truck      # truck index
        self.cost = cost        # deviation cost ($)
        self.pairs = pairs      # tuple of (u, v, entry_slot)


def build_itineraries(G, trips, cfg):
    """Returns ITIN[f] = list of Itinerary for truck f, and LEN[(u, v)] = miles."""
    delta = float(cfg["delta_min"])
    speed = float(cfg["speed_mph"])
    LEN = {(u, v): d["miles"] for u, v, d in G.edges(data=True)}
    slots = {e: max(1, int(round(m / speed * 60.0 / delta))) for e, m in LEN.items()}
    w = int(round(cfg["departure_window_min"] / delta))      # half-width of departure window
    slack = int(round(cfg["arrival_slack_min"] / delta))     # allowed lateness at destination
    route_cache = {}
    ITIN = []
    for f, tr in enumerate(trips):
        od = (tr["o"], tr["d"])
        if od not in route_cache:
            routes = []
            gen = nx.shortest_simple_paths(G, od[0], od[1], weight="miles")
            for path in itertools.islice(gen, int(cfg["k_routes"])):
                miles = sum(LEN[(path[i], path[i + 1])] for i in range(len(path) - 1))
                if routes and miles > (1.0 + cfg["rho"]) * routes[0][1] + EPS:
                    break
                routes.append((path, miles))
            route_cache[od] = routes
        routes = route_cache[od]
        t0 = int(round(tr["pref"] / delta))
        short_slots = sum(slots[(routes[0][0][i], routes[0][0][i + 1])] for i in range(len(routes[0][0]) - 1))
        deadline = t0 + short_slots + slack
        its = []
        for path, miles in routes:
            edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            dur = sum(slots[e] for e in edges)
            for dt in range(-w, w + 1):
                dep = t0 + dt
                if dep + dur > deadline:
                    continue
                cost = cfg["kappa_per_mile"] * (miles - routes[0][1]) + cfg["gamma_per_hour"] * abs(dt) * delta / 60.0
                pairs, t = [], dep
                for e in edges:
                    pairs.append((e[0], e[1], t))
                    t += slots[e]
                its.append(Itinerary(f, cost, tuple(pairs)))
        ITIN.append(its)
    return ITIN, LEN


# ----------------------------------------------------------------------------------------
# 3. Coalition values: compact MIP (Appendix A)
# ----------------------------------------------------------------------------------------
class Benefit(object):
    """Per-truck saving alpha_l ($ per truck-mile) and the link benefit V(n) of eq. (link benefit)."""

    def __init__(self, cfg):
        fuel = float(cfg["fuel_cost_per_mile"])
        sav = {int(k): float(v) * fuel for k, v in cfg["saving_share"].items()}   # {2: .., 3: ..}
        self.m0 = max(sav)                    # savings are constant for platoons of m0 trucks or more
        self.alpha = [0.0, 0.0] + [sav[l] for l in range(2, self.m0 + 1)]
        for l in range(2, self.m0 + 1):
            assert self.alpha[l] >= self.alpha[l - 1] - EPS, "alpha must be nondecreasing"
        self.cap = int(cfg["platoon_cap"])
        self.plateau = self.cap >= 2 * self.m0 - 1      # then V(n) = n * alpha_n for every n
        self.beta = max(l * self.a(l) - (l - 1) * self.a(l - 1) for l in range(2, self.m0 + 1))
        self._V = [0.0, 0.0]

    def a(self, l):
        return self.alpha[min(l, self.m0)] if l >= 1 else 0.0

    def concave(self, nmax):
        """True if V(n) is concave for n = 1..nmax, i.e., the gain from one more truck never increases."""
        inc = [self.V(n) - self.V(n - 1) for n in range(2, nmax + 1)]
        return all(inc[i + 1] <= inc[i] + 1e-12 for i in range(len(inc) - 1))

    def V(self, n):
        """Benefit per mile of n trucks travelling a link together (optimal split into platoons)."""
        if self.plateau:
            return n * self.a(n)
        while len(self._V) <= n:
            m = len(self._V)
            self._V.append(max(l * self.a(l) + self._V[m - l] for l in range(1, min(self.cap, m) + 1)))
        return self._V[n]


def _reduce(its, LEN, ben):
    """Exact reduction rules (R1)-(R3). Returns the kept itineraries and reach[pair] = set of trucks."""
    active = list(range(len(its)))
    while True:
        reach = defaultdict(set)
        for j in active:
            for p in its[j].pairs:
                reach[p].add(its[j].truck)
        keep = []
        for j in active:
            shared_len = sum(LEN[(p[0], p[1])] for p in its[j].pairs if len(reach[p]) >= 2)
            if shared_len > 0 and its[j].cost < ben.beta * shared_len - 1e-12:
                keep.append(j)
        if len(keep) == len(active):
            break
        active = keep
    reach = {p: s for p, s in reach.items() if len(s) >= 2}
    return [its[j] for j in active], reach


def _components(its, reach):
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p, s in reach.items():
        s = list(s)
        for t in s[1:]:
            parent[find(t)] = find(s[0])
    comp = defaultdict(list)
    for it in its:
        comp[find(it.truck)].append(it)
    return list(comp.values())


def _solve_component(its, reach, LEN, ben, cfg):
    """Solves the compact MIP for one connected component.
    Returns (value, platooned trucks, platooned truck-miles, proven optimal).

    If V(n) is concave for n >= 1 (true for the savings used in the paper), the benefit of each
    link-time pair is modelled by a continuous variable bounded by the tangent lines of V, which gives a
    much tighter formulation than Appendix A.1 with the same optimal value. Otherwise the formulation of
    Appendix A.1 (one binary per number of credited trucks) is used."""
    pairs = sorted({p for it in its for p in it.pairs if p in reach})
    pidx = {p: k for k, p in enumerate(pairs)}
    plen = [LEN[(p[0], p[1])] for p in pairs]
    nbar = [len(reach[p]) for p in pairs]
    nx_ = len(its)
    concave = ben.concave(max(nbar))
    rows, cols, vals, rhs = [], [], [], []
    r = 0

    def add(entries, b):
        nonlocal r
        for cc, vv in entries:
            rows.append(r); cols.append(cc); vals.append(vv)
        rhs.append(b); r += 1

    by_truck = defaultdict(list)
    for j, it in enumerate(its):
        by_truck[it.truck].append(j)
    for t, js in by_truck.items():                       # one itinerary (or stand-alone) per truck
        add([(j, 1.0) for j in js], 1.0)
    cover = defaultdict(lambda: defaultdict(list))       # cover[pair][truck] -> itineraries
    for j, it in enumerate(its):
        for p in it.pairs:
            if p in pidx:
                cover[pidx[p]][it.truck].append(j)
    if concave:
        # columns: x (binary) | a_p in [0,1] | B_p >= 0
        ca = lambda k: nx_ + k
        cb = lambda k: nx_ + len(pairs) + k
        nvar = nx_ + 2 * len(pairs)
        c = np.zeros(nvar); integrality = np.zeros(nvar); ub = np.ones(nvar)
        integrality[:nx_] = 1
        for j, it in enumerate(its):
            c[j] = it.cost
        for k in range(len(pairs)):
            c[cb(k)] = -plen[k]
            ub[cb(k)] = ben.V(nbar[k])
            for t, js in cover[k].items():               # a_p = 1 as soon as one truck is present
                add([(j, 1.0) for j in js] + [(ca(k), -1.0)], 0.0)
            allx = [(j, 1.0) for js in cover[k].values() for j in js]
            seen = set()
            for m in range(1, nbar[k]):                  # tangent lines of V at m = 1, 2, ...
                slope = ben.V(m + 1) - ben.V(m)
                icpt = slope * m - ben.V(m)
                key = (round(slope, 9), round(icpt, 9))
                if key in seen:
                    continue
                seen.add(key)
                add([(cb(k), 1.0), (ca(k), icpt)] + [(j, -slope) for j, _ in allx], 0.0)
    else:
        # columns: x (binary) | lam_{p,m} (binary), m = 2..nbar_p
        lam, col = {}, nx_
        for k in range(len(pairs)):
            for m in range(2, nbar[k] + 1):
                lam[(k, m)] = col; col += 1
        nvar = col
        c = np.zeros(nvar); integrality = np.ones(nvar); ub = np.ones(nvar)
        for j, it in enumerate(its):
            c[j] = it.cost
        for (k, m), cc in lam.items():
            c[cc] = -plen[k] * ben.V(m)
        for k in range(len(pairs)):
            allx = [(j, -1.0) for js in cover[k].values() for j in js]
            add(allx + [(lam[(k, m)], float(m)) for m in range(2, nbar[k] + 1)], 0.0)   # credited <= present
            add([(lam[(k, m)], 1.0) for m in range(2, nbar[k] + 1)], 1.0)                # one size credited
    A = coo_matrix((vals, (rows, cols)), shape=(r, nvar)).tocsr()
    res = milp(c, constraints=LinearConstraint(A, -np.inf, np.array(rhs)), integrality=integrality,
               bounds=Bounds(np.zeros(nvar), ub),
               options={"time_limit": float(cfg["mip_time_limit_s"]), "mip_rel_gap": float(cfg["mip_rel_gap"])})
    if res.x is None:
        return 0.0, 0, 0.0, False            # stand-alone for everyone is always feasible (lower bound)
    chosen = [j for j in range(nx_) if res.x[j] > 0.5]
    count = defaultdict(int)
    for j in chosen:
        for p in its[j].pairs:
            if p in pidx:
                count[p] += 1
    value = sum(LEN[(p[0], p[1])] * ben.V(n) for p, n in count.items() if n >= 2) - sum(its[j].cost for j in chosen)
    platooned = sum(1 for j in chosen if any(count.get(p, 0) >= 2 for p in its[j].pairs))
    miles = sum(LEN[(p[0], p[1])] * n for p, n in count.items() if n >= 2)
    ok = (res.status == 0)
    return (value, platooned, miles, ok) if value > 0 else (0.0, 0, 0.0, ok)


def coalition_value(trucks, ITIN, LEN, ben, cfg):
    """v'(S) for the set of trucks of a coalition. Returns dict(value, trucks, miles, time, optimal)."""
    t0 = time.time()
    its = [it for f in trucks for it in ITIN[f]]
    its, reach = _reduce(its, LEN, ben)
    value, plat, miles, ok = 0.0, 0, 0.0, True
    for comp in _components(its, reach):
        comp_reach = {p: reach[p] for it in comp for p in it.pairs if p in reach}
        v, n, m, o = _solve_component(comp, comp_reach, LEN, ben, cfg)
        value += v; plat += n; miles += m; ok = ok and o
    return {"value": value, "trucks": plat, "miles": miles, "time": time.time() - t0, "optimal": ok}


_W = {}


def _init_worker(ITIN, LEN, cfg):
    _W["ITIN"], _W["LEN"], _W["cfg"], _W["ben"] = ITIN, LEN, cfg, Benefit(cfg)


def _work(args):
    mask, trucks = args
    return mask, coalition_value(trucks, _W["ITIN"], _W["LEN"], _W["ben"], _W["cfg"])


def coalition_table(fleets, ITIN, LEN, cfg, masks=None, cache=None):
    """fleets[i] = list of trucks of company i. Returns table[mask] = dict(value, trucks, miles, time, optimal)."""
    N = len(fleets)
    if cache and os.path.exists(cache):
        with open(cache) as fh:
            return close_table({int(k): v for k, v in json.load(fh).items()}, N)
    masks = list(range(1, 1 << N)) if masks is None else masks
    jobs = [(m, [f for i in range(N) if m >> i & 1 for f in fleets[i]]) for m in masks]
    n_jobs = int(cfg.get("n_jobs", 1))
    if n_jobs > 1:
        import multiprocessing as mp
        with mp.Pool(n_jobs, initializer=_init_worker, initargs=(ITIN, LEN, cfg)) as pool:
            out = dict(pool.imap_unordered(_work, jobs))
    else:
        _init_worker(ITIN, LEN, cfg)
        out = {}
        for n_done, j in enumerate(jobs, 1):
            mask, res = _work(j)
            out[mask] = res
            if n_done % 50 == 0:
                print("   ... %d of %d coalitions" % (n_done, len(jobs)), flush=True)
    out[0] = {"value": 0.0, "trucks": 0, "miles": 0.0, "time": 0.0, "optimal": True}
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "w") as fh:
            json.dump(out, fh)
    return close_table(out, N)


def close_table(table, N):
    """The MIPs are solved to a tolerance, so a computed v'(S) can be slightly below the value obtained by
    running the schedules of two disjoint sub-coalitions side by side. That combined schedule is feasible
    for S, so we use it whenever it is better. Afterwards v' is monotone and super-additive, and every value
    is still the value of a schedule that the platform can implement."""
    for m in range(1, 1 << N):
        if m not in table:
            continue
        low = m & -m
        for s in submasks(m & ~low):
            a, b = s | low, m ^ (s | low)
            if b and a in table and b in table and table[a]["value"] + table[b]["value"] > table[m]["value"] + 1e-9:
                table[m] = dict(table[m], value=table[a]["value"] + table[b]["value"],
                                trucks=table[a]["trucks"] + table[b]["trucks"],
                                miles=table[a]["miles"] + table[b]["miles"])
    return table


# ----------------------------------------------------------------------------------------
# 4. Game computations (companies are bits of a mask)
# ----------------------------------------------------------------------------------------
def members(mask):
    return [i for i in range(mask.bit_length()) if mask >> i & 1]


def submasks(mask):
    s = mask
    while True:
        yield s
        if s == 0:
            return
        s = (s - 1) & mask


def msum(vec, mask):
    return sum(vec[i] for i in members(mask))


def net_game(vp, u, N):
    """v(S) = max over C subset of S of v'(C) - u(C) (Lemma in Section 4), and the maximiser A*(S)."""
    full = (1 << N) - 1
    v, arg = [0.0] * (full + 1), [0] * (full + 1)
    for m in range(1, full + 1):
        best, a = vp[m] - msum(u, m), m
        if best < 0:
            best, a = 0.0, 0
        for i in members(m):
            sub = m & ~(1 << i)
            if v[sub] > best + EPS:
                best, a = v[sub], arg[sub]
        v[m], arg[m] = best, a
    return v, arg


def shapley(w, T):
    """Shapley value of the game w restricted to the player set T (mask). Returns dict i -> value."""
    pl = members(T)
    n = len(pl)
    out = {}
    for i in pl:
        rest = T & ~(1 << i)
        tot = 0.0
        for R in submasks(rest):
            r = bin(R).count("1")
            tot += math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n) * (w[R | (1 << i)] - w[R])
        out[i] = tot
    return out


def envelope_shares(vp, N):
    """Shapley-envelope shares: env[S][i] = min over T containing S of the Shapley value of i in (T, v')."""
    full = (1 << N) - 1
    phi = {T: shapley(vp, T) for T in range(1, full + 1)}
    env = {S: {i: float("inf") for i in members(S)} for S in range(1, full + 1)}
    for T in range(1, full + 1):
        for S in submasks(T):
            if S:
                for i in members(S):
                    if phi[T][i] < env[S][i]:
                        env[S][i] = phi[T][i]
    return env, phi


def moulin(vp, bids, N, env):
    S = (1 << N) - 1
    while S:
        R = [i for i in members(S) if bids[i] > env[S][i] + EPS]
        if not R:
            break
        for i in R:
            S &= ~(1 << i)
    x = [env[S][i] if S >> i & 1 else 0.0 for i in range(N)]
    return S, x


def moulin_bound(vp, u, N, A):
    """Right-hand side of the efficiency-loss bound (Theorem 4)."""
    full = (1 << N) - 1
    tot = msum(u, A)
    for i in range(N):
        if not A >> i & 1:
            free = full & ~A & ~(1 << i)
            tot += max(vp[A | X | (1 << i)] - vp[A | X] for X in submasks(free))
    return tot


def greedy(vp, bids, fleet, N, k):
    cand = [m for m in range(1, 1 << N) if bin(m).count("1") <= k]
    rho = {m: vp[m] / msum(fleet, m) for m in cand}
    order = sorted(cand, key=lambda m: (-rho[m], m))         # ties: fixed order, independent of bids
    dead, formed, x = 0, [], [0.0] * N
    while True:
        sel = next((m for m in order if m & dead == 0), None)
        if sel is None:
            break
        R = [i for i in members(sel) if bids[i] > fleet[i] * rho[sel] + EPS]
        if R:
            for i in R:
                dead |= 1 << i
        else:
            formed.append(sel)
            for i in members(sel):
                x[i] = fleet[i] * rho[sel]
            dead |= sel
    A = 0
    for m in formed:
        A |= m
    return formed, A, x


def best_k_partition(vp, A, k):
    """Largest gross benefit of partitions of A into coalitions of at most k companies."""
    best = {0: 0.0}
    for m in sorted(submasks(A)):
        if m == 0:
            continue
        low = m & -m
        val = 0.0
        for s in submasks(m & ~low):
            c = s | low
            if bin(c).count("1") <= k:
                val = max(val, vp[c] + best[m ^ c])
        best[m] = val
    return best[A]


def max_excess(v, pay, N):
    """Largest amount a coalition could gain by leaving: max_S v(S) - pay(S)."""
    return max(v[m] - msum(pay, m) for m in range(1, 1 << N))


def least_core(v, N):
    """Smallest e such that some allocation x with x(N)=v(N) has v(S)-x(S) <= e for all S. Core nonempty iff e <= 0."""
    full = (1 << N) - 1
    if N == 1:
        return 0.0
    A, b = [], []
    for m in range(1, full):
        A.append([-1.0 if m >> i & 1 else 0.0 for i in range(N)] + [-1.0])
        b.append(-v[m])
    res = linprog([0.0] * N + [1.0], A_ub=A, b_ub=b, A_eq=[[1.0] * N + [0.0]], b_eq=[v[full]],
                  bounds=[(None, None)] * (N + 1), method="highs")
    return res.fun


# ----------------------------------------------------------------------------------------
# 5. One instance: all methods and metrics
# ----------------------------------------------------------------------------------------
def evaluate(table, u, fleet, N, ks):
    """Returns a list of result rows (dicts), one per method."""
    full = (1 << N) - 1
    vp = [table[m]["value"] for m in range(full + 1)]
    v, arg = net_game(vp, u, N)
    opt = v[full]
    lc = least_core(v, N)
    rows = []

    def row(method, net, served, trucks, pay, **extra):
        r = {"method": method, "net_benefit": net, "pct_of_optimum": 100.0 * net / opt if opt > EPS else float("nan"),
             "companies_served": bin(served).count("1"), "trucks_platooned": trucks,
             "max_excess": max_excess(v, pay, N), "optimum": opt, "least_core": lc, "core_nonempty": lc <= 1e-6}
        r["max_excess_pct"] = 100.0 * r["max_excess"] / opt if opt > EPS else float("nan")
        r.update(extra)
        rows.append(r)

    # (i) system optimum with Shapley allocation of v
    A = arg[full]
    sh = shapley(v, full)
    row("optimum_shapley", opt, A, table[A]["trucks"], [sh[i] for i in range(N)])

    # (ii) Shapley-envelope Moulin mechanism (truthful bids)
    t0 = time.time()
    env, _ = envelope_shares(vp, N)
    S, x = moulin(vp, u, N, env)
    t_m = time.time() - t0
    net = vp[S] - msum(u, S)
    surplus = vp[S] - sum(x)
    pay = [x[i] - u[i] if S >> i & 1 else 0.0 for i in range(N)]
    row("moulin", net, S, table[S]["trucks"], pay, surplus=surplus,
        surplus_pct=100.0 * surplus / vp[S] if vp[S] > EPS else 0.0,
        loss=opt - net, loss_bound=moulin_bound(vp, u, N, S), mechanism_time=t_m)

    # (iii) Greedy mechanism (truthful bids)
    for k in ks:
        k = min(k, N)
        t0 = time.time()
        formed, A, x = greedy(vp, u, fleet, N, k)
        t_g = time.time() - t0
        gross = sum(vp[m] for m in formed)
        net = gross - msum(u, A)
        pay = [x[i] - u[i] if A >> i & 1 else 0.0 for i in range(N)]
        fl = [fleet[i] for i in range(N)]
        kappa = k * max(fl) / float(min(fl))
        vk = best_k_partition(vp, A, k)
        row("greedy_k%d" % k, net, A, sum(table[m]["trucks"] for m in formed), pay, surplus=0.0, surplus_pct=0.0,
            loss=opt - net, gross=gross, best_k_partition=vk, kappa=kappa,
            guaranteed_gross=vk / kappa, coalitions_formed=len(formed), mechanism_time=t_g)
    return rows


# ----------------------------------------------------------------------------------------
# 6. Experiment runner
# ----------------------------------------------------------------------------------------
def run(cfg, scenarios=None, n_inst=None):
    rng = random.Random(cfg["seed"])
    G = load_network(cfg["network_file"])
    trips = load_trips(cfg["trips_file"], cfg, rng)
    t0 = time.time()
    ITIN, LEN = build_itineraries(G, trips, cfg)
    print("trucks: %d | itineraries: %d | built in %.1fs" % (len(trips), sum(len(x) for x in ITIN), time.time() - t0))
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    all_rows = []
    for sc_name, sizes in cfg["scenarios"].items():
        if scenarios and str(sc_name) not in scenarios:
            continue
        assert sum(sizes) == len(trips), "scenario %s: fleet sizes must add up to the number of trucks" % sc_name
        N = len(sizes)
        for inst in range(n_inst or cfg["instances_per_scenario"]):
            irng = random.Random("%s-%s-%d" % (cfg["seed"], sc_name, inst))
            perm = list(range(len(trips)))
            irng.shuffle(perm)
            fleets, pos = [], 0
            for s in sizes:
                fleets.append(perm[pos:pos + s]); pos += s
            t0 = time.time()
            cache = os.path.join(out_dir, "tables", "scenario%s_inst%d.json" % (sc_name, inst))
            table = coalition_table(fleets, ITIN, LEN, cfg, cache=cache)
            t_table = sum(table[m]["time"] for m in table)
            not_opt = sum(1 for m in table if not table[m]["optimal"])
            base = [irng.uniform(0.0, cfg["entry_cost_max_per_truck"]) for _ in range(N)]
            for level in cfg["entry_cost_levels"]:
                u = [base[i] * sizes[i] * level for i in range(N)]
                for r in evaluate(table, u, sizes, N, cfg["greedy_k"]):
                    r.update({"scenario": sc_name, "instance": inst, "n_companies": N, "entry_cost_level": level,
                              "table_time": t_table, "coalitions": len(table) - 1, "not_proven_optimal": not_opt})
                    all_rows.append(r)
            print("scenario %s, instance %d: %d coalitions, MIP time %.1fs (wall %.1fs), not proven optimal: %d"
                  % (sc_name, inst, len(table) - 1, t_table, time.time() - t0, not_opt))
    keys = ["scenario", "instance", "n_companies", "entry_cost_level", "method", "optimum", "net_benefit",
            "pct_of_optimum", "companies_served", "trucks_platooned", "surplus", "surplus_pct", "max_excess",
            "max_excess_pct", "least_core", "core_nonempty", "loss", "loss_bound", "gross", "best_k_partition",
            "kappa", "guaranteed_gross", "coalitions_formed", "mechanism_time", "table_time", "coalitions",
            "not_proven_optimal"]
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(all_rows)
    summarize(all_rows, os.path.join(out_dir, "summary.csv"))
    return all_rows


def summarize(rows, path):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["scenario"], r["entry_cost_level"], r["method"])].append(r)
    cols = ["pct_of_optimum", "companies_served", "trucks_platooned", "surplus_pct", "max_excess_pct", "loss",
            "loss_bound", "gross", "guaranteed_gross", "mechanism_time", "table_time"]
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["scenario", "entry_cost_level", "method", "instances", "core_nonempty_share"] + ["mean_" + c for c in cols])
        for key in sorted(groups, key=str):
            g = groups[key]
            means = []
            for c in cols:
                vals = [r[c] for r in g if c in r and r[c] is not None and not (isinstance(r[c], float) and math.isnan(r[c]))]
                means.append("%.4f" % (sum(vals) / len(vals)) if vals else "")
            wr.writerow(list(key) + [len(g), "%.2f" % (sum(1 for r in g if r["core_nonempty"]) / float(len(g)))] + means)
    print("wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--scenarios", nargs="*", help="run only these scenarios (names as in the config)")
    ap.add_argument("--instances", type=int, help="override instances_per_scenario")
    args = ap.parse_args()
    with open(args.config) as fh:
        config = yaml.safe_load(fh)
    run(config, args.scenarios, args.instances)
