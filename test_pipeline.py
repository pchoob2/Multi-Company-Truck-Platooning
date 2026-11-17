#!/usr/bin/env python3
"""Checks of the pipeline. Run:  python test_pipeline.py"""
import itertools
import random
from collections import defaultdict

import networkx as nx

import platooning_experiments as pe

CFG = {"speed_mph": 60, "delta_min": 10, "departure_window_min": 20, "arrival_slack_min": 20,
       "k_routes": 2, "rho": 0.3, "fuel_cost_per_mile": 100.0, "saving_share": {2: 0.0725, 3: 0.0783},
       "platoon_cap": 5, "kappa_per_mile": 0.02, "gamma_per_hour": 3.0, "mip_time_limit_s": 60,
       "mip_rel_gap": 0.0, "n_jobs": 1}


def brute_force(trucks, ITIN, LEN, ben):
    """Enumerates every choice of itinerary (or stand-alone) for every truck."""
    best = 0.0
    options = [[None] + ITIN[f] for f in trucks]
    for combo in itertools.product(*options):
        count = defaultdict(int)
        cost = 0.0
        for it in combo:
            if it is not None:
                cost += it.cost
                for p in it.pairs:
                    count[p] += 1
        val = sum(LEN[(p[0], p[1])] * ben.V(n) for p, n in count.items() if n >= 2) - cost
        best = max(best, val)
    return best


def test_mip_against_brute_force(n_cases=60):
    rng = random.Random(1)
    worst = 0.0
    for case in range(n_cases):
        G = nx.DiGraph()
        nodes = list("ABCDEF")
        edges = [("A", "B"), ("B", "C"), ("C", "D"), ("B", "E"), ("E", "D"), ("D", "F"), ("A", "E")]
        for u, v in edges:
            m = rng.choice([10, 20, 30])
            G.add_edge(u, v, miles=m)
            G.add_edge(v, u, miles=m)
        cfg = dict(CFG)
        cfg["platoon_cap"] = rng.choice([2, 3, 5])                       # also exercises the capped case
        cfg["saving_share"] = rng.choice([{2: 0.0725, 3: 0.0783}, {2: 0.05}, {2: 0.04, 3: 0.06, 4: 0.07}])
        trips = []
        for t in range(rng.randint(3, 5)):
            o, d = rng.sample(nodes, 2)
            trips.append({"id": "t%d" % t, "o": o, "d": d, "pref": rng.choice([0, 10, 20, 30])})
        ITIN, LEN = pe.build_itineraries(G, trips, cfg)
        ben = pe.Benefit(cfg)
        trucks = list(range(len(trips)))
        got = pe.coalition_value(trucks, ITIN, LEN, ben, cfg)["value"]
        ref = brute_force(trucks, ITIN, LEN, ben)
        worst = max(worst, abs(got - ref))
        assert abs(got - ref) < 1e-6, (case, got, ref)
    print("MIP vs brute force: %d cases, largest difference %.2e" % (n_cases, worst))


def table_from(values, N):
    vp = [0.0] * (1 << N)
    for members, val in values.items():
        vp[sum(1 << i for i in members)] = val
    return vp


def test_paper_examples():
    # Example (Moulin): three one-truck companies, any pair generates 14.5, all three generate 14.5
    N = 3
    vp = table_from({(0, 1): 14.5, (0, 2): 14.5, (1, 2): 14.5, (0, 1, 2): 14.5}, N)
    env, phi = pe.envelope_shares(vp, N)
    assert abs(phi[0b011][0] - 7.25) < 1e-9 and abs(phi[0b111][0] - 14.5 / 3) < 1e-9
    assert abs(env[0b011][0] - 14.5 / 3) < 1e-9 and env[0b001][0] == 0.0
    S, x = pe.moulin(vp, [2, 3, 6], N, env)
    assert S == 0b011 and abs(sum(x) - 2 * 14.5 / 3) < 1e-9          # companies 1,2 served, surplus 4.83
    S, x = pe.moulin(vp, [5, 3, 6], N, env)
    assert S == 0                                                      # nobody served
    v, arg = pe.net_game(vp, [5, 3, 6], N)
    assert abs(v[0b111] - 6.5) < 1e-9                                  # efficient net benefit 6.5
    S, x = pe.moulin(vp, [4, 6, 0], N, env)                            # company 1 under-bids (true cost 6)
    assert S == 0b101 and x[0] - 6 < 0                                 # served, but at a loss
    # Example (Greedy): four one-truck companies
    N = 4
    vals = {(0, 1): 130.5, (2, 3): 130.5, (1, 2): 145.0, (0, 1, 2): 145.0, (1, 2, 3): 145.0,
            (0, 1, 3): 130.5, (0, 2, 3): 130.5, (0, 1, 2, 3): 261.0}
    vp = table_from(vals, N)
    formed, A, x = pe.greedy(vp, [0, 0, 0, 0], [1, 1, 1, 1], N, 4)
    assert 0b0110 in formed and abs(sum(vp[m] for m in formed) - 145.0) < 1e-9 and abs(x[1] - 72.5) < 1e-9
    formed, A, x = pe.greedy(vp, [0, 80, 0, 0], [1, 1, 1, 1], N, 4)
    assert 0b1100 in formed and abs(x[2] - 65.25) < 1e-9 and not A >> 1 & 1
    assert abs(pe.best_k_partition(vp, 0b1111, 4) - 261.0) < 1e-9
    print("paper examples: ok")


def utility(method, vp, bids, u, fleet, N, i, env=None, k=None):
    if method == "moulin":
        S, x = pe.moulin(vp, bids, N, env)
        return x[i] - u[i] if S >> i & 1 else 0.0
    formed, A, x = pe.greedy(vp, bids, fleet, N, k)
    return x[i] - u[i] if A >> i & 1 else 0.0


def test_properties_on_real_tables(n_tables=6):
    """Tables of v' from small platooning instances: truthfulness, bounds, budget, internal stability."""
    rng = random.Random(5)
    G = pe.load_network("sample_network.csv")
    cities = ["Chicago", "Detroit", "Columbus", "Indianapolis", "StLouis", "Madison", "Minneapolis"]
    cfg = dict(CFG)
    cfg.update({"fuel_cost_per_mile": 0.71, "kappa_per_mile": 2.27, "gamma_per_hour": 10.0, "rho": 0.05,
                "departure_window_min": 30, "arrival_slack_min": 30})
    checks = 0
    for t in range(n_tables):
        N = rng.choice([3, 4, 5])
        sizes = [rng.choice([1, 2, 4, 8]) for _ in range(N)]
        trips = []
        for f in range(sum(sizes)):
            o, d = rng.sample(cities, 2)
            trips.append({"id": str(f), "o": o, "d": d, "pref": rng.uniform(360, 600)})
        ITIN, LEN = pe.build_itineraries(G, trips, cfg)
        fleets, pos = [], 0
        for s in sizes:
            fleets.append(list(range(pos, pos + s))); pos += s
        table = pe.coalition_table(fleets, ITIN, LEN, cfg)
        full = (1 << N) - 1
        vp = [table[m]["value"] for m in range(full + 1)]
        for m in range(1, full + 1):                                   # v' is monotone and super-additive
            for s in pe.submasks(m):
                assert vp[s] <= vp[m] + 1e-6 and vp[s] + vp[m ^ s] <= vp[m] + 1e-6
        env, _ = pe.envelope_shares(vp, N)
        for S in range(1, full + 1):                                   # envelope: no deficit, cross-monotonic
            assert sum(env[S].values()) <= vp[S] + 1e-6
            for T in range(1, full + 1):
                if S & T == S:
                    assert all(env[S][i] <= env[T][i] + 1e-9 for i in pe.members(S))
        scale = max(vp[full] / max(sum(sizes), 1), 1e-3)
        for _ in range(40):
            u = [rng.uniform(0, 2.5) * scale * sizes[i] for i in range(N)]
            v, arg = pe.net_game(vp, u, N)
            # Moulin: efficiency-loss bound, no deficit
            S, x = pe.moulin(vp, u, N, env)
            assert v[full] - (vp[S] - pe.msum(u, S)) <= pe.moulin_bound(vp, u, N, S) + 1e-6
            assert sum(x) <= vp[S] + 1e-6
            for k in (2, N):
                formed, A, x = pe.greedy(vp, u, sizes, N, k)
                gross = sum(vp[m] for m in formed)
                assert abs(sum(x) - gross) < 1e-6                      # budget balance
                kappa = k * max(sizes) / float(min(sizes))
                assert gross >= pe.best_k_partition(vp, A, k) / kappa - 1e-6
                for T in formed:                                       # internal stability
                    for B in pe.submasks(T):
                        if B:
                            assert sum(x[i] - u[i] for i in pe.members(B)) >= v[B] - 1e-6
            # strategyproofness: no single company gains by misreporting
            grid = sorted(set([0.0] + [b * scale * max(sizes) for b in (0.1, 0.3, 0.6, 1, 1.5, 2, 3, 5)]
                              + [e for S2 in env for e in env[S2].values()]))
            for i in range(N):
                for method, kw in (("moulin", {"env": env}), ("greedy", {"k": 2}), ("greedy", {"k": N})):
                    truthful = utility(method, vp, u, u, sizes, N, i, **kw)
                    for b in grid:
                        for bb in (b - 1e-6, b + 1e-6):
                            bids = list(u); bids[i] = max(bb, 0.0)
                            assert utility(method, vp, bids, u, sizes, N, i, **kw) <= truthful + 1e-6
                            checks += 1
            # group strategyproofness of Moulin for pairs
            for i, j in itertools.combinations(range(N), 2):
                ti = utility("moulin", vp, u, u, sizes, N, i, env=env)
                tj = utility("moulin", vp, u, u, sizes, N, j, env=env)
                for bi in grid[::3]:
                    for bj in grid[::3]:
                        bids = list(u); bids[i] = bi; bids[j] = bj
                        gi = utility("moulin", vp, bids, u, sizes, N, i, env=env)
                        gj = utility("moulin", vp, bids, u, sizes, N, j, env=env)
                        assert not (gi >= ti - 1e-9 and gj >= tj - 1e-9 and (gi > ti + 1e-6 or gj > tj + 1e-6))
                        checks += 1
    print("properties on %d platooning tables: ok (%d misreport checks)" % (n_tables, checks))


if __name__ == "__main__":
    test_paper_examples()
    test_mip_against_brute_force()
    test_properties_on_real_tables()
    print("all tests passed")
