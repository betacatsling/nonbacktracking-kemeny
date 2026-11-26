"""Exact checks for vertex hitting times and projected Kemeny constants.

Run with Python 3. No external packages are required.
"""

from fractions import Fraction as Q
from itertools import combinations
from math import prod


def solve(matrix, rhs):
    """Solve a nonsingular system by Gauss-Jordan elimination over the rationals."""
    rows = [[Q(x) for x in row] + [Q(y)] for row, y in zip(matrix, rhs)]
    for col in range(len(rows)):
        pivot = next(i for i in range(col, len(rows)) if rows[i][col])
        rows[col], rows[pivot] = rows[pivot], rows[col]
        divisor = rows[col][col]
        rows[col] = [x / divisor for x in rows[col]]
        for i, row in enumerate(rows):
            if i != col and row[col]:
                factor = row[col]
                rows[i] = [a - factor * b for a, b in zip(row, rows[col])]
    return [row[-1] for row in rows]


def graph(edges):
    neighbors = {}
    for u, v in edges:
        neighbors.setdefault(u, []).append(v)
        neighbors.setdefault(v, []).append(u)
    return neighbors


def arcs_and_successors(neighbors):
    arcs = [(u, v) for u in neighbors for v in neighbors[u]]
    return {arc: [(arc[1], w) for w in neighbors[arc[1]] if w != arc[0]]
            for arc in arcs}


def mean_times(successors, absorbing):
    """Solve h=0 on the target and h=1+mean(h at successors) elsewhere."""
    states = list(successors)
    index = {state: i for i, state in enumerate(states)}
    matrix = []
    rhs = []
    for i, state in enumerate(states):
        row = [Q(0)] * len(states)
        row[i] = 1
        if state in absorbing:
            rhs.append(0)
        else:
            options = successors[state]
            for following in options:
                row[index[following]] -= Q(1, len(options))
            rhs.append(1)
        matrix.append(row)
    return dict(zip(states, solve(matrix, rhs)))


def hitting_times(neighbors, successors, target):
    simple = mean_times(neighbors, {target})
    edge = mean_times(successors, {arc for arc in successors if arc[1] == target})
    # The first step has no previous vertex. At the target, first hitting is zero.
    nb = {x: 1 + sum(edge[x, y] for y in options) / len(options)
          for x, options in neighbors.items()}
    nb[target] = Q(0)
    return simple, nb


def projected_kemeny(neighbors, successors):
    """Compute tr(D^{-1} T (I-P+W)^{-1} T^t) - 1; T records arc tails."""
    arcs = list(successors)
    size = len(arcs)
    index = {arc: i for i, arc in enumerate(arcs)}
    matrix = [[Q(i == j) + Q(1, size) for j in range(size)] for i in range(size)]
    for i, arc in enumerate(arcs):
        for following in successors[arc]:
            matrix[i][index[following]] -= Q(1, len(successors[arc]))
    trace = Q(0)
    for v, neighbors_v in neighbors.items():
        col = solve(matrix, [arc[0] == v for arc in arcs])
        trace += sum(col[index[v, w]] for w in neighbors_v) / len(neighbors_v)
    return trace - 1


def stationary_mean(values, stationary):
    return sum(stationary[v] * values[v] for v in stationary)


def tree_criterion(neighbors):
    """Enumerate all spanning trees and compare each exact edge inclusion probability."""
    edges = [(u, v) for u in neighbors for v in neighbors[u] if u < v]
    q = {v: Q(1, len(options)) for v, options in neighbors.items()}
    conductance = {}
    for u, v in edges:
        resistance = 1 - q[u] - q[v]
        if resistance == 0:
            return False, f"edge {u}-{v} joins two vertices of degree two"
        conductance[u, v] = 1 / resistance

    total = Q(0)
    included = dict.fromkeys(edges, Q(0))
    for subset in combinations(edges, len(neighbors) - 1):
        selected = graph(subset)
        start = next(iter(neighbors))
        reached, stack = {start}, [start]
        while stack:
            for v in selected.get(stack.pop(), ()):
                if v not in reached:
                    reached.add(v)
                    stack.append(v)
        if len(reached) != len(neighbors):
            continue
        weight = prod(conductance[edge] for edge in subset)
        total += weight
        for edge in subset:
            included[edge] += weight
    for u, v in edges:
        observed = included[u, v] / total
        required = q[u] + q[v] - Q(1, len(edges))
        if observed != required:
            return False, f"edge {u}-{v} has inclusion probability {observed}, required {required}"
    return True, f"every edge has its required inclusion probability ({len(edges)} checked)"


def check_local_uniformity(name, neighbors, expected):
    # This second test uses only the walk. The count starts with the specified first step.
    successors = arcs_and_successors(neighbors)
    walk_uniform = True
    witness = ""
    for v, options in neighbors.items():
        times = mean_times(successors, {arc for arc in successors if arc[1] == v})
        returns = [1 + times[v, w] for w in options]
        if len(set(returns)) > 1:
            walk_uniform = False
            witness = f"; conditional means at {v}: " + ", ".join(map(str, returns))
            break
    tree_uniform, detail = tree_criterion(neighbors)
    assert walk_uniform == tree_uniform == expected, (name, walk_uniform, tree_uniform)
    print(f"{name}: local uniformity={walk_uniform}; {detail}{witness}")


def check_graph(name, neighbors, expected_trace, expected_hit):
    successors = arcs_and_successors(neighbors)
    pi = {v: Q(len(options), len(successors)) for v, options in neighbors.items()}
    target_nb, target_simple = {}, {}
    for target in sorted(neighbors):
        simple, nb = hitting_times(neighbors, successors, target)
        target_nb[target] = stationary_mean(nb, pi)
        target_simple[target] = stationary_mean(simple, pi)
        assert target_nb[target] < target_simple[target], (name, target)
    k_hit = stationary_mean(target_nb, pi)
    k_simple = stationary_mean(target_simple, pi)
    k_trace = projected_kemeny(neighbors, successors)
    assert k_trace == expected_trace, (name, k_trace)
    assert k_hit == expected_hit, (name, k_hit)
    print(f"{name}: K_trace={k_trace}, K_hit={k_hit}, "
          f"gap={k_hit - k_trace}, K_simple={k_simple}")
    print("  Stationary-start means (target, NB, simple):", ", ".join(
        f"({v}, {target_nb[v]}, {target_simple[v]})" for v in sorted(neighbors)))


def two_cycles(left, right):
    cycles = [list(range(left)), [0] + list(range(left, left + right - 1))]
    return graph((u, v) for c in cycles for u, v in zip(c, c[1:] + c[:1]))


def check_pointwise(left, right):
    # The cycles meet at w=0. On the first cycle, 0 and x=2 both neighbor v=1.
    neighbors = two_cycles(left, right)
    successors = arcs_and_successors(neighbors)
    simple, nb = hitting_times(neighbors, successors, target=1)
    assert simple[2] == left - 1 + Q(2 * right, left)
    assert nb[2] == Q(4 * left + 3 * right - 2, 6)
    pi = {v: Q(len(options), len(successors)) for v, options in neighbors.items()}
    mean_nb = stationary_mean(nb, pi)
    mean_simple = stationary_mean(simple, pi)
    assert mean_nb < mean_simple
    print(f"Two cycles ({left}, {right}), x=2 to v=1: NB={nb[2]}, simple={simple[2]}")
    print(f"  Stationary start to the same target: NB={mean_nb}, simple={mean_simple}")
    return nb[2], simple[2]


if __name__ == "__main__":
    complete = list(combinations(range(4), 2))
    diamond = [e for e in complete if e != (2, 3)]
    check_graph("Diamond", graph(diamond), expected_trace=Q(17, 10), expected_hit=Q(128, 75))
    check_graph("K4", graph(complete), expected_trace=Q(7, 4), expected_hit=Q(7, 4))
    bipartite = graph((u, v) for u in (0, 1) for v in (2, 3, 4))
    check_graph("K2,3", bipartite, expected_trace=Q(7, 3), expected_hit=Q(7, 3))
    check_local_uniformity("Diamond", graph(diamond), expected=False)
    check_local_uniformity("K4", graph(complete), expected=True)
    check_local_uniformity("K2,3", bipartite, expected=True)
    check_local_uniformity("Two triangles sharing a vertex", two_cycles(3, 3), expected=False)
    # k is the length of each cycle. Every pinwheel here has exactly two cycles.
    for k in range(3, 8):
        expected_trace = Q(5 * (k - 1), 4)
        expected_gap = Q((k - 1) * (k - 2), 36 * k)
        check_graph(f"Two {k}-cycles", two_cycles(k, k), expected_trace,
                    expected_hit=expected_trace + expected_gap)
    assert check_pointwise(9, 9) == (Q(61, 6), Q(10))
    for lengths in [(4, 7), (6, 11), (7, 13), (11, 5)]:
        check_pointwise(*lengths)
