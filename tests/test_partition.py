"""The partitions: the graph algorithm, and the subregions of small networks.

The graph tests reproduce, one by one, cases the reviews of the method found:
each fails on the code before its fix.
"""

import importlib

import numpy as np
import pytest

import flowtopo
from flowtopo.core import D8_NODATA

partition = importlib.import_module("flowtopo.partition")

TRANSFORM = (0.0, 1 / 1200, 0.0, 40.0, 0.0, -1 / 1200)


def build(d8):
    return flowtopo.FlowTopo.from_d8(np.asarray(d8, dtype=np.uint8), transform=TRANSFORM)


def rivers(nrow, widths):
    """Side by side basins; in each, rows drain sideways into a mainstem column
    that drains south to a pit on the last row."""
    ncol = int(sum(widths))
    cells = np.arange(nrow * ncol, dtype=np.int32).reshape(nrow, ncol)
    idxs_ds = cells.copy()
    first = 0
    for width in widths:
        center = first + width // 2
        idxs_ds[:, first:center] = cells[:, first:center] + 1
        idxs_ds[:, center + 1 : first + width] = cells[:, center + 1 : first + width] - 1
        idxs_ds[:-1, center] = cells[:-1, center] + ncol
        first += width
    return flowtopo.FlowTopo(idxs_ds.ravel(), (nrow, ncol))


def check(topo, part, load):
    """Every cell has a subregion or the second stage; the loads count the
    rest; flow leaves a subregion only into the second stage, which drains
    only on into itself; and the stages accumulate what one pass does."""
    mask = topo.mask
    stem = part == flowtopo.MAINSTEM
    assert np.all((part[mask] >= 0) | stem[mask])
    assert np.all(part[~mask] == -1)
    reaching = mask & (topo.basins > 0)
    assert load.sum() + np.count_nonzero(stem) == np.count_nonzero(reaching)
    cells = np.flatnonzero(reaching & ~stem)
    downstream = topo.idxs_ds[cells]
    crossing = part[downstream] != part[cells]
    assert np.all(stem[downstream[crossing]])
    stem_cells = np.flatnonzero(stem)
    assert np.all(stem[topo.idxs_ds[stem_cells]])
    order = topo.ordering("dfs", "u2d")
    serial = mask.astype(np.int64)
    staged = serial.copy()
    for cell in order:
        down = topo.idxs_ds[cell]
        if down != cell:
            serial[down] += serial[cell]
    n_parts = load.size
    for stage in list(range(n_parts)) + [flowtopo.MAINSTEM]:
        for cell in order[part[order] == stage]:
            down = topo.idxs_ds[cell]
            if down != cell:
                staged[down] += staged[cell]
    assert np.array_equal(staged, serial)


def pieces(topo, part, value):
    """Pieces of one subregion on the ground, the second stage joined in (the
    two banks of a held-back mainstem touch across it)."""
    from scipy import ndimage

    grid = part.reshape(topo.shape)
    joined = (grid == value) | (grid == flowtopo.MAINSTEM)
    labels, _ = ndimage.label(joined, structure=np.ones((3, 3), dtype=int))
    return np.unique(labels[grid == value]).size


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------


def test_graph_counts_cells_and_shared_sides():
    nodes = np.tile(np.arange(8, dtype=np.int64), (8, 1))
    graph = partition.graph_from_node_raster(nodes, 8)
    assert np.all(graph.weights == 8)
    assert np.allclose(graph.rows, 3.5)
    assert np.allclose(graph.cols, np.arange(8))
    assert np.array_equal(graph.edges, np.column_stack((np.arange(7), np.arange(1, 8))))
    assert np.all(graph.edge_weights == 8)


def line_graph(weights):
    weights = np.asarray(weights, dtype=np.int64)
    n = weights.size
    edges = np.column_stack((np.arange(n - 1), np.arange(1, n)))
    return partition.PartitionGraph(
        weights,
        np.zeros(n),
        np.arange(n, dtype=np.float64),
        edges,
        np.ones(n - 1, dtype=np.int64),
    )


def test_merge_edges_sums_and_drops_loops():
    edges, weights = partition.merge_edges(
        [np.array([[1, 0], [0, 1], [2, 2]]), np.array([[0, 1]])],
        3,
        [np.array([2, 3, 9]), np.array([4])],
    )
    assert np.array_equal(edges, [[0, 1]])
    assert np.array_equal(weights, [9])


def test_parts_go_where_the_heaviest_part_is_lightest():
    # 2.9 and 1.1 shares: 3 + 1 parts (1.1 the heaviest), not 2 + 2 (1.45)
    assert np.array_equal(partition._apportion(np.array([2.9, 1.1]), np.array([9, 9]), 4), [3, 1])
    assert np.array_equal(partition._apportion(np.array([2.0, 1.0, 1.0]), np.array([9, 9, 9]), 4), [2, 1, 1])
    # no more parts than a mass has nodes to place
    assert np.array_equal(partition._apportion(np.array([3.5, 0.5]), np.array([2, 9]), 4), [2, 2])


def test_islands_move_whole_to_balance_and_land_stays_one_piece():
    # land A (12 nodes) and land B (8 nodes) far apart, four one-node islands
    # between them; two parts of 12: two islands nearer A go to B's part
    weights = np.ones(24, dtype=np.int64)
    edges = np.array([[i, i + 1] for i in range(11)] + [[i, i + 1] for i in range(12, 19)])
    cols = np.r_[np.arange(12.0), 100 + np.arange(8.0), 40, 50, 60, 70]
    graph = partition.PartitionGraph(
        weights, np.zeros(24), cols, edges, np.ones(edges.shape[0], dtype=np.int64)
    )
    parts = partition._partition_components(graph, graph, 2, 42, 1.0, True)
    assert np.array_equal(np.bincount(parts, minlength=2), [12, 12])
    assert np.unique(parts[:12]).size == 1 and np.unique(parts[12:20]).size == 1
    assert parts[0] != parts[12]


def peninsula_graph():
    """part 1 = {p}; part 0 = {c, a, b, x, y, z} with c holding the peninsula
    a-b and touching the body x-y-z and p."""
    names = ["p", "c", "a", "b", "x", "y", "z"]
    weights = np.array([1, 1, 1, 1, 2, 2, 2], dtype=np.int64)
    pairs = [("p", "c"), ("c", "a"), ("a", "b"), ("c", "x"), ("x", "y"), ("y", "z")]
    edges = np.array([sorted((names.index(a), names.index(b))) for a, b in pairs])
    graph = partition.PartitionGraph(
        weights,
        np.zeros(7),
        np.arange(7, dtype=np.float64),
        edges,
        np.ones(len(pairs), dtype=np.int64),
    )
    parts = np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.int32)
    return names, graph, parts


def test_detach_takes_the_peninsula_and_leaves_the_body():
    names, graph, parts = peninsula_graph()
    xadj, adjncy, _ = partition._csr(graph.size, graph.edges, graph.edge_weights)
    out = np.empty(graph.size, dtype=np.int64)
    taken, _ = partition._detach(
        xadj,
        adjncy,
        graph.weights,
        parts,
        names.index("c"),
        0,
        np.zeros(graph.size, dtype=np.int64),
        np.empty(graph.size, dtype=np.int64),
        1,
        100,
        out,
    )
    assert sorted(names[node] for node in out[:taken]) == ["a", "b", "c"]


def test_refinement_moves_peninsulas_and_keeps_parts_connected():
    names, graph, parts = peninsula_graph()
    moves = partition._refine(graph, parts, 2, np.array([0.5, 0.5]), 1.0)
    assert moves == 1
    assert {names[node] for node in np.flatnonzero(parts == 1)} == {"p", "c", "a", "b"}
    assert {names[node] for node in np.flatnonzero(parts == 0)} == {"x", "y", "z"}


def test_refinement_balances_a_line():
    graph = line_graph(np.ones(10))
    parts = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int32)
    partition._refine(graph, parts, 2, np.array([0.5, 0.5]), 1.0)
    assert np.array_equal(parts, [0] * 5 + [1] * 5)


def test_fragments_join_the_part_around_them():
    # part 0 = 0..4 and a one-node fragment 7 inside part 1 = 5..9; island 10 stays
    graph = line_graph(np.ones(10))
    graph = partition.PartitionGraph(
        np.r_[graph.weights, 1],
        np.zeros(11),
        np.arange(11, dtype=np.float64),
        graph.edges,
        graph.edge_weights,
    )
    parts = np.array([0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0], dtype=np.int32)
    assert partition._absorb_fragments(graph, parts, 2) == 1
    assert np.array_equal(parts, [0] * 5 + [1] * 5 + [0])


def test_dominant_basin_is_a_part_of_its_own():
    graph = line_graph([70] + [10] * 9)
    parts = partition.assign_basins(graph, 4, imbalance_target=1.0)
    assert parts[0] == 0 and np.all(parts[1:] != 0)
    loads = np.bincount(parts, weights=graph.weights, minlength=4)
    assert np.array_equal(loads, [70, 30, 30, 30])
    for part in (1, 2, 3):
        assert np.all(np.diff(np.flatnonzero(parts == part)) == 1)


def test_basins_a_dominant_basin_cuts_off_stay_with_it():
    # node 0 is dominant; node 1 touches only node 0 (an enclave); 2..9 a line
    weights = np.array([70, 2] + [10] * 8, dtype=np.int64)
    edges = np.array([[0, 1], [0, 2]] + [[i, i + 1] for i in range(2, 9)])
    rows = np.zeros(10)
    cols = np.array([0, -5] + list(range(1, 9)), dtype=np.float64)
    graph = partition.PartitionGraph(
        weights, rows, cols, edges, np.ones(edges.shape[0], dtype=np.int64)
    )
    parts = partition.assign_basins(graph, 4, imbalance_target=1.0)
    assert parts[1] == parts[0]
    assert np.all(parts[2:] != parts[0])


def test_parts_never_span_land_that_does_not_touch():
    # land A: nodes 0..7, land B: nodes 8..11 (apart), an island 12 by B
    weights = np.array([10] * 12 + [1], dtype=np.int64)
    edges = np.array([[i, i + 1] for i in range(7)] + [[i, i + 1] for i in range(8, 11)])
    rows = np.zeros(13)
    cols = np.array(list(range(8)) + [100, 101, 102, 103, 104.5], dtype=np.float64)
    graph = partition.PartitionGraph(
        weights, rows, cols, edges, np.ones(edges.shape[0], dtype=np.int64)
    )
    parts = partition._partition_components(graph, graph, 3, 42, 1.0, True)
    assert set(parts[:8]).isdisjoint(set(parts[8:12]))
    assert len(set(parts[:8])) == 2 and len(set(parts[8:12])) == 1
    assert parts[12] == parts[11]


def test_subbasins_of_a_line_of_tributaries_are_equal_and_connected():
    n = 24
    graph = line_graph(np.full(n, 10))
    parts = partition.assign_subbasins(
        graph,
        np.ones(n, dtype=bool),
        np.arange(n),
        4,
        min_subtree_size=1,
        imbalance_target=1.0,
    )
    assert np.array_equal(np.bincount(parts, minlength=4), np.full(4, 6))
    for part in range(4):
        assert np.all(np.diff(np.flatnonzero(parts == part)) == 1)


def test_mainstem_cells_weigh_with_the_tributary_entering_them():
    stem = np.array([-1, 0, 0, 0, 0])
    position = np.array([-1, 50, 51, 52, 53])
    weights = partition._mainstem_weights(stem, position, np.array([100]))
    assert np.array_equal(weights, [0, 51, 1, 1, 47])
    weights = partition._mainstem_weights(stem, position, np.array([51]))
    assert np.array_equal(weights, [0, 51, 0, 0, 0])
    parts = np.array([0, 0, 1, 1, 2])
    assert np.array_equal(
        partition._p_mins(parts, stem, position, np.array([100])), [51]
    )


def test_metis_never_leaves_a_part_empty():
    graph = line_graph([101, 1])
    parts = partition._metis_parts(graph, 2, np.array([0.5, 0.5]), 42, 1.0)
    assert sorted(parts) == [0, 1]
    graph = line_graph([1, 1, 5, 1, 2])
    parts = partition._fill_empty_parts(graph, np.zeros(5, dtype=np.int32), 2)
    assert np.array_equal(parts, [0, 0, 0, 0, 1]) or np.array_equal(parts, [1, 0, 0, 0, 0])


def test_nodes_without_cells_follow_a_neighbour():
    weights = np.array([1] * 20 + [0], dtype=np.int64)
    rows = np.zeros(21)
    cols = np.r_[np.arange(20) * 2.0, np.nan]
    edges = np.array([[19, 20]])
    graph = partition.PartitionGraph(weights, rows, cols, edges, np.ones(1, dtype=np.int64))
    parts = partition.assign_basins(graph, 4)
    assert np.array_equal(np.bincount(parts[:20], minlength=4), [5, 5, 5, 5])
    assert parts[20] == parts[19]


def connected_parts(parts, edges):
    """Is every part connected over ``edges``?"""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = parts.size
    for part in np.unique(parts):
        members = parts == part
        keep = members[edges[:, 0]] & members[edges[:, 1]]
        matrix = coo_matrix(
            (np.ones(int(keep.sum())), (edges[keep, 0], edges[keep, 1])), shape=(n, n)
        )
        labels = connected_components(matrix, directed=False)[1]
        if np.unique(labels[members]).size > 1:
            return False
    return True


def connected_within_land(parts, edges, n):
    """Is every part one piece within each component of ``edges``?"""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    land = connected_components(
        coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)),
        directed=False,
    )[1]
    for component in np.unique(land):
        nodes = np.flatnonzero(land == component)
        index = np.full(n, -1)
        index[nodes] = np.arange(nodes.size)
        inside = (index[edges[:, 0]] >= 0) & (index[edges[:, 1]] >= 0)
        if not connected_parts(parts[nodes], index[edges[inside]]):
            return False
    return True


def test_a_node_without_cells_still_joins_its_neighbours():
    # a hub without cells (and so without a centroid) is all that joins six basins
    weights = np.array([0, 1, 1, 1, 1, 1, 1], dtype=np.int64)
    rows = np.zeros(7)
    cols = np.array([np.nan, 0, 10, 20, 30, 10.1, 20.1])
    edges = np.column_stack((np.zeros(6, dtype=np.int64), np.arange(1, 7)))
    graph = partition.PartitionGraph(weights, rows, cols, edges, np.ones(6, dtype=np.int64))
    parts = partition.assign_basins(graph, 4)
    assert np.unique(parts).size == 4
    assert connected_parts(parts, edges)


def test_refinement_keeps_parts_joined_across_the_mainstem():
    weights = np.array([22, 14, 11, 3, 24, 9, 17, 12], dtype=np.int64)
    ground = np.array([[0, 3], [0, 6], [1, 4], [2, 6], [3, 7], [4, 5], [4, 6]])
    graph = partition.PartitionGraph(
        weights, np.zeros(8), np.arange(8, dtype=np.float64), ground,
        np.ones(ground.shape[0], dtype=np.int64),
    )
    is_tributary = np.arange(8) > 0
    position = np.arange(8)
    parts = partition.assign_subbasins(
        graph, is_tributary, position, 4, stem_lengths=np.array([8]), min_subtree_size=1
    )
    banks = np.column_stack((np.arange(1, 7), np.arange(2, 8)))
    assert connected_parts(parts, np.concatenate((ground, banks)))


def test_basins_cut_off_by_a_dominant_basin_without_a_part_join_it():
    # a dominant hub with four one-cell basins around it, and three parts besides
    weights = np.array([10, 1, 1, 1, 1], dtype=np.int64)
    edges = np.column_stack((np.zeros(4, dtype=np.int64), np.arange(1, 5)))
    graph = partition.PartitionGraph(
        weights, np.zeros(5), np.array([0.0, -1, 1, 2, 3]), edges, np.ones(4, dtype=np.int64)
    )
    parts = partition.assign_basins(graph, 4)
    assert np.unique(parts).size == 4
    assert connected_parts(parts, edges)


def test_an_archipelago_a_dominant_basin_encloses_joins_it_but_three():
    # a dominant basin around ten one-cell basins: three keep parts, seven join it
    weights = np.array([490] + [1] * 10, dtype=np.int64)
    edges = np.column_stack((np.zeros(10, dtype=np.int64), np.arange(1, 11)))
    graph = partition.PartitionGraph(
        weights, np.zeros(11), np.arange(11, dtype=np.float64), edges,
        np.ones(10, dtype=np.int64),
    )
    parts = partition.assign_basins(graph, 4)
    assert np.array_equal(np.sort(np.bincount(parts, weights=weights)), [1, 1, 1, 497])
    assert connected_parts(parts, edges)


def test_enclaves_that_keep_parts_never_share_one():
    # four one-cell basins inside a dominant one, and four islands far away:
    # three enclaves keep parts of their own, the islands go with them
    weights = np.array([100] + [1] * 8, dtype=np.int64)
    edges = np.column_stack((np.zeros(4, dtype=np.int64), np.arange(1, 5)))
    cols = np.array([0, 1, 2, 3, 4, 100, 200, 300, 400], dtype=np.float64)
    graph = partition.PartitionGraph(
        weights, np.zeros(9), cols, edges, np.ones(4, dtype=np.int64)
    )
    parts = partition.assign_basins(graph, 4)
    assert connected_within_land(parts, edges, 9)
    assert np.unique(parts).size == 4


def test_a_cell_less_tributary_keeps_its_banks_joined():
    weights = np.array([24, 2, 0, 1] + [2] * 31, dtype=np.int64)
    ground = np.array([[0, 2], [1, 2], [2, 3], [1, 4]] + [[k, k + 1] for k in range(4, 34)])
    cols = np.arange(35, dtype=np.float64)
    cols[2] = np.nan
    graph = partition.PartitionGraph(
        weights, np.zeros(35), cols, ground, np.ones(ground.shape[0], dtype=np.int64)
    )
    is_tributary = np.arange(35) > 0
    position = np.full(35, -1)
    position[4:] = np.arange(31)
    position[[1, 2, 3]] = [31, 32, 33]
    parts = partition.assign_subbasins(graph, is_tributary, position, 4)
    order = np.r_[np.arange(4, 35), 1, 2, 3]
    banks = np.column_stack((order[:-1], order[1:]))
    assert connected_parts(parts, np.concatenate((ground, banks)))


def test_a_merged_tributary_keeps_its_mainstem_cells():
    # a star around basin 0; tributary 1 (no centroid) enters the source and is
    # given the first 80 cells of a 113-cell mainstem; 2..34 enter cells 80..112
    weights = np.array([1, 1] + [10] * 33, dtype=np.int64)
    cols = np.arange(35, dtype=np.float64)
    cols[1] = np.nan
    edges = np.column_stack((np.zeros(34, dtype=np.int64), np.arange(1, 35)))
    graph = partition.PartitionGraph(
        weights, np.zeros(35), cols, edges, np.ones(34, dtype=np.int64)
    )
    is_tributary = np.arange(35) > 0
    position = np.r_[-1, 0, np.arange(80, 113)]
    lengths = np.array([113])
    parts = partition.assign_subbasins(
        graph, is_tributary, position, 4, stem_lengths=lengths, min_subtree_size=1,
        imbalance_target=1.09,
    )
    stem = np.where(is_tributary, 0, -1)
    p_min = partition._p_mins(parts, stem, position, lengths)
    loads = np.bincount(
        parts, weights=weights + partition._mainstem_weights(stem, position, p_min), minlength=4
    )
    # it follows basin 0 into the trunk, its 80 cells with it (they were lost: 1.75)
    assert parts[1] == parts[0] and p_min[0] >= 80
    assert loads.max() / loads.mean() < 1.15


def test_land_chosen_for_its_nodes_never_moves_across_water():
    # seven one-node components; 0 is a part, 1..3 are land for want of nodes
    weights = np.array([1000, 30, 25, 24, 20, 20, 20], dtype=np.int64)
    cols = np.array([0, 100, 110, 120, 101, 102, 103], dtype=np.float64)
    graph = partition.PartitionGraph(
        weights, np.zeros(7), cols, np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64)
    )
    parts = partition._partition_components(graph, graph, 4, 42, 1.005, True)
    assert np.unique(parts[:4]).size == 4


def test_a_dense_island_still_finds_the_parts_around_it():
    # four one-node lands and a connected island of twenty nodes by the lightest
    weights = np.r_[120, 80, 120, 120, np.full(20, 2)].astype(np.int64)
    cols = np.r_[0.0, 100, 200, 300, 40 + 0.01 * np.arange(20)]
    edges = np.column_stack((np.arange(4, 23), np.arange(5, 24)))
    graph = partition.PartitionGraph(
        weights, np.zeros(24), cols, edges, np.ones(19, dtype=np.int64)
    )
    parts = partition.assign_basins(graph, 4)
    assert np.array_equal(np.bincount(parts, weights=weights), [120, 120, 120, 120])
    assert np.unique(parts[4:]).size == 1


def test_a_metis_part_left_in_two_pieces_is_joined_up():
    weights = np.array([5, 2, 1, 1, 8, 7, 4, 1, 12, 7], dtype=np.int64)
    edges = np.array([[0, 1], [1, 4], [2, 6], [3, 4], [4, 6], [4, 9], [5, 8], [5, 9], [7, 9]])
    graph = partition.PartitionGraph(
        weights, np.zeros(10), np.arange(10, dtype=np.float64), edges,
        np.array([1, 1, 3, 2, 2, 2, 9, 1, 1], dtype=np.int64),
    )
    for refine in (True, False):
        parts = partition.assign_basins(graph, 4, seed=42, refine=refine)
        assert connected_parts(parts, edges)


def test_a_node_heavier_than_its_share_of_a_land_mass_is_a_part_of_its_own():
    # a land mass of 30 nodes around a heavy middle node that cuts it in two
    # (left 0..9, right 11..20, below 21..29 touching only the middle node),
    # and three parts: the heavy node alone, its enclave with it, each side one
    weights = np.r_[np.ones(10), 100, np.ones(10), np.ones(9)].astype(np.int64)
    edges = np.array(
        [[i, i + 1] for i in range(9)] + [[9, 10], [10, 11]]
        + [[i, i + 1] for i in range(11, 20)] + [[10, 21]] + [[i, i + 1] for i in range(21, 29)]
    )
    graph = partition.PartitionGraph(
        weights, np.zeros(30), np.arange(30, dtype=np.float64), edges,
        np.ones(edges.shape[0], dtype=np.int64),
    )
    parts = partition._partition_components(graph, graph, 3, 42, 1.005, True)
    assert connected_parts(parts, edges)
    assert np.unique(parts[:10]).size == 1 and np.unique(parts[11:21]).size == 1
    assert len({parts[0], parts[10], parts[11]}) == 3 and np.all(parts[21:] == parts[10])


def test_land_masses_get_parts_before_a_dominant_basin():
    # three apart land masses {0,1}, {2,3}, {4}: none shares a part with another
    weights = np.array([200, 100, 200, 100, 90], dtype=np.int64)
    edges = np.array([[0, 1], [2, 3]])
    graph = partition.PartitionGraph(
        weights, np.zeros(5), np.array([0.0, 1, 100, 101, 2]), edges,
        np.ones(2, dtype=np.int64),
    )
    for refine in (True, False):
        parts = partition.assign_basins(graph, 4, refine=refine)
        assert {parts[0], parts[1]}.isdisjoint({parts[2], parts[3], parts[4]})
        assert parts[4] not in (parts[2], parts[3])


def test_a_long_chain_of_cell_less_nodes_follows_its_one_live_node():
    n = 20_000
    weights = np.r_[1, np.zeros(n - 1)].astype(np.int64)
    edges = np.column_stack((np.arange(n - 1), np.arange(1, n)))
    graph = partition.PartitionGraph(
        weights, np.zeros(n), np.arange(n, dtype=np.float64), edges,
        np.ones(n - 1, dtype=np.int64),
    )
    owner = np.r_[0, np.full(n - 1, -1)]
    partition._follow_neighbours(graph, owner, fill=-1)
    assert np.all(owner == 0)


def test_a_group_a_dominant_node_cuts_is_carved_in_pieces():
    # 34 tributaries along one stem, the first one dominant; the only ground
    # edge joins it to node 18, so without it the small ones group across a gap
    weights = np.full(34, 10, dtype=np.int64)
    weights[0], weights[1], weights[18] = 1000, 1, 1
    ground = np.array([[0, 18]])
    graph = partition.PartitionGraph(
        weights, np.zeros(34), np.arange(34, dtype=np.float64), ground, np.ones(1, dtype=np.int64)
    )
    banks = np.column_stack((np.arange(33), np.arange(1, 34)))
    for refine in (True, False):
        parts = partition.assign_subbasins(
            graph, np.ones(34, dtype=bool), np.arange(34), 4, refine=refine
        )
        assert connected_parts(parts, np.concatenate((ground, banks)))


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


def test_whole_basins_stay_whole_and_contiguous():
    topo = rivers(12, [3] * 8)
    part, load = topo.partition(4, "basin", refine=False)
    check(topo, part, load)
    for basin in np.unique(topo.basins[topo.mask]):
        assert np.unique(part[topo.basins == basin]).size == 1
    for value in range(4):
        assert pieces(topo, part, value) == 1
    assert np.array_equal(np.sort(load), [72, 72, 72, 72])


def test_one_basin_opened_into_balanced_connected_subregions():
    topo = rivers(61, [61])
    part, load = topo.partition(4, "subbasin", min_subtree_size=1)
    check(topo, part, load)
    assert load.max() / load.mean() <= 1.04
    for value in range(4):
        assert pieces(topo, part, value) == 1


def test_two_large_basins_are_both_opened_to_balance():
    # 41 x 41 and 41 x 27: both larger than an equal share of the 41 x 68 raster
    topo = rivers(41, [41, 27])
    part, load = topo.partition(4, "subbasin", min_subtree_size=1)
    check(topo, part, load)
    assert load.max() / load.mean() <= 1.04


def test_basins_are_opened_when_whole_basins_are_unequal():
    # five equal basins: none is larger than a share, whole basins give 231 x [1, 1, 1, 2]
    topo = rivers(21, [11] * 5)
    _, load = topo.partition(4, "basin")
    assert load.max() / load.mean() == pytest.approx(1.6)
    part, load = topo.partition(4, "subbasin", min_subtree_size=1)
    check(topo, part, load)
    assert np.any(part == flowtopo.MAINSTEM)
    assert load.max() / load.mean() <= 1.05


def _in_or_beside(cells, region):
    """Whether the 2-D mask ``cells`` lies in ``region`` or shares a side with it."""
    grown = cells.copy()
    grown[1:, :] |= cells[:-1, :]
    grown[:-1, :] |= cells[1:, :]
    grown[:, 1:] |= cells[:, :-1]
    grown[:, :-1] |= cells[:, 1:]
    return bool(np.any(grown & region))


def test_subbasin_level_tries_basins_in_or_beside_the_heaviest_and_lightest_subregions(monkeypatch):
    topo = rivers(21, [11] * 5)
    part_1, load_1 = topo.partition(4, "basin")
    ends = np.isin(part_1, [int(np.argmax(load_1)), int(np.argmin(load_1))]).reshape(topo.shape)
    labels = np.unique(topo.basins[topo.mask & (topo.basins > 0)])
    tried = []
    original = partition._open_basins

    def recording(topo_, opened, *args, **kwargs):
        tried.append(np.asarray(opened).copy())
        return original(topo_, opened, *args, **kwargs)

    monkeypatch.setattr(partition, "_open_basins", recording)
    topo.partition(4, "subbasin", min_subtree_size=1)
    first_round = [opened for opened in tried if opened.size == 1]
    # in or beside every part over the target and the lightest: at most 4 x 2
    assert 1 <= len(first_round) <= 8
    assert len({int(opened[0]) for opened in first_round}) == len(first_round)
    for opened in first_round:
        cells = (topo.basins == labels[opened[0]]).reshape(topo.shape)
        assert _in_or_beside(cells, ends)
    for size in {opened.size for opened in tried}:
        assert sum(opened.size == size for opened in tried) <= 8


def test_subbasin_level_opens_no_basin_without_a_clear_gain(monkeypatch):
    topo = rivers(21, [11] * 5)
    part_1, load_1 = topo.partition(4, "basin")
    monkeypatch.setattr(partition, "OPEN_MIN_GAIN", 10.0)
    part, load = topo.partition(4, "subbasin", min_subtree_size=1)
    assert not np.any(part == flowtopo.MAINSTEM)
    assert np.array_equal(part, part_1)


def test_an_archipelago_is_divided_by_proximity():
    d8 = np.full((1, 39), D8_NODATA, dtype=np.uint8)
    d8[0, ::2] = 0
    topo = build(d8)
    part, load = topo.partition(4, "basin")
    check(topo, part, load)
    assert np.array_equal(np.sort(load), [5, 5, 5, 5])
    for value in range(4):
        columns = np.flatnonzero(part == value)
        assert columns.max() - columns.min() == 8


def test_two_tributaries_entering_one_cell_keep_a_second_stage():
    d8 = np.full((100, 3), D8_NODATA, dtype=np.uint8)
    d8[:-1, 1] = 4
    d8[-1, 1] = 0
    d8[50, 0] = 1
    d8[50, 2] = 16
    topo = build(d8)
    part, load = topo.partition(4, "subbasin", min_subtree_size=1)
    check(topo, part, load)
    assert np.array_equal(np.sort(load), [0, 0, 1, 51])
    assert np.count_nonzero(part == flowtopo.MAINSTEM) == 50


def test_the_trunk_keeps_its_mainstem_above_the_cut():
    # a mainstem column with one-cell tributaries entering it from the east
    d8 = np.full((100, 2), D8_NODATA, dtype=np.uint8)
    d8[:-1, 0] = 4
    d8[-1, 0] = 0
    d8[50:54, 1] = 16
    topo = build(d8)
    part, load = topo.partition(4, "subbasin", min_subtree_size=1)
    check(topo, part, load)
    assert load.max() <= 52 and np.count_nonzero(part == flowtopo.MAINSTEM) >= 47


def test_a_basin_without_tributaries_stays_whole_and_apart():
    # a 180-cell column apart (nodata between), and a 100-cell mainstem that four
    # 250-cell tributaries enter from the east: the column is not opened
    d8 = np.full((180, 253), D8_NODATA, dtype=np.uint8)
    d8[:-1, 0] = 4
    d8[-1, 0] = 0
    d8[:99, 2] = 4
    d8[99, 2] = 0
    d8[50:54, 3:] = 16
    topo = build(d8)
    part, load = topo.partition(4, "subbasin")
    check(topo, part, load)
    grid = part.reshape(topo.shape)
    column = np.unique(grid[:, 0])
    assert column.size == 1 and column[0] >= 0
    assert not np.any(grid[:, 2:] == column[0])


def test_a_basin_that_touches_only_the_mainstem_is_joined_across_it():
    # (2,5) is a one-cell basin whose only neighbour, (1,5), is a mainstem cell
    # of the right basin; the tributary (0,8) is the one that cell weighs with
    code = {".": D8_NODATA, "S": 4, "E": 1, "o": 0}
    rows = ["........S", ".S.S.EEES", "EEEo.o.Eo"]
    topo = build([[code[c] for c in row] for row in rows])
    for refine in (True, False):
        part, load = topo.partition(4, "subbasin", refine=refine)
        check(topo, part, load)
        grid = part.reshape(topo.shape)
        if grid[2, 7] == grid[2, 5]:
            assert grid[0, 8] == grid[2, 5]


@pytest.mark.parametrize("level", ["basin", "subbasin"])
def test_one_subregion_holds_everything(level):
    topo = rivers(21, [11] * 5)
    part, load = topo.partition(1, level)
    assert np.all(part[topo.mask] == 0) and load[0] == topo.ncells


def test_the_options_are_checked():
    topo = rivers(5, [5])
    for options in ({"min_subtree_size": 0}, {"imbalance_target": 0.9}):
        with pytest.raises(ValueError):
            topo.partition(4, "subbasin", **options)
    with pytest.raises(ValueError):
        topo.partition(2.5)
