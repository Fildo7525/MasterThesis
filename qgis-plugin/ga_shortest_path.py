import argparse
import random
import time
from pathlib import Path

import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree
from shapely.geometry import LineString

POPULATION_SIZE = 150
GENERATIONS = 150
MUTATION_RATE = 0.015
TOURNAMENT_SIZE = 5
ELITE_COUNT = round(0.1 * POPULATION_SIZE)
SEED = 45
STOP_IF_UNCHANGED = 10
POLISH_ELITES = True

# ══════════════════════════════════════════════════════════════════════════════
# 1. Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_centres(shp_path: Path | str) -> tuple[np.ndarray, gpd.GeoDataFrame]:
    """
    Read the shapefile and return (centres, gdf).
    centres : float64 array of shape (N, 2) – one (x, y) per bounding box centre
    gdf     : original GeoDataFrame (used for CRS when saving output)
    """
    print(f"\n{'═'*60}")
    print(f"  Loading : {shp_path}")
    gdf = gpd.read_file(shp_path)
    bounds = gdf.geometry.bounds                          # minx miny maxx maxy
    centres = np.column_stack([
        (bounds["minx"].values + bounds["maxx"].values) / 2.0,
        (bounds["miny"].values + bounds["maxy"].values) / 2.0,
    ])
    print(f"  CRS     : {gdf.crs}")
    print(f"  Points  : {len(centres):,}")
    print(f"{'═'*60}\n")
    return centres.astype(np.float64), gdf


# ══════════════════════════════════════════════════════════════════════════════
# 2. Tour helpers
# ══════════════════════════════════════════════════════════════════════════════

def tour_length(centres: np.ndarray, tour: np.ndarray) -> float:
    """Total Euclidean length of an open path (not a closed loop)."""
    pts = centres[tour]
    diffs = np.diff(pts, axis=0)
    return float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))


def nn_tour(centres: np.ndarray, start: int, tree: cKDTree) -> np.ndarray:
    """
    Greedy Nearest-Neighbour tour starting from `start`.
    Uses a pre-built cKDTree for O(n log n) construction.
    """
    n = len(centres)
    visited = np.zeros(n, dtype=bool)
    tour = np.empty(n, dtype=np.int32)
    tour[0] = start
    visited[start] = True
    current = start

    k = min(64, n)
    for step in range(1, n):
        while True:
            _, idxs = tree.query(centres[current], k=k)
            idxs = np.atleast_1d(idxs)
            found = None
            for idx in idxs:
                if not visited[idx]:
                    found = idx
                    break
            if found is not None:
                break
            k = min(k * 2, n)          # widen search if all neighbours visited
        tour[step] = found
        visited[found] = True
        current = found
        k = min(64, n)                 # reset k for next step

    return tour


# ══════════════════════════════════════════════════════════════════════════════
# 3. Genetic Algorithm components
# ══════════════════════════════════════════════════════════════════════════════

# ── 3a. Initialisation ──────────────────────────────────────────────────────

def initialise_population(
    centres: np.ndarray,
    pop_size: int,
    nn_seeds: int,
) -> list[np.ndarray]:
    """
    Build the initial population.
    nn_seeds tours are built with the NN heuristic (from random starts).
    The remainder are random shuffles – this keeps diversity high.
    """
    n = len(centres)
    tree = cKDTree(centres)
    population = []

    nn_count = min(nn_seeds, pop_size)
    # print(f"  Seeding {nn_count} NN tours …", flush=True)
    used_starts = set()
    for i in range(nn_count):
        start = random.randint(0, n - 1)
        while start in used_starts and len(used_starts) < n:
            start = random.randint(0, n - 1)
        used_starts.add(start)
        population.append(nn_tour(centres, start, tree))
        # print(f"    NN tour {i+1}/{nn_count}  length={tour_length(centres, population[-1]):,.1f}")

    random_count = pop_size - nn_count
    print(f"  Adding {random_count} random tours …", flush=True)
    base = np.arange(n, dtype=np.int32)
    for _ in range(random_count):
        t = base.copy()
        np.random.shuffle(t)
        population.append(t)

    return population


# ── 3b. Selection ───────────────────────────────────────────────────────────

def tournament_select(
    population: list[np.ndarray],
    fitnesses: np.ndarray,
    k: int,
) -> np.ndarray:
    """Tournament selection: pick k candidates, return the best."""
    idxs = random.sample(range(len(population)), k)
    best = min(idxs, key=lambda i: fitnesses[i])
    return population[best].copy()


# ── 3c. Crossover (Order Crossover / OX1) ───────────────────────────────────

def order_crossover(parent_a: np.ndarray, parent_b: np.ndarray) -> np.ndarray:
    """
    OX1 crossover.
    1. Copy a random slice from parent A into the child.
    2. Fill remaining positions with the order they appear in parent B,
       skipping nodes already placed.
    """
    n = len(parent_a)
    lo, hi = sorted(random.sample(range(n), 2))

    child = np.full(n, -1, dtype=np.int32)
    child[lo:hi] = parent_a[lo:hi]

    # Set of already placed nodes (fast membership test)
    placed = set(child[lo:hi].tolist())

    # Fill from parent B in order, starting just after the slice
    fill_pos = hi % n
    for node in np.roll(parent_b, n - hi):
        if node not in placed:
            child[fill_pos] = node
            placed.add(node)
            fill_pos = (fill_pos + 1) % n

    return child


# ── 3d. Mutation (2-opt segment reversal) ───────────────────────────────────

def mutate_2opt(tour: np.ndarray, mutation_rate: float) -> np.ndarray:
    """
    Apply a single random 2-opt reversal with probability `mutation_rate`.
    This is far more effective than simple swap mutation for TSP.
    """
    if random.random() < mutation_rate:
        n = len(tour)
        lo, hi = sorted(random.sample(range(n), 2))
        tour[lo:hi] = tour[lo:hi][::-1]
    return tour


# ── 3e. Local 2-opt polish on a single tour ─────────────────────────────────

def quick_two_opt(centres: np.ndarray, tour: np.ndarray, passes: int = 1) -> np.ndarray:
    """
    Fast neighbour-limited 2-opt polish applied to a single tour.
    Used to improve elite individuals each generation.
    """
    n = len(tour)
    tree = cKDTree(centres)
    k_nn = min(15, n)
    _, nn_idx = tree.query(centres, k=k_nn + 1)  # +1 because self is index 0
    nn_idx = nn_idx[:, 1:]                         # drop self

    pos = np.empty(n, dtype=np.int32)
    for i, node in enumerate(tour):
        pos[node] = i

    best = tour.copy()
    for _ in range(passes):
        for i in range(1, n - 1):
            ni = best[i - 1]
            ni1 = best[i]
            ci = centres[ni]
            ci1 = centres[ni1]

            for cand in nn_idx[ni1]:
                j = pos[cand]
                if j <= i:
                    continue
                nj = best[j - 1]
                nj1 = best[j]

                d_old = (np.hypot(*(ci - ci1)) +
                         np.hypot(*(centres[nj] - centres[nj1])))
                d_new = (np.hypot(*(ci - centres[nj])) +
                         np.hypot(*(ci1 - centres[nj1])))

                if d_new < d_old - 1e-10:
                    best[i:j] = best[i:j][::-1]
                    for k_idx, node in enumerate(best[i:j], start=i):
                        pos[node] = k_idx
    return best


# ══════════════════════════════════════════════════════════════════════════════
# 4. Main GA loop
# ══════════════════════════════════════════════════════════════════════════════

def run_ga(
    centres: np.ndarray,
    pop_size: int = 60,
    generations: int = 150,
    mutation_rate: float = 0.015,
    tournament_size: int = 5,
    elite_count: int = 3,
    nn_seeds: int = 10,
    patience: int = 0,          # 0 = no early stopping
    polish_elites: bool = True,  # run quick 2-opt on elite each generation
) -> np.ndarray:
    """
    Run the Genetic Algorithm and return the best tour found.
    """
    n = len(centres)
    t0 = time.time()

    # ── Initialise ──────────────────────────────────────────────────────────
    print("Initialising population …")
    population = initialise_population(centres, pop_size, nn_seeds)
    fitnesses = np.array([tour_length(centres, t) for t in population])

    best_idx = int(np.argmin(fitnesses))
    best_tour = population[best_idx].copy()
    best_length = fitnesses[best_idx]
    no_improve_count = 0

    print(f"\nStarting GA  |  pop={pop_size}  gen={generations}  "
          f"mut={mutation_rate}  tourn={tournament_size}  elite={elite_count}")
    print(f"{'─'*60}")
    print(f"  Gen {'Gen':>4}  |  Best length       |  Elapsed")
    print(f"{'─'*60}")

    for gen in range(1, generations + 1):
        # ── Sort by fitness ────────────────────────────────────────────────
        order = np.argsort(fitnesses)
        elites = [population[i].copy() for i in order[:elite_count]]

        # ── Optional: polish elites with quick 2-opt ───────────────────────
        if polish_elites:
            elites = [quick_two_opt(centres, e) for e in elites]
            elite_fits = [tour_length(centres, e) for e in elites]
        else:
            elite_fits = [fitnesses[i] for i in order[:elite_count]]

        # ── Breed next generation ──────────────────────────────────────────
        next_pop = elites[:]
        next_fits = elite_fits[:]

        while len(next_pop) < pop_size:
            pa = tournament_select(population, fitnesses, tournament_size)
            pb = tournament_select(population, fitnesses, tournament_size)
            child = order_crossover(pa, pb)
            child = mutate_2opt(child, mutation_rate)
            length = tour_length(centres, child)
            next_pop.append(child)
            next_fits.append(length)

        population = next_pop
        fitnesses = np.array(next_fits)

        # ── Track best ────────────────────────────────────────────────────
        gen_best_idx = int(np.argmin(fitnesses))
        gen_best = fitnesses[gen_best_idx]

        if gen_best < best_length - 1e-6:
            best_length = gen_best
            best_tour = population[gen_best_idx].copy()
            no_improve_count = 0
        else:
            no_improve_count += 1

        elapsed = time.time() - t0
        print(f"  Gen {gen:>4}  |  {best_length:>18,.2f}  |  {elapsed:6.1f}s")

        # ── Early stopping ─────────────────────────────────────────────────
        if patience and no_improve_count >= patience:
            print(f"\n  Early stop: no improvement for {patience} generations.")
            break

    print(f"{'─'*60}")
    print(f"  GA complete in {time.time()-t0:.1f}s")
    print(f"  Best tour length : {best_length:,.2f} units")
    return best_tour


# ══════════════════════════════════════════════════════════════════════════════
# 5. Save output shapefile
# ══════════════════════════════════════════════════════════════════════════════

def save_route_shapefile(
    centres: np.ndarray,
    tour: np.ndarray,
    source_gdf: gpd.GeoDataFrame,
    out_path: str,
) -> None:
    """
    Write the route as a shapefile containing a single straight-line
    LineString connecting the centres in tour order.
    The CRS is inherited from the source shapefile.
    """
    ordered_pts = centres[tour]                      # (N, 2) in visit order
    line = LineString(ordered_pts)                   # straight segments only

    length = tour_length(centres, tour)
    gdf_out = gpd.GeoDataFrame(
        {
            "n_points": [len(tour)],
            "length":   [round(length, 4)],
        },
        geometry=[line],
        crs=source_gdf.crs,
    )

    out_path = str(out_path)
    gdf_out.to_file(out_path)
    print(f"\n  Route saved → {out_path}")
    print(f"  LineString  : {len(tour):,} vertices, length = {length:,.2f} units")


# ══════════════════════════════════════════════════════════════════════════════
# 6. CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Genetic Algorithm shortest path through bounding-box centres.\n"
            "Output: a single-feature LineString shapefile."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--input", required=True, metavar="SHP",
        help="Input shapefile containing bounding boxes.",
    )
    p.add_argument(
        "--output", required=True, metavar="SHP",
        help="Output shapefile for the straight-line route.",
    )

    # ── GA parameters ──────────────────────────────────────────────────────
    g = p.add_argument_group("GA parameters")
    g.add_argument(
        "--pop_size", type=int, default=60, metavar="N",
        help="Population size (default: 60). Larger → better quality, slower.",
    )
    g.add_argument(
        "--generations", type=int, default=150, metavar="N",
        help="Maximum number of generations (default: 150).",
    )
    g.add_argument(
        "--mutation_rate", type=float, default=0.015, metavar="F",
        help="Probability of mutating a child (default: 0.015).",
    )
    g.add_argument(
        "--tournament_size", type=int, default=5, metavar="N",
        help="Tournament size for selection (default: 5).",
    )
    g.add_argument(
        "--elite_count", type=int, default=3, metavar="N",
        help="Number of elites to carry over each generation (default: 3).",
    )
    g.add_argument(
        "--nn_seeds", type=int, default=10, metavar="N",
        help=(
            "Number of Nearest-Neighbour tours used to seed the population "
            "(default: 10). The rest are random."
        ),
    )
    g.add_argument(
        "--patience", type=int, default=0, metavar="N",
        help=(
            "Stop early if no improvement for N generations "
            "(default: 0 = disabled)."
        ),
    )
    g.add_argument(
        "--no_polish", action="store_true",
        help="Disable the quick 2-opt polish applied to elites each generation.",
    )
    g.add_argument(
        "--seed", type=int, default=None, metavar="N",
        help="Random seed for reproducibility (default: None).",
    )

    return p.parse_args()


def compute_shortest_route(inp, output) -> None:
    centres, gdf = load_centres(inp)

    if len(centres) < 3:
        # raise ValueError("Shapefile must contain at least 3 features.")
        print("Shapefile must contain at least 3 features.")
        return


    # Warn if dataset is very large
    if len(centres) > 5000:
        print(
            f"  ⚠  {len(centres):,} features detected.\n"
            "     The GA will run, but each generation is slow at this scale.\n"
            "     Recommended: keep --pop_size ≤ 60 and use --patience 20\n"
            "     to stop automatically when converged.\n"
        )

    # ── Run GA ──────────────────────────────────────────────────────────────
    best_tour = run_ga(
        centres,
        pop_size=POPULATION_SIZE,
        generations=GENERATIONS,
        mutation_rate=MUTATION_RATE,
        tournament_size=TOURNAMENT_SIZE,
        elite_count=ELITE_COUNT,
        nn_seeds=SEED,
        patience=STOP_IF_UNCHANGED,
        polish_elites=POLISH_ELITES,
    )

    # ── Save ────────────────────────────────────────────────────────────────
    save_route_shapefile(centres, best_tour, gdf, output)


if __name__ == "__main__":
    POPULATION_SIZE = 150
    GENERATIONS = 150
    MUTATION_RATE = 0.015
    TOURNAMENT_SIZE = 5
    ELITE_COUNT = round(0.1 * POPULATION_SIZE)
    SEED = 45
    STOP_IF_UNCHANGED = 10
    POLISH_ELITES = True

    random.seed(SEED)
    np.random.seed(SEED)

    # inp = Path.home() / "SDU/MasterThesis/OpenCV"
    # shapefiles = [ f for f in inp.rglob("*.shp") ]
    # for shapefile in shapefiles:
    ai_shp = Path.home() / "Downloads/predictions_all_mosaics/small/yolo_small_shp.shp"
    compute_shortest_route(ai_shp, str(ai_shp).replace(".shp", "_ga_path.shp"))
        # print()

