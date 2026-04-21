"""
shortest_path_bbox.py
─────────────────────
Finds the shortest path through the centres of all bounding boxes in a
shapefile, using:
  1. k-d tree accelerated Nearest Neighbour construction  – O(n log n)
  2. 2-opt local search improvement                       – O(n² per pass)

Dependencies:
    pip install geopandas scipy numpy shapely

Usage:
    python shortest_path_bbox.py --input path/to/your.shp

    # Save result shapefile + CSV:
    python shortest_path_bbox.py --input data.shp --output_shp route.shp --output_csv route.csv

    # Limit 2-opt to a neighbourhood for very large datasets (faster):
    python shortest_path_bbox.py --input data.shp --two_opt_neighbors 20
"""

import argparse
import time
import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point


# ──────────────────────────────────────────────
# 1. Load shapefile and extract centres
# ──────────────────────────────────────────────

def load_centres(shp_path: str) -> tuple[np.ndarray, gpd.GeoDataFrame]:
    """Return (N×2 float array of centres, original GeoDataFrame)."""
    print(f"Loading shapefile: {shp_path}")
    gdf = gpd.read_file(shp_path)
    print(f"  CRS        : {gdf.crs}")
    print(f"  Features   : {len(gdf):,}")

    centres = np.column_stack([
        gdf.geometry.bounds[["minx", "maxx"]].mean(axis=1),
        gdf.geometry.bounds[["miny", "maxy"]].mean(axis=1),
    ])
    return centres, gdf


# ──────────────────────────────────────────────
# 2. Nearest-Neighbour tour (k-d tree)
# ──────────────────────────────────────────────

def nearest_neighbour_tour(centres: np.ndarray, start: int = 0) -> list[int]:
    """
    Greedy nearest-neighbour construction using a k-d tree.
    Returns an ordered list of indices (the tour).
    """
    n = len(centres)
    tree = cKDTree(centres)
    visited = np.zeros(n, dtype=bool)

    tour = [start]
    visited[start] = True
    current = start

    print("Building nearest-neighbour tour …")
    t0 = time.time()

    for step in range(1, n):
        # Query enough neighbours to find the nearest unvisited one.
        # k=min(32,n) is a good balance; fall back to larger k if needed.
        k = min(64, n)
        while True:
            dists, idxs = tree.query(centres[current], k=k)
            # idxs is a 1-D array; find first unvisited
            for idx in idxs:
                if not visited[idx]:
                    next_node = idx
                    break
            else:
                # All k neighbours were visited – expand search
                k = min(k * 2, n)
                continue
            break

        tour.append(next_node)
        visited[next_node] = True
        current = next_node

        if step % 1000 == 0:
            elapsed = time.time() - t0
            print(f"  … {step:,}/{n:,} nodes placed  ({elapsed:.1f}s)")

    print(f"  NN tour built in {time.time()-t0:.2f}s")
    return tour


# ──────────────────────────────────────────────
# 3. Tour length helper
# ──────────────────────────────────────────────

def tour_length(centres: np.ndarray, tour: list[int]) -> float:
    pts = centres[tour]
    diffs = np.diff(pts, axis=0)
    return float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))


# ──────────────────────────────────────────────
# 4. 2-opt improvement
# ──────────────────────────────────────────────

def two_opt(
    centres: np.ndarray,
    tour: list[int],
    max_passes: int = 5,
    neighbor_limit: int | None = None,
) -> list[int]:
    """
    2-opt local search.  For large n, set neighbor_limit (e.g. 20) so that
    each node only considers swaps with its nearest `neighbor_limit` neighbours
    – this trades a small amount of quality for a large speed gain.
    """
    n = len(tour)
    best = tour[:]
    improved = True
    pass_num = 0

    # Pre-build neighbour lists for the candidate set
    if neighbor_limit:
        tree = cKDTree(centres)
        _, neighbor_idx = tree.query(centres, k=min(neighbor_limit + 1, n))
        # neighbor_idx[i] = indices of nearest neighbours of point i
        # Build a position lookup: pos[node] = index in tour
        pos = np.empty(n, dtype=int)
        for i, node in enumerate(best):
            pos[node] = i

    print("Running 2-opt improvement …")
    t0 = time.time()

    while improved and pass_num < max_passes:
        improved = False
        pass_num += 1
        improvements = 0

        for i in range(1, n - 1):
            node_i = best[i - 1]
            node_i1 = best[i]

            if neighbor_limit:
                candidates = neighbor_idx[node_i1]
            else:
                candidates = range(i + 1, n)

            for cand in candidates:
                if neighbor_limit:
                    j = pos[cand]
                    if j <= i:
                        continue
                else:
                    j = cand

                node_j = best[j - 1]
                node_j1 = best[j]

                # Current edges: (i-1→i) and (j-1→j)
                d_old = (
                    np.hypot(*(centres[node_i] - centres[node_i1]))
                    + np.hypot(*(centres[node_j] - centres[node_j1]))
                )
                # Proposed edges: (i-1→j-1) and (i→j)
                d_new = (
                    np.hypot(*(centres[node_i] - centres[node_j]))
                    + np.hypot(*(centres[node_i1] - centres[node_j1]))
                )

                if d_new < d_old - 1e-10:
                    best[i:j] = best[i:j][::-1]
                    improved = True
                    improvements += 1
                    # Update position lookup
                    if neighbor_limit:
                        for k_idx, node in enumerate(best[i:j], start=i):
                            pos[node] = k_idx

        length = tour_length(centres, best)
        print(
            f"  Pass {pass_num}: {improvements:,} improvements  "
            f"| length = {length:,.2f}  ({time.time()-t0:.1f}s)"
        )

    print(f"  2-opt finished in {time.time()-t0:.2f}s")
    return best


# ──────────────────────────────────────────────
# 5. Save outputs
# ──────────────────────────────────────────────

def save_route_shapefile(
    centres: np.ndarray,
    tour: list[int],
    original_gdf: gpd.GeoDataFrame,
    out_shp: str,
) -> None:
    """Save the ordered route as a point shapefile with sequence numbers."""
    ordered = centres[tour]
    gdf_out = gpd.GeoDataFrame(
        {
            "sequence": range(len(tour)),
            "orig_fid": [int(i) for i in tour],
            "x": ordered[:, 0],
            "y": ordered[:, 1],
        },
        geometry=[Point(x, y) for x, y in ordered],
        crs=original_gdf.crs,
    )
    gdf_out.to_file(out_shp)
    print(f"  Route points saved → {out_shp}")

    # Also save the connecting line
    line_shp = out_shp.replace(".shp", "_line.shp")
    line_gdf = gpd.GeoDataFrame(
        {"length": [tour_length(centres, tour)]},
        geometry=[LineString(ordered)],
        crs=original_gdf.crs,
    )
    line_gdf.to_file(line_shp)
    print(f"  Route line  saved → {line_shp}")


def save_route_csv(centres: np.ndarray, tour: list[int], out_csv: str) -> None:
    """Save sequence, original index, x, y to CSV."""
    import csv
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "orig_fid", "x", "y"])
        for seq, idx in enumerate(tour):
            x, y = centres[idx]
            writer.writerow([seq, idx, x, y])
    print(f"  Route CSV   saved → {out_csv}")


# ──────────────────────────────────────────────
# 6. Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Shortest path through bounding-box centres in a shapefile."
    )
    parser.add_argument(
        "--input", required=True, help="Path to input shapefile (.shp)"
    )
    parser.add_argument(
        "--output_shp",
        default=None,
        help="Path for output route shapefile (optional)",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Path for output CSV (optional)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Index of the starting feature (default: 0)",
    )
    parser.add_argument(
        "--two_opt_passes",
        type=int,
        default=5,
        help="Maximum number of 2-opt passes (default: 5)",
    )
    parser.add_argument(
        "--two_opt_neighbors",
        type=int,
        default=None,
        help=(
            "Limit 2-opt candidate swaps to this many nearest neighbours. "
            "Recommended for n > 5000 (e.g. --two_opt_neighbors 20). "
            "Omit for exact (slower) 2-opt."
        ),
    )
    args = parser.parse_args()

    t_start = time.time()

    # --- Load ---
    centres, gdf = load_centres(args.input)
    n = len(centres)

    # Recommend neighbor limit for large datasets
    if n > 5000 and args.two_opt_neighbors is None:
        print(
            f"\n  ⚠  {n:,} features detected.  Consider using "
            "--two_opt_neighbors 20 to speed up 2-opt.\n"
        )

    # --- Build tour ---
    tour = nearest_neighbour_tour(centres, start=args.start)
    length_nn = tour_length(centres, tour)
    print(f"\nNearest-neighbour tour length : {length_nn:,.2f} units")

    # --- Improve ---
    tour = two_opt(
        centres,
        tour,
        max_passes=args.two_opt_passes,
        neighbor_limit=args.two_opt_neighbors,
    )
    length_final = tour_length(centres, tour)
    improvement = 100 * (length_nn - length_final) / length_nn
    print(f"\nFinal tour length : {length_final:,.2f} units")
    print(f"Improvement over NN : {improvement:.2f}%")

    # --- Save ---
    if args.output_shp:
        save_route_shapefile(centres, tour, gdf, args.output_shp)
    if args.output_csv:
        save_route_csv(centres, tour, args.output_csv)

    # Always print the ordered feature indices
    print(f"\nOrdered feature indices (first 20): {tour[:20]} …")
    print(f"\nTotal time: {time.time()-t_start:.2f}s")


if __name__ == "__main__":
    main()
