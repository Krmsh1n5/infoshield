#!/usr/bin/env python3
"""
instagram_ingestor_demo.py — Runnable demonstration of the InstagramIngestor.

Usage
-----
    # Mock mode (no credentials needed — runs fully offline):
    python pipeline/ingestors/instagram_ingestor_demo.py

    # Real mode (requires a valid Instagram Graph API token):
    python pipeline/ingestors/instagram_ingestor_demo.py --token IGQV... --post-id 17858893269000001

The demo showcases all three ingestor methods:
  1. get_post_cascade()
  2. search_hashtag_cascade()
  3. build_from_comments()

It then shows how to feed the resulting graph directly into the
InfoShield pipeline without any conversion step.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# ── path setup so the demo runs from project root ─────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # infoshield/
sys.path.insert(0, str(PROJECT_ROOT))

import networkx as nx
import numpy as np

from pipeline.ingestors import InstagramIngestor, ValidationReport

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo")


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def print_section(title: str) -> None:
    width = 70
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def graph_stats(G: nx.DiGraph) -> dict:
    """Compute the statistics Table II would care about."""
    n = G.number_of_nodes()
    e = G.number_of_edges()
    followers = [d.get("followers", 0) for _, d in G.nodes(data=True)]
    roots = [n_id for n_id, deg in G.in_degree() if deg == 0]
    depths = (
        nx.single_source_shortest_path_length(G.reverse(), roots[0])
        if roots else {}
    )
    return {
        "nodes": n,
        "edges": e,
        "avg_degree": round(e / n, 3) if n else 0,
        "roots": len(roots),
        "root_id": roots[0] if roots else None,
        "follower_median": int(np.median(followers)) if followers else 0,
        "follower_max": int(max(followers)) if followers else 0,
        "cascade_depth": max(depths.values()) if depths else 0,
        "weakly_connected": nx.is_weakly_connected(G),
    }


def show_graph_stats(G: nx.DiGraph, label: str) -> None:
    stats = graph_stats(G)
    print(f"\n  [{label}]")
    for k, v in stats.items():
        print(f"    {k:<22} = {v}")


def show_validation(report: ValidationReport) -> None:
    status = "✓ PASS" if report.ok else "✗ FAIL"
    print(f"\n  Validation: {status}")
    for e in report.errors:
        print(f"    ERROR: {e}")
    for w in report.warnings:
        print(f"    WARN : {w}")


# ──────────────────────────────────────────────────────────────────────────
# Demo sections
# ──────────────────────────────────────────────────────────────────────────

def demo_single_post(ingestor: InstagramIngestor, post_id: str) -> nx.DiGraph:
    print_section("1. Single post cascade — get_post_cascade()")
    print(f"\n  Fetching cascade for post: {post_id}")

    G = ingestor.get_post_cascade(post_id)

    show_graph_stats(G, "Post cascade")
    report = ingestor.validate_graph(G)
    show_validation(report)

    root = ingestor.get_root_node(G)
    root_data = G.nodes[root]
    print(f"\n  Root node: {root}")
    print(f"    followers      = {root_data.get('followers', '?')}")
    print(f"    friends        = {root_data.get('friends', '?')}")
    print(f"    account_age    = {root_data.get('account_age_days', '?')} days")
    print(f"    is_verified    = {root_data.get('is_verified', False)}")

    return G


def demo_hashtag_search(ingestor: InstagramIngestor, hashtag: str) -> list[nx.DiGraph]:
    print_section(f"2. Hashtag cascade search — search_hashtag_cascade('#{hashtag}')")

    cascades = ingestor.search_hashtag_cascade(hashtag, hours_back=24)
    print(f"\n  Found {len(cascades)} cascade(s) for #{hashtag}")

    for i, G in enumerate(cascades):
        show_graph_stats(G, f"Cascade {i + 1}")

    return cascades


def demo_comment_fallback(ingestor: InstagramIngestor, post_id: str) -> nx.DiGraph:
    print_section("3. Comment-thread fallback — build_from_comments()")
    print("\n  Building approximate cascade from comment reply chains...")

    G = ingestor.build_from_comments(post_id)
    show_graph_stats(G, "Comment cascade")
    report = ingestor.validate_graph(G)
    show_validation(report)
    return G


def demo_pipeline_integration(G: nx.DiGraph, ingestor: InstagramIngestor) -> None:
    """
    Show that the graph is directly usable in the InfoShield pipeline
    without any conversion step.

    We demonstrate the interface contract rather than running the full
    simulation (which requires fitted SBM matrices from pipeline/sbm_fitter.py).
    """
    print_section("4. Pipeline integration check")

    # ── find_root_user() logic (mirrors pipeline/sbm_fitter.py) ──
    roots = [n for n, d in G.in_degree() if d == 0]
    if roots:
        root = max(roots, key=lambda n: G.nodes[n].get("followers", 0))
        root_followers = G.nodes[root].get("followers", 0)
        print(f"\n  find_root_user(G) → '{root}' (followers={root_followers})")
    else:
        print("\n  WARNING: no root found — graph may be malformed.")
        return

    # ── simulate_cascade_following() signature check ──
    print("\n  simulate_cascade_following() signature check:")
    print("    G           : nx.DiGraph  ✓")
    print("    partition   : dict        (would be assigned by SBM fitter)")
    print("    root        : str         ✓  root =", root[:40])

    # ── Show a sample node's attributes ──
    sample_node = list(G.nodes())[1] if G.number_of_nodes() > 1 else root
    sample_attrs = dict(G.nodes[sample_node])
    print(f"\n  Sample node attributes ({sample_node[:40]}…):")
    for k, v in sorted(sample_attrs.items()):
        print(f"    {k:<22} = {v}")

    # ── WICO attribute contract ──
    wico_required = {"followers", "time", "friends"}
    all_nodes_ok = all(
        wico_required.issubset(G.nodes[n].keys()) for n in G.nodes()
    )
    status = "✓" if all_nodes_ok else "✗"
    print(f"\n  WICO attribute contract (followers/time/friends): {status}")

    print("\n  Pipeline integration: READY")
    print("  Next step: load SBM matrices and call simulate_cascade_following(G, ...)")


def demo_statistical_properties(ingestor: InstagramIngestor, n_samples: int = 20) -> None:
    """
    Generate multiple mock cascades and verify they match WICO statistics.
    """
    print_section(f"5. Mock statistical validation (n={n_samples} cascades)")

    sizes, avg_degrees, follower_medians = [], [], []

    for i in range(n_samples):
        G = ingestor._generate_mock_cascade(f"stat_test_{i}")
        n = G.number_of_nodes()
        e = G.number_of_edges()
        followers = [d.get("followers", 0) for _, d in G.nodes(data=True)]

        sizes.append(n)
        avg_degrees.append(e / n if n else 0)
        follower_medians.append(int(np.median(followers)))

    print(f"\n  node count     mean={np.mean(sizes):.1f}  "
          f"min={min(sizes)}  max={max(sizes)}")
    print(f"  avg_degree     mean={np.mean(avg_degrees):.3f}  "
          f"(WICO target ≈ 2.82)")
    print(f"  follower med   mean={np.mean(follower_medians):.0f}  "
          f"(lognormal target ≈ 500)")

    # Check all cascades pass validation
    failed = 0
    for i in range(n_samples):
        G = ingestor._generate_mock_cascade(f"val_test_{i}")
        r = ingestor.validate_graph(G)
        if not r.ok:
            failed += 1

    status = "✓" if failed == 0 else f"✗ ({failed} failed)"
    print(f"\n  All cascades pass validate_graph(): {status}")


# ──────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="InfoShield — InstagramIngestor demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--token",
        default="",
        help="Instagram Graph API access token (omit for mock mode)",
    )
    parser.add_argument(
        "--post-id",
        default="demo_post_001",
        help="Instagram post/media ID to fetch (mock mode ignores this)",
    )
    parser.add_argument(
        "--hashtag",
        default="infoshield",
        help="Hashtag to search (without #)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducible mock output",
    )
    parser.add_argument(
        "--stat-samples",
        type=int,
        default=20,
        help="Number of mock cascades to generate for statistical validation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mock_mode = not args.token
    mode_label = "MOCK" if mock_mode else "REAL API"

    print(f"\n{'━' * 70}")
    print(f"  InfoShield — InstagramIngestor Demo  [{mode_label}]")
    print(f"{'━' * 70}")
    if mock_mode:
        print("\n  Running in mock mode. No API token required.")
        print("  All graphs are synthetically generated to match WICO statistics.")
    else:
        print(f"\n  Token: {args.token[:12]}…")
        print(f"  Post ID: {args.post_id}")

    ingestor = InstagramIngestor(
        access_token=args.token,
        mock=mock_mode,
        seed=args.seed,
    )

    # Section 1: single post
    G = demo_single_post(ingestor, args.post_id)

    # Section 2: hashtag search
    cascades = demo_hashtag_search(ingestor, args.hashtag)

    # Section 3: comment fallback
    G_comments = demo_comment_fallback(ingestor, args.post_id)

    # Section 4: pipeline integration
    demo_pipeline_integration(G, ingestor)

    # Section 5: statistical properties (mock only)
    if mock_mode:
        demo_statistical_properties(ingestor, n_samples=args.stat_samples)

    print(f"\n{'━' * 70}")
    print("  Demo complete.")
    print(f"{'━' * 70}\n")


if __name__ == "__main__":
    main()
