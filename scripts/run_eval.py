import argparse
import sys
from pathlib import Path

from app.dependencies import build_default_dependencies
from app.models import EvalSummary
from scripts.eval.dataset import load_qa_pairs
from scripts.eval.runner import run_eval


def format_report(summary: EvalSummary, repo: str, variant: str) -> str:
    """Build the human-readable report string shown in the console."""
    lines: list[str] = []
    lines.append(
        f"\n=== Eval results for {repo} "
        f"(variant={variant}, {summary.n} questions) ===\n"
    )
    lines.append(f"  Hit Rate@5:   {summary.hit_at_5:.3f}")
    lines.append(f"  Hit Rate@10:  {summary.hit_at_10:.3f}")
    lines.append(f"  MRR:          {summary.mrr:.3f}")
    if summary.keyword_coverage is not None:
        lines.append(f"  Keyword cov:  {summary.keyword_coverage:.3f}")

    lines.append("\nPer-question:")
    for r in summary.results:
        marker = "✓" if r.hit_at_5 else "✗"
        extra = (
            f"  kw={r.keyword_coverage:.2f}" if r.keyword_coverage is not None else ""
        )
        lines.append(f"  {marker} rr={r.rr:.2f}{extra}  {r.question[:70]}")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", default="eval_data/qa_pairs.json")
    parser.add_argument("--repo", required=True)
    parser.add_argument(
        "--variant",
        choices=["full", "no-rerank", "vector-only"],
        default="full",
    )
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()

    qa_pairs = load_qa_pairs(args.qa)
    deps = build_default_dependencies()

    available = deps.store.list_collections()
    if args.repo not in available:
        print(
            f"Repo '{args.repo}' not indexed. Available: {available}",
            file=sys.stderr,
        )
        return 1

    summary = run_eval(
        qa_pairs,
        args.repo,
        store=deps.store,
        embedder=deps.embedder,
        reranker=deps.reranker,
        generate=args.generate,
        variant=args.variant,
    )

    report = format_report(summary, args.repo, args.variant)
    print(report)

    out_dir = Path("eval_data")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"results-{args.repo}-{args.variant}-latest.json"
    txt_path = out_dir / f"results-{args.repo}-{args.variant}-latest.txt"

    json_path.write_text(summary.model_dump_json(indent=2))
    txt_path.write_text(report + "\n")

    print(f"\nWrote {json_path}", file=sys.stderr)
    print(f"Wrote {txt_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
