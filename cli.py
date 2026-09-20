import argparse
import os
import sys
from typing import Any

import requests

DEFAULT_BASE_URL = os.getenv("RAG_API_URL", "http://localhost:8000")


def main() -> None:
    p = argparse.ArgumentParser(prog="cli")
    p.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="API base URL (default: $RAG_API_URL or http://localhost:8000)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest")
    pi.add_argument("repo_url")
    pi.add_argument("--branch", default=None)
    pi.add_argument("--ext", nargs="+", default=[".py"])
    pi.set_defaults(func=cmd_ingest)

    pq = sub.add_parser("query")
    pq.add_argument("question")
    pq.add_argument("--repo", required=True)
    pq.set_defaults(func=cmd_query)

    pr = sub.add_parser("repos")
    pr.set_defaults(func=cmd_repos)

    args = p.parse_args()
    args.func(args)


def _request(base_url: str, method: str, path: str, **kwargs: Any) -> dict:
    try:
        r = requests.request(method, f"{base_url}{path}", **kwargs)
    except requests.RequestException as e:
        print(f"Connection error: {e}", file=sys.stderr)
        sys.exit(1)

    if not r.ok:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        print(f"Error ({r.status_code}): {detail}", file=sys.stderr)
        sys.exit(1)

    return r.json()


def cmd_ingest(args: argparse.Namespace) -> None:
    body: dict[str, Any] = {"repo_url": args.repo_url, "file_extensions": args.ext}
    if args.branch:
        body["branch"] = args.branch
    d = _request(args.base_url, "POST", "/ingest", json=body)
    print(
        f"Ingested {d['repo_name']}: "
        f"{d['files_indexed']} files, {d['chunks_created']} chunks"
    )


def cmd_query(args: argparse.Namespace) -> None:
    d = _request(
        args.base_url,
        "POST",
        "/query",
        json={"question": args.question, "repo_name": args.repo},
    )
    print("\n" + d["answer"] + "\n")
    print("Citations:")
    for c in d["citations"]:
        name = f" ({c['name']})" if c.get("name") else ""
        print(f"  {c['file_path']}:{c['start_line']}-{c['end_line']}{name}")


def cmd_repos(args: argparse.Namespace) -> None:
    d = _request(args.base_url, "GET", "/repos")
    for repo in d["repos"]:
        print(repo)


if __name__ == "__main__":
    main()
