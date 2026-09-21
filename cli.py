import argparse
import os
import sys
from typing import Any

import requests

DEFAULT_BASE_URL = os.getenv("RAG_API_URL", "http://localhost:8000")


def main() -> None:
    p = argparse.ArgumentParser(
        prog="cli",
        description=(
            "Query public GitHub repositories with natural-language questions.\n"
            "The server must be running before any command is issued."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  Ingest a repository:\n"
            "    $ uv run python cli.py ingest https://github.com/pallets/click\n"
            "\n"
            "  List indexed repositories:\n"
            "    $ uv run python cli.py repos\n"
            "\n"
            "  Ask a question:\n"
            '    $ uv run python cli.py query "How does the echo function work?" --repo click\n'
            "\n"
            "  Point at a different server:\n"
            "    $ uv run python cli.py --base-url http://192.168.1.10:8000 repos\n"
            "\n"
            "  Get help for a specific command:\n"
            "    $ uv run python cli.py ingest --help\n"
            "    $ uv run python cli.py query --help\n"
        ),
    )
    p.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="API base URL (default: $RAG_API_URL or http://localhost:8000)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser(
        "ingest",
        help="Clone and index a public GitHub repository.",
        description=("Clone a public GitHub repository to be queried."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  $ uv run python cli.py ingest https://github.com/pallets/click\n"
            "  $ uv run python cli.py ingest https://github.com/tqdm/tqdm --ext .py .pyi\n"
            "  $ uv run python cli.py ingest https://github.com/psf/requests --branch main\n"
        ),
    )
    pi.add_argument("repo_url")
    pi.add_argument(
        "--branch",
        default=None,
        metavar="NAME",
        help=(
            "Branch to clone. If omitted, uses the repository's default "
            "branch (usually main or master)."
        ),
    )
    pi.add_argument(
        "--ext",
        nargs="+",
        default=[".py"],
        metavar="EXT",
        help="File extensions to index. Default: .py",
    )
    pi.set_defaults(func=cmd_ingest)

    pq = sub.add_parser(
        "query",
        help="Ask a question about an indexed repository.",
        description=(
            "Retrieve relevant code chunks from an indexed repository and "
            "generate an answer with citations. Requires a Groq API key on "
            "the server (or passed via the request)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            '  $ uv run python cli.py query "How does echo work?" --repo click\n'
            '  $ uv run python cli.py query "Where is TMonitor defined?" --repo tqdm\n'
        ),
    )
    pq.add_argument(
        "question",
        help="Natural-language question about the repository.",
    )
    pq.add_argument(
        "--repo",
        required=True,
        metavar="NAME",
        help=("Indexed repository to query. Use `repos` to list what's available."),
    )
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
