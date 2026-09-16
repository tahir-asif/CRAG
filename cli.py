import argparse

import requests

BASE = "http://localhost:8000"


def main():
    p = argparse.ArgumentParser(prog="cli")
    sub = p.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("ingest")
    pi.add_argument("repo_url")
    pi.add_argument(
        "--branch",
        default=None,
        help="Branch to clone (default: repo's default branch)",
    )
    pi.add_argument("--ext", nargs="+", default=[".py"])
    pi.set_defaults(func=cmd_ingest)
    pq = sub.add_parser("query")
    pq.add_argument("question")
    pq.set_defaults(func=cmd_query)
    pr = sub.add_parser("repos")
    pr.set_defaults(func=cmd_repos)
    args = p.parse_args()
    args.func(args)


def cmd_ingest(args):
    body = {"repo_url": args.repo_url, "file_extensions": args.ext}
    if args.branch:
        body["branch"] = args.branch
    r = requests.post(f"{BASE}/ingest", json=body)
    r.raise_for_status()
    d = r.json()
    print(
        f"Ingested {d['repo_name']}: {d['files_indexed']} files, {d['chunks_created']} chunks"
    )


def cmd_query(args):
    r = requests.post(f"{BASE}/query", json={"question": args.question})
    r.raise_for_status()
    d = r.json()
    print("\n" + d["answer"] + "\n")
    print("Citations:")
    for c in d["citations"]:
        name = f" ({c['name']})" if c.get("name") else ""
        print(f"  {c['file_path']}:{c['start_line']}-{c['end_line']}{name}")


def cmd_repos(args):
    r = requests.get(f"{BASE}/repos")
    r.raise_for_status()
    for repo in r.json()["repos"]:
        print(repo)


if __name__ == "__main__":
    main()
