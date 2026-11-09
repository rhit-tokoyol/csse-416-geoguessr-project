"""Upload the demo to a Hugging Face Space.

One-time setup (in the demo environment):
    pip install huggingface_hub
    hf auth login            # paste a token with "write" access from huggingface.co/settings/tokens

Then, from the project folder:
    python demo_app/deploy_space.py YOUR_HF_USERNAME/where-is-this
    python demo_app/deploy_space.py YOUR_HF_USERNAME/where-is-this --private   # only you can see it

Run it again after any change to re-upload. To test the same files locally
without uploading:
    python demo_app/deploy_space.py --stage space_build
    docker build -t where-is-this space_build && docker run -p 7860:7860 where-is-this
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def space_files():
    """(local file, path in the Space) for everything the Space needs."""
    files = [
        (ROOT / "demo_app/space/Dockerfile", "Dockerfile"),
        (ROOT / "demo_app/space/README.md", "README.md"),
        (ROOT / "demo_app/app.py", "demo_app/app.py"),
        (ROOT / "demo_app/index.html", "demo_app/index.html"),
        (ROOT / "demo_predict.py", "demo_predict.py"),
        (ROOT / "driving_side.py", "driving_side.py"),
    ]
    models = sorted((ROOT / "demo_models").glob("*.pt"))
    if not models:
        sys.exit("No models in demo_models/. Run export_demo.py and download demo_models/ first.")
    files += [(m, f"demo_models/{m.name}") for m in models]
    for local, _ in files:
        if not local.exists():
            sys.exit(f"Missing {local}")
    return files


def stage(out):
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    for local, remote in space_files():
        (out / remote).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, out / remote)
    print(f"Wrote the Space's files to {out}/")


def upload(repo_id, private):
    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError:
        sys.exit("Run `pip install huggingface_hub` first.")
    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception:
        sys.exit("Not logged in to Hugging Face. Run `hf auth login` first.")
    print(f"Logged in as {user}. Uploading to Space {repo_id} ({'private' if private else 'public'})...")
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", private=private, exist_ok=True)
    ops = [CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local)) for local, remote in space_files()]
    api.create_commit(repo_id, repo_type="space", operations=ops, commit_message="Update demo")
    print(f"Done. It builds in a few minutes: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("repo_id", nargs="?", help="USERNAME/space-name on Hugging Face")
    p.add_argument("--private", action="store_true", help="make a new Space private")
    p.add_argument("--stage", metavar="DIR", help="write the Space's files to DIR instead of uploading")
    a = p.parse_args()
    if a.stage:
        stage(a.stage)
    elif a.repo_id:
        upload(a.repo_id, a.private)
    else:
        p.error("give USERNAME/space-name, or --stage DIR")
