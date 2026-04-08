"""
Prepare notebook environment for autoresearch.
Clones the repository, installs dependencies, and configures Git.

Usage:
    python prepare_notebook.py
"""

import argparse
import os
import subprocess


def main(repo_dir):
    # Determine the absolute repository directory
    current_repo_dir = os.path.abspath(repo_dir)

    # Determine if we are on a Colab TPU
    tpu_env = os.environ.get("COLAB_TPU_ADDR") or os.environ.get("TPU_NAME")

    if tpu_env:
        print(
            "TPU runtime detected. Installing dependencies into system Python to preserve torch_xla..."
        )
        # uv pip install --system installs into Colab's default environment
        subprocess.run(
            ["uv", "pip", "install", "--system", "-e", ".[colab]"],
            capture_output=True,
            cwd=current_repo_dir,
        )
    else:
        # Sync up dependencies through uv in an isolated venv
        print("Synchronizing dependencies using uv...")
        subprocess.run(
            ["uv", "sync", "--extra", "colab"],
            capture_output=True,
            cwd=current_repo_dir,
        )

    # Configure Git
    print("Configuring Git bot identity...")
    subprocess.run(
        ["git", "config", "user.email", "autoresearch@colab"],
        capture_output=True,
        cwd=current_repo_dir,
    )
    subprocess.run(
        ["git", "config", "user.name", "AutoResearch Bot"],
        capture_output=True,
        cwd=current_repo_dir,
    )

    # Use GITHUB_TOKEN from current environment if it exists
    token_to_use = os.environ.get("GITHUB_TOKEN")

    if token_to_use:
        print("Setting up GitHub authentication...")
        subprocess.run(
            [
                "git",
                "config",
                "credential.helper",
                '!f() { echo "username=git"; echo "password=${GITHUB_TOKEN}"; }; f',
            ],
            capture_output=True,
            cwd=current_repo_dir,
        )
    else:
        print("Skipping GitHub authentication setup (no token provided).")

    print("\n✅ Notebook environment prepared.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare the notebook environment")
    parser.add_argument(
        "--repo-dir",
        type=str,
        default="autoresearch",
        help="The directory to clone the repository into",
    )
    args = parser.parse_args()

    main(args.repo_dir)
