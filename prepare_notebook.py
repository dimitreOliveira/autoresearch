"""
Prepare notebook environment for autoresearch.
Clones the repository, installs dependencies, and configures Git.

Usage:
    python prepare_notebook.py
"""

import argparse
import os
import subprocess


def main(repo_dir, train_file):
    # Determine the absolute repository directory
    current_repo_dir = os.path.abspath(repo_dir)

    train_file_base = os.path.splitext(os.path.basename(train_file))[0]

    if train_file_base == "train_tpu_jax":
        print(
            "JAX training script detected. Synchronizing dependencies with jax[tpu] using uv..."
        )
        subprocess.run(
            ["uv", "pip", "install", "--system", ".[colab_tpu_jax]"],
            cwd=current_repo_dir,
            check=True,
        )
    elif train_file_base == "train_tpu_pytorch":
        print(
            "PyTorch TPU training script detected. Synchronizing dependencies with torch_xla using uv..."
        )
        subprocess.run(
            ["uv", "pip", "install", "--system", ".[colab_pytorch_tpu]"],
            cwd=current_repo_dir,
            check=True,
        )
    elif train_file_base == "train_gpu_pytorch":
        print(
            "PyTorch GPU training script detected. Synchronizing dependencies using uv..."
        )
        subprocess.run(
            ["uv", "sync", "--extra", "colab_pytorch_gpu"],
            cwd=current_repo_dir,
            check=True,
        )
    else:
        raise ValueError(f"Unknown train file: {train_file}")

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
    parser.add_argument(
        "--train-file",
        type=str,
        default="train_gpu_pytorch.py",
        help="The training file to use, determines dependencies",
    )
    args = parser.parse_args()

    main(args.repo_dir, args.train_file)
