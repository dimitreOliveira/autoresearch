"""
Prepare notebook environment for autoresearch.
Clones the repository, installs dependencies, and configures Git.

Usage:
    python prepare_notebook.py
    python prepare_notebook.py --github-token YOUR_TOKEN
"""

import argparse
import getpass
import os
import subprocess


def main(repo_url, repo_dir, github_token):
    # Clone the repository if missing
    if not os.path.exists(repo_dir):
        print("Cloning repository...")
        subprocess.run(['git', 'clone', repo_url, repo_dir], capture_output=True)
    else:
        print(f"✅ Repository already exists at {repo_dir}")

    # Navigate into the project directory natively
    os.chdir(repo_dir)
    current_repo_dir = os.path.abspath(".")

    # Sync up dependencies through uv
    print("Synchronizing dependencies using uv...")
    subprocess.run(['uv', 'sync'], capture_output=True)

    # Configure Git
    print("Configuring Git bot identity...")
    subprocess.run(['git', 'config', 'user.email', 'autoresearch@colab'], capture_output=True)
    subprocess.run(['git', 'config', 'user.name', 'AutoResearch Bot'], capture_output=True)

    if github_token:
        print("Setting up GitHub authentication...")
        auth_repo_url = repo_url.replace("https://", f"https://{github_token}@")
        subprocess.run(['git', 'remote', 'set-url', 'origin', auth_repo_url], capture_output=True, cwd=current_repo_dir)
    else:
        print("Skipping GitHub authentication setup (no token provided).")
        
    print("\n✅ Notebook environment prepared.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare the notebook environment")
    parser.add_argument("--repo-url", type=str, default="https://github.com/dimitreOliveira/autoresearch.git", help="The repository URL to clone")
    parser.add_argument("--repo-dir", type=str, default="autoresearch", help="The directory to clone the repository into")
    parser.add_argument("--github-token", type=str, default=None, help="GitHub token for authentication")
    
    args = parser.parse_args()

    token = args.github_token
    if token is None:
        try:
            token_input = getpass.getpass("Please enter your GitHub Token manually (or press Enter to skip): ")
            if token_input.strip() != "":
                token = token_input.strip()
        except EOFError:
            pass

    main(args.repo_url, args.repo_dir, token)
