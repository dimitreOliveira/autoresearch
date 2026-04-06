# autoresearch

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. The training code here is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat). The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` Markdown files that provide context to the AI agents and set up your autonomous research org. The default `program.md` in this repo is intentionally kept as a bare bones baseline, though it's obvious how one would iterate on it over time to find the "research org code" that achieves the fastest research progress, how you'd add more agents to the mix, etc. A bit more context on this project is here in this [tweet](https://x.com/karpathy/status/2029701092347630069) and [this tweet](https://x.com/karpathy/status/2031135152349524125).

## How it works

The repo is deliberately kept small and primarily revolves around these core files:

- **`notebook_runner.ipynb`** — the central orchestrator notebook meant to be run in Colab. It automates the LLM API calls, handles the experiment execution loop, and manages creating automated Pull Requests.
- **`prepare_notebook.py`** — a utility script used by the notebook to prepare the environment (clones the repo, installs dependencies via `uv`, and configures Git credentials).
- **`prepare.py`** — fixed constants, one-time data prep (downloads training data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). Not modified.
- **`train.py`** — the single file the agent edits. Contains the full GPT model, optimizer (Muon + AdamW), and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation), regardless of the details of your compute. The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared.

## Quick start

The experiments are designed to run on a Google Colab instance, which handles the environment setup, runs experiments, and manages GitHub Pull Requests.

**Requirements:** A Google Colab account with a GPU instance. You will also need a GitHub account, a GitHub Personal Access Token (for PRs), and a Gemini API key.

**Setup Instructions:**

1. Open `notebook_runner.ipynb` in Google Colab.
2. Make sure you are using a GPU runtime (Runtime > Change runtime type > T4, L4, V100, or A100 GPU).
3. Run the notebook. It will prompt you for your API keys (GitHub Personal Access Token and Gemini API Key).
4. The notebook cells will then sequentially:
   - Clone the repository and install dependencies via `uv` (`prepare_notebook.py`).
   - Download the training data and build the tokenizer (`prepare.py`).
   - Call the LLM to modify the training code based on `program.md`.
   - Run a single 5-minute training experiment.
   - Commit the changes and open a Pull Request automatically.

If the notebook runs successfully, your Google Colab setup is working correctly and the agent is iterating in an autonomous loop.

## Running the agent

You just need to execute all cells in `notebook_runner.ipynb`. The notebook automates calling the LLM API, modifying the code based on the instructions in `program.md`, running the training, checking the `val_bpb` results, and submitting a Pull Request to your repository.

The `program.md` file serves as essentially a super lightweight "skill" providing baseline instructions for the agent across these notebook runs.

## Project structure

```
notebook_runner.ipynb — central orchestrator for Colab
prepare_notebook.py   — sets up the Colab environment and Git PR workflow
prepare.py            — constants, data prep + runtime utilities (do not modify)
train.py              — model, optimizer, training loop (agent modifies this)
program.md            — agent instructions
pyproject.toml        — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `train_*.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This means you can expect approx 12 experiments/hour and approx 100 experiments while you sleep. There are two upsides of this design decision. First, this makes experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc). Second, this means that autoresearch will find the most optimal model for your platform in that time budget. The downside is that your runs (and results) become not comparable to other people running on other compute platforms.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages. No distributed training, no complex configs. One GPU, one file, one metric.

## License

MIT