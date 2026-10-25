# kernel-lab

Seven Triton exercises that build up the kernels of a transformer layer, from vector add to
a fused-MoE grouped GEMM. Each comes with tests, a benchmark against PyTorch, and a
roofline report so you know how close to the hardware limit you are.

| # | Exercise | You learn |
|---|----------|-----------|
| 1 | `ex01_vector_add` | programs, offsets, masks; your GPU's real bandwidth |
| 2 | `ex02_swiglu` | elementwise fusion, fp32 math, 2-D tiles and strides |
| 3 | `ex03_rmsnorm` | row reductions, fusing residual add + norm |
| 4 | `ex04_matmul` | tiled GEMM, `tl.dot`, pipelining, autotuning, L2 grouping |
| 5 | `ex05_matmul_swiglu` | epilogue fusion, operand reuse, register pressure |
| 6 | `ex06_attention` | stable softmax, online softmax, causal FlashAttention |
| 7 | `ex07_moe` | grouped GEMM, gather/scatter, padding vs tokens per expert |

## Setup (once)

You need a Mac or Linux machine, a GitHub account and a Google account. No local GPU needed:
tests run on your machine on Triton's CPU interpreter, and benchmarks run on a free Colab GPU,
driven from your editor.

1. **Install the tools:**
   - [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python and dependencies)
   - [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Triton's CPU
     interpreter only runs on Linux, so local tests run in a container)
   - [VS Code](https://code.visualstudio.com) or [Cursor](https://cursor.com), plus its shell
     command so `poe open` can launch it: Command Palette → "Shell Command: Install 'code'
     command" (or `'cursor'`)
2. **Get your own copy of the repo.** On GitHub, click **Use this template** → **Create a new
   repository** (private is fine, and keeps your solutions to yourself), or fork it. The Colab
   GPU works from your GitHub repo, so you need push access.
3. **Install dependencies and build the test image:**

   ```
   uv sync
   uv run poe setup      # Docker Desktop must be running
   uv run poe test 1     # should fail on the TODOs, not on setup errors
   ```

4. **Connect your editor to a Colab GPU.**
   - Install the recommended extensions when the editor prompts you (listed in
     `.vscode/extensions.json`): **Colab** by Google, Jupyter and Python.
   - Open `colab.ipynb` → **Select Kernel** (top right) → **Colab** → **New Colab Server** →
     **T4 GPU**. Sign in with Google when the browser opens.
   - In the notebook, set `REPO` to your repo.
   - Run the first two cells. The second prints your GPU name. For a private repo it first
     asks for a GitHub token: create a [fine-grained token](https://github.com/settings/personal-access-tokens/new)
     limited to this repo with **Contents: read-only**, and paste it. Each new Colab
     runtime asks again. In the browser you can store it as a Colab secret named
     `GITHUB_TOKEN` instead (key icon in the left sidebar).

   Prefer the browser? Open the notebook there instead:
   `https://colab.research.google.com/github/<you>/kernel-lab/blob/main/colab.ipynb`.

## Workflow

Commands run from the repo root through [poe](https://poethepoet.natn.io/) tasks (defined
under `[tool.poe.tasks]` in `pyproject.toml`, scripts in `scripts/`). Run `uv run poe` to list them.

1. `uv run poe open 1` opens an exercise in your editor (README > kernel > hints). Edit
   `exNN_*/kernel.py`; only the `TODO`s need filling.
2. `uv run poe test 1` runs that exercise's tests on the CPU interpreter (slow, correctness
   only). Extra args go to pytest, e.g. `uv run poe test 6 -k softmax`.
3. Once green, `uv run poe push "msg"` commits everything and pushes.
4. In `colab.ipynb`, set `EX` and run **Test and benchmark**. It pulls your push, runs the
   tests on the GPU, then the benchmark (speed and % of roofline).

The Colab kernel runs on Google's machine and can't see your local files, which is why
step 3 goes through GitHub. A Colab runtime shuts down when it's idle; if that happens, re-run the
setup cells.

```
uv run poe              # list tasks
uv run poe setup        # build the CPU test image (once)
uv run poe open 3       # open ex03 in VS Code / Cursor
uv run poe test 3       # ex03 tests on the CPU interpreter; no number = every exercise
uv run poe agent-help 3 # AI coach for ex03 (add `claude` or `codex` for a terminal CLI)
uv run poe push "msg"   # commit everything and push, for the Colab GPU to pull
```

## Stuck?

Each exercise has `hints.md` with three levels of hint. Open one at a time.

Or get an AI coach: `uv run poe agent-help 3`. If the Claude Code extension is installed in
your editor, it opens in the right sidebar with a coaching prompt filled in; press Enter to
start. Otherwise it starts the `claude` CLI in your terminal. Add `claude` or `codex` to pick
a terminal CLI yourself.

The coach follows `AGENTS.md` (`CLAUDE.md` points to it): it asks where you're stuck, runs your
tests, and gives hints rather than writing the kernel for you. Ask it outright if you want the
answer. Paste your benchmark output from Colab for a performance review.

`AGENTS.md` also tells agents how to write new exercises in this format, and lists ideas
for harder ones. Ask for one, e.g. "write exercise 8: Flash-Decoding".
