# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

**This repo is a documentation-only scaffold.** There is no source code, no `pyproject.toml`, and no test suite yet — only `SPEC.md`, `PLAN.md`, `DECISIONS.md`, `WHITEBOARD-DRILL.md`, `README.md`. The build has not started past M0. Do not assume code exists; check before referencing files like `src/train.py`.

`SPEC.md` and `PLAN.md` are the source of truth and are deliberately self-contained — everything needed to build lives in them. Read both before writing code.

## What this project actually is

A from-scratch DCGAN in PyTorch — but the working model is **one of three equal deliverables**, not the goal:

1. The hosted Gradio app (generate samples from a checkpoint).
2. `DECISIONS.md` — a Situation · Decision · Risk · Change record.
3. A recorded whiteboard session defending the evaluation.

The hireable signal is **honest training**, not pretty pictures. The single most important constraint, which shapes every implementation choice: **never cherry-pick.** Instrumentation (FID over epochs, fixed-noise grid per epoch, D/G loss curves, nearest-neighbor memorization check) is mandatory **from day one** so failures are visible rather than hidden. A clean training run with no documented failure mode is treated as *suspicious*, not success — surfacing and documenting a mode-collapse/instability episode and its fix is a deliverable (M4), not a setback. When in doubt, favor honest/measurable over impressive.

The anime-faces dataset is just a public *vehicle*; the stated transfer is synthetic defect-image augmentation for vision-based manufacturing QA. Keep that framing intact in any docs you touch.

## Build sequence (PLAN.md, M0→M7)

Work follows milestones in order. The ordering is itself a decision: get **architecture + a single clean train step correct first** (shapes, conv arithmetic, the alternating G/D update — built TDD), *then* train with instrumentation already wired in.

- **M0** scaffold — `pyproject.toml` (pin torch, torchvision, torchmetrics, gradio; ruff + pytest), GitHub Actions (ruff + pytest).
- **M1** models + single train step, **tests first** — `src/models/generator.py`, `src/models/discriminator.py`.
- **M2** `src/data.py` — loader + transforms (resize/crop to 32 or 64, normalize to [-1, 1]).
- **M3** `src/train.py` (custom alternating loop, checkpointing, per-epoch fixed-noise grid), `src/eval/fid.py`, `src/eval/nearest_neighbor.py`. Save FID/loss curves + grids to `assets/`.
- **M4** stability pass — document a real failure mode and its fix in `DECISIONS.md`.
- **M5** `app/app.py` — Gradio (inference only; no training in the app).
- **M6** deploy to Hugging Face Spaces + finalize README.
- **M7** complete `DECISIONS.md` + record the whiteboard session. **Do not skip — this is the portfolio.**

## Commands

None exist yet — they activate once M0 lands `pyproject.toml`. The intended toolchain (per PLAN.md / SPEC.md §6) is **Python 3.12 · ruff · pytest**:

```bash
ruff check .          # lint
pytest                # all tests
pytest tests/test_shapes.py::<name>   # single test
```

CI runs **only** ruff + the shape/step tests. **Keep FID and training out of CI** — they're too heavy. Tests cover *mechanics* (G/D I/O shapes, one alternating step produces finite losses), never sample quality (that's FID + the recording).

## ML invariants (you will be questioned on these — get them right)

These come straight from the DCGAN paper and SPEC.md §7; deviating silently undermines the whole project:

- **Generator:** project/reshape `z` → stack of `ConvTranspose2d` + BatchNorm + ReLU → `tanh` output in [-1, 1].
- **Discriminator:** strided `Conv2d` + BatchNorm + LeakyReLU → logits (use BCEWithLogits). **No fully-connected layers.**
- Weight init ~ N(0, 0.02); **Adam, lr=2e-4, betas=(0.5, 0.999)**.
- Resolution 32×32 or 64×64 only — modest resolution is a *consciously accepted* trade for honest eval within a bounded compute budget. No SOTA/diffusion/StyleGAN scope creep (explicit non-goal).
- Training is **offline** (Colab/Kaggle/HF GPU); the Space does inference from a committed small checkpoint.

## Repo conventions

- **Never commit** the dataset or large checkpoints — gitignored (`data/`, `*.pt`, `*.pth`, `runs/`, `wandb/`). Document how to fetch data; ship a small checkpoint or a Release. State the dataset license in the README.
- This is a **public** repo. No employer data; generated images are synthetic (no real persons).
- When you make a build decision worth defending (or revise one), record it in `DECISIONS.md` under "Engineering decisions" — that file is a deliverable, kept current as built.
