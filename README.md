# DCGAN Image Generator

A Deep Convolutional GAN trained from scratch in PyTorch to synthesize images from noise — built to *show the training dynamics and the failure modes I fought*, not a cherry-picked gallery.

> **Status:** scaffolded (spec + plan in place). Build follows `PLAN.md` (M0 → M7).
> **Demonstration project on a public image dataset.** Generated images are synthetic; no real persons; no employer data.

Part of [hector-garza.com](https://hector-garza.com)'s portfolio. One of three equal deliverables: the app, a **Decision Record** ([`DECISIONS.md`](./DECISIONS.md)), and a recorded whiteboard session. A working model no longer proves competence — the judgment behind it does. See [`SPEC.md`](./SPEC.md) §0.

## What it does
- Generates images from random noise via a from-scratch **DCGAN** (Generator: fractional-strided convs; Discriminator: strided convs; BatchNorm, LeakyReLU, no FC layers; Adam betas (0.5, 0.999)).
- A **custom training loop** with alternating D/G gradient steps.
- **Honest evaluation:** FID over epochs, a fixed-noise sample grid across epochs (diversity = no mode collapse), loss curves, and a nearest-neighbor check (memorization guard).
- A documented **failure mode** (mode collapse / instability) and the fix.

## Why it's here (and the transfer)
Proves generative-modeling + adversarial-training competence in PyTorch. The skill **transfers to manufacturing**: GANs synthesize **defect images to augment training data for vision-based quality inspection** when real defect samples are scarce — the anime-faces dataset is just a clean public *vehicle* for the technique.

## Tech stack
- **Modeling:** PyTorch + torchvision · FID via torchmetrics
- **Demo:** Gradio · **Host:** Hugging Face Spaces (free GPU helps)
- **Tooling:** Python 3.12 · `pyproject.toml` · ruff + pytest · GitHub Actions CI

> Stack note: a PyTorch/ML project, so it deviates from the portfolio's Django/Render default on purpose — **Gradio + Hugging Face Spaces** is the ML-idiomatic venue (and gives free GPU). The [hector-garza.com](https://hector-garza.com) hub links out to the Space.

## Links (filled in as the build progresses)
- 🔗 Live demo (HF Space): _TBD_
- 🧠 Decision record: [`DECISIONS.md`](./DECISIONS.md)
- 🎥 Whiteboard walkthrough: _TBD_

## Build
See [`PLAN.md`](./PLAN.md) — M0 (scaffold) → M7. Architecture + a single clean train step come first; then training with **honest instrumentation from day one** (FID, fixed-noise grids, loss curves) so failures are visible, not hidden.
