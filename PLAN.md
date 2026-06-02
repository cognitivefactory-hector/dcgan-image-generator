# DCGAN Image Generator — Implementation Plan

Companion to `SPEC.md`. The build sequence: milestones, tasks, acceptance criteria, definition of done. Self-contained.

- **Repo:** `dcgan-image-generator` (public, under `cognitivefactory-hector`)
- **Approach:** get the **architecture + a single clean train step** correct first (shapes, conv arithmetic, the alternating G/D update), then train with **honest instrumentation from day one** (FID, fixed-noise grid, loss curves) so failures are visible — not discovered after cherry-picking.

> **Demonstration project on a public dataset.** Generated images are synthetic. Keep dataset license + provenance in the README.

---

## The spine (carry through every milestone)
Keep `DECISIONS.md` open: **Situation · Decision** (incl. rejected cherry-picking / SOTA-chasing) **· Risk** (incl. accepted modest-resolution + bounded compute) **· Change**. The hardest stance (show the failure modes, not just pretty samples) is the whiteboard centerpiece — `SPEC.md` §3.

## Prerequisites
- Python 3.12, git, `gh` authenticated. A GPU for training (Colab / Kaggle / HF). A Hugging Face account for the Space. No API keys.

---

## Milestones

### M0 — Repo scaffold *(½ day)*
- [ ] Folder + `SPEC.md` + `PLAN.md`; `README.md` (stub + dataset license note + manufacturing-transfer note), `DECISIONS.md` (paste template from `SPEC.md` §10), `.gitignore` (Python + `data/` + `*.pt` + `runs/` + `wandb/`), `LICENSE` (MIT).
- [ ] `pyproject.toml` (deps + ruff + pytest); pin `torch`, `torchvision`, `torchmetrics`, `gradio`.
- [ ] GitHub Actions: ruff + pytest.
- [ ] `gh repo create … --public --push`.
- **Acceptance:** `pytest` runs; CI green; repo on GitHub.

### M1 — Models + a single train step (TDD) *(1–2 days)* — **get the mechanics right**
- [ ] `src/models/generator.py`: project/reshape `z` → `ConvTranspose2d` + BatchNorm + ReLU stack → `tanh`.
- [ ] `src/models/discriminator.py`: `Conv2d` (strided) + BatchNorm + LeakyReLU stack → logits. **No FC layers.**
- [ ] Weight init ~N(0, 0.02); Adam lr=2e-4, betas=(0.5, 0.999).
- [ ] **Tests first:** `test_shapes.py` — G(noise) → (B,3,H,W); D(image) → (B,1). `test_train_step.py` — one D step + one G step run and produce finite losses.
- **Acceptance:** `pytest` green; shapes and a single alternating step are correct.

### M2 — Data pipeline *(½ day)*
- [ ] `src/data.py`: dataset loader + transforms (resize/crop to 32 or 64, normalize [-1,1]); document fetch (don't commit data).
- [ ] Test: a batch has the expected shape/range.
- **Acceptance:** a real batch loads and normalizes correctly.

### M3 — Training loop with honest instrumentation *(2–3 days)* — **the heart**
- [ ] `src/train.py`: custom alternating D/G loop; checkpointing; **log a fixed-noise grid every epoch**; log D/G losses.
- [ ] `src/eval/fid.py`: FID (torchmetrics `FrechetInceptionDistance`) computed over epochs.
- [ ] `src/eval/nearest_neighbor.py`: nearest-neighbor check vs. training set (memorization guard).
- [ ] Train to a reasonable result within the compute budget; **save the FID curve, loss curves, and fixed-noise grids to `assets/`.**
- **Acceptance:** training runs end-to-end; FID trends down; the fixed-noise grid shows increasing diversity; the artifacts are saved.

### M4 — Stability pass (document the fight) *(1 day)*
- [ ] If/when mode collapse or instability appears, apply the playbook (balance D/G, lr, label smoothing, noise) and **record what helped** in `DECISIONS.md`.
- **Acceptance:** at least one documented failure mode + the fix (this is the honesty signal — a clean run with no notes is suspicious).

### M5 — Gradio app *(1 day)*
- [ ] `app/app.py`: "Generate" (noise → sample grid from checkpoint) + a tab with the FID curve, fixed-noise progression, loss curves, and the failure-mode note. Optional latent-stroll.
- **Acceptance:** app runs locally; generate works; the training story is visible.

### M6 — Deploy to Hugging Face Spaces + README *(½–1 day)*
- [ ] Push a Gradio **HF Space**; commit a small checkpoint or pull from a Release.
- [ ] `README.md`: what/why, dataset provenance + license, manufacturing-transfer note, one-command run, links to Space + `DECISIONS.md` + whiteboard.
- **Acceptance:** public Space URL generates samples; README matches reality.

### M7 — Decision Record + Whiteboard session *(½ day)* — **the differentiator, don't skip**
- [ ] Complete `DECISIONS.md`; record the 5–8 min session using `SPEC.md` §3.1 — center #2 (show your mode collapse) and #4 (give me a number / FID).
- [ ] Embed/link on hector-garza.com.
- **Acceptance:** a stranger can read `DECISIONS.md` + watch the video and explain how you knew the model worked (and didn't just memorize).

---

## Testing strategy
- **Tests cover the mechanics** (G/D I/O shapes, a single train step runs with finite losses) — not sample quality (that's FID + the recording).
- Keep FID/training out of CI (too heavy); CI runs ruff + the shape/step tests.

## Risk register (project execution)
| Risk | Mitigation |
|---|---|
| Mode collapse | Detected via fixed-noise grid + FID; stability playbook; documented. |
| Cherry-picking passed off as success | FID-over-epochs + progression grids are mandatory artifacts; the recording shows them. |
| Memorization mistaken for generation | Nearest-neighbor check vs. training set. |
| Compute blows up | Cap resolution (32/64) + epochs; train on free GPU; bounded budget stated. |
| Huge checkpoints/datasets in git | Gitignored; small checkpoint or Release; document dataset fetch. |
| Skipping M7 | M7 *is* the portfolio — "I trained an unstable model honestly." |

## Definition of Done
See `SPEC.md` §8 — app + decision record + whiteboard, all linked; the training story (FID + progression) and a documented failure mode are visible; shape/step tests pass in CI.
