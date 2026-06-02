# Decision Record — DCGAN Image Generator

The four questions that make judgment portable. These are **first-draft answers** (from `SPEC.md` §1.1) — pressure-test and revise them in the recorded whiteboard session, then keep what survives.

## Situation
Generative image models look magical in a demo, but a DCGAN is **notoriously unstable** to train — mode collapse, non-convergence, brittle hyperparameters. The temptation is to cherry-pick a handful of good samples and hide the failures. Facts I have: an image dataset + random noise vectors. Facts I'm missing: any guarantee the generator learned the *distribution* rather than collapsing or memorizing.

## Decision
A DCGAN faithful to the paper — **Generator with fractional-strided (transposed) convs, Discriminator with strided convs, BatchNorm, LeakyReLU, no FC layers, Adam with tuned betas** — with a **custom training loop** (alternating G/D steps), **honest evaluation via FID** (not eyeballing), and a **fixed-noise sample grid logged every epoch.**
**Rejected:** reporting only cherry-picked best samples; chasing SOTA (diffusion) — the goal is mastering adversarial-training mechanics within a real compute budget.

## Risk
The killers: **mode collapse** and unstable training **presented as success via cherry-picking.** Mitigations: monitor D/G losses, track **FID over epochs**, log the **fixed-noise grid**, apply standard stabilizers (sensible LR/betas, label smoothing, balanced steps), and a **nearest-neighbor check** vs. training images (memorization guard). Document the failures and the fixes.
**Consciously accepted:** a modest-resolution, honestly-evaluated model over an impressive cherry-picked gallery; a bounded compute budget.

## Change
A model I can *defend* — training curves, FID trend, fixed-noise progression, and the failure modes I diagnosed — not just outputs. The skill transfers to synthetic defect-image augmentation for vision QA. The prevented loss: a portfolio piece that's secretly cherry-picked and falls apart under one good question.

## Whiteboard session
- Recording: _TBD_
- The failure mode I hit and fixed: _…_
- Why FID, and its limits: _…_
- The manufacturing transfer: _synthetic defect images to augment vision-inspection training._

---

## Engineering decisions (recorded as built)
- **Modeling:** PyTorch + torchvision; weight init ~N(0, 0.02); Adam lr=2e-4, betas=(0.5, 0.999); 32/64 px per compute budget.
- **Honest instrumentation from day one:** FID-over-epochs, fixed-noise grid per epoch, D/G loss curves, nearest-neighbor check — saved to `assets/`. A clean run with no documented failures is suspicious.
- **Stack deviation (on purpose):** PyTorch + **Gradio + Hugging Face Spaces** (free GPU) instead of the portfolio's Django/Render — the ML-idiomatic venue.
- **Artifacts:** don't commit dataset or large checkpoints; small checkpoint or a Release; state the dataset license.
