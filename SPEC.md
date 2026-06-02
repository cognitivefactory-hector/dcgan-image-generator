# DCGAN Image Generator — Design Spec

**Project 8 of the Hector Garza portfolio.** Self-contained: everything needed to start this as its own repository is in this file and its companion `PLAN.md`. You do not need any other file from the `career/` folder to build this.

- **Owner:** Hector Garza · hectorg@smartxchain.com · hector-garza.com
- **Status:** Spec — ready to build
- **Suggested repo name:** `dcgan-image-generator`
- **One-liner:** A Deep Convolutional GAN trained from scratch in PyTorch to synthesize images from noise — built to *show the training dynamics and the failure modes I fought*, not a cherry-picked gallery.

> **Demonstration project on a public image dataset (e.g., anime faces).** Synthetic generated images; no real persons; no employer data.

---

## 0. Read this first — what this project is *really* for

This is a job-search portfolio project to **demonstrate generative-modeling and adversarial-training competence in PyTorch** — but it is **not** a "look at these pretty pictures" demo. Anyone can run a DCGAN tutorial and screenshot the best outputs. The hireable signal is **what you understood and fought**: the custom training loop, the instability (mode collapse), how you *detected* it, how you *measured* quality (FID, not eyeballing), and the hyperparameter discipline.

Three deliverables of equal weight:
1. **The working app** (hosted, clickable) — generate samples from the trained generator on demand.
2. **A Decision Record** (`DECISIONS.md`) structured around the four questions below.
3. **A recorded whiteboard session** (5–8 min) defending your evaluation and showing the failure modes.

A hiring manager who opens this repo should learn you can *train an unstable model honestly* — and that the skill transfers (see §2).

---

## 1. The spine — four questions that make judgment portable

> **1 · Situation** — What's happening, the constraints, the facts you have and the facts that are *missing*.
> **2 · Decision** — The path taken and the credible options *rejected*.
> **3 · Risk** — What could go wrong, what you removed, what you *consciously accepted*.
> **4 · Change** — What's different now; the judgment tied to a real change.

### 1.1 First-draft answers for DCGAN (defend/revise on camera)

- **Situation.** Generative image models look magical in a demo, but a DCGAN is **notoriously unstable** to train — mode collapse (the generator emits a few outputs), non-convergence, brittle hyperparameters. The temptation is to cherry-pick a handful of good samples and hide the failures. Facts I have: an image dataset + random noise vectors. Facts I'm missing: any guarantee the generator learned the *distribution* rather than collapsing or memorizing.
- **Decision.** Implement a DCGAN faithful to the paper's architectural guidelines — **Generator with fractional-strided (transposed) convolutions, Discriminator with strided convolutions, BatchNorm, LeakyReLU, no fully-connected layers, Adam with tuned betas** — with a **custom training loop handling the alternating G/D gradient steps**, **honest evaluation via FID** (not eyeballing), and a **fixed-noise sample grid logged every epoch** to show progression. **Rejected:** reporting only cherry-picked best samples; chasing SOTA (diffusion) — the goal is mastering adversarial-training mechanics within a real compute budget.
- **Risk.** The killers: **mode collapse** and unstable training **presented as success via cherry-picking.** Mitigations: monitor D/G losses, track **FID over epochs**, log the **fixed-noise grid**, apply standard stabilizers (sensible LR/betas, label smoothing, balanced D/G steps), and a **nearest-neighbor check** to rule out memorization. Document the failures and what fixed them. **Consciously accepted:** a modest-resolution, honestly-evaluated model over an impressive cherry-picked gallery; a bounded compute budget.
- **Change.** A model I can *defend* — I can show the training curves, the FID trend, the fixed-noise progression, and the failure modes I diagnosed — not just outputs. Prevented loss: a portfolio piece that's secretly cherry-picked and falls apart under one good question.

---

## 2. Why this project (market fit + the transfer)

- **Generative modeling and adversarial training** are core PyTorch competencies; a from-scratch DCGAN with honest evaluation proves you understand training loops, conv arithmetic, and instability — not just `model.fit()`.
- **The skill transfers to your manufacturing narrative:** GANs/generative models are used to **synthesize defect images to augment training data for vision-based quality inspection** when real defect samples are scarce. The anime-faces dataset is a clean, public *vehicle* for the technique; the README names this transfer explicitly so the project still ladders to your domain.
- Pairs with the LSTM project to show **breadth across PyTorch** (sequence + generative/vision).

---

## 3. The staged whiteboard session (recorded deliverable)

**Format.** 5–8 min, screen + voice, defending the design against push-back. Use a strong ML-literate friend or answer the script below on camera. Preserve what survives in `DECISIONS.md`.

### 3.1 Adversarial challenge script
1. **"Anyone can run a DCGAN tutorial. What did *you* actually understand?"** *(The alternating G/D gradient steps, why transposed-vs-strided convs, why BatchNorm/LeakyReLU, the failure modes.)*
2. **"Every GAN mode-collapses. Show me yours — how did you detect and fight it?"** *(Fixed-noise grid losing diversity; FID spiking; the stabilizer that helped.)*
3. **"GANs are obsolete; diffusion won. Why DCGAN?"** *(Learning adversarial mechanics; compute/scope; honest framing.)*
4. **"You're eyeballing samples. Give me a number."** *(FID — what it measures, and its limits.)*
5. **"How do I know it isn't just memorizing training images?"** *(Nearest-neighbor check vs. training set; FID on held-out.)*
6. **"An anime generator — how does this relate to your manufacturing pivot?"** *(The transfer: synthetic defect augmentation for vision QA; honesty that the dataset is a vehicle for the technique.)*

### 3.2 What the recording must show
- The **Situation → Decision → Risk → Change** arc, in your words.
- An actual **failure mode** (mode collapse / instability) and how you diagnosed it.
- The **FID trend + fixed-noise progression**, not just final samples.
- A pointer to `DECISIONS.md`.

---

## 4. Product specification

### 4.1 Users
- **Primary:** you, demonstrating the technique.
- **Demo viewer:** a hiring manager who clicks "generate" and (more importantly) sees the training-dynamics story.

### 4.2 Core features (MVP)
1. **Generator demo:** sample random noise → generate an image grid on demand (from a committed checkpoint).
2. **Training-story panel:** the **FID-over-epochs** curve, the **fixed-noise grid across epochs**, and D/G loss curves — the honest evidence.
3. **Latent stroll (optional):** interpolate between two noise vectors to show smooth latent space (evidence of learning, not memorizing).
4. **Failure-mode note:** a short, honest writeup of a collapse/instability you hit and fixed.

### 4.3 Screens / UI
- A **Gradio** app: "Generate" button (noise → samples) + a tab showing the FID curve and fixed-noise progression + the failure-mode note.

### 4.4 Explicit non-goals (YAGNI)
- No SOTA fidelity / high resolution; 64×64 (or 32×32) is fine and honest about compute.
- No diffusion, no StyleGAN — the scope is *DCGAN done honestly*.
- No training in the hosted app — training is offline; the app does inference from a checkpoint.
- No claim of novel research; this is a competence demonstration.

---

## 5. Data (public — be honest about provenance)

- A **public image dataset** (e.g., an anime-faces dataset, or CelebA / a flowers set if preferred). State the source and license in the README.
- Standard preprocessing (resize/crop to 64×64, normalize to [-1, 1]).
- **Generated images are synthetic;** no real individuals are depicted. **No employer data.**
- Don't commit the dataset; document how to fetch it.

---

## 6. Architecture & stack

A deliberate deviation from the portfolio's Django/Render default — PyTorch modeling + an ML-idiomatic demo/host (HF Spaces gives free GPU, which helps GAN training/inference). **Recorded as a decision** in `DECISIONS.md`.

```
repo
├── src/
│   ├── models/   generator.py (ConvTranspose2d stack), discriminator.py (Conv2d stack)
│   ├── data.py   dataset + transforms (64x64, normalize [-1,1])
│   ├── train.py  custom loop: alternating D/G steps, checkpointing, fixed-noise logging
│   └── eval/     fid.py, nearest_neighbor.py
├── app/  app.py  Gradio demo (load checkpoint → generate)
├── assets/       fixed-noise grids per epoch, FID curve, loss curves (the training story)
├── notebooks/    one clean training/eval notebook
└── tests/        test_shapes.py (G/D I/O shapes), test_train_step.py (one step runs)
```

- **Modeling:** **PyTorch + torchvision** (datasets, transforms, `FrechetInceptionDistance` via torchmetrics or a standard FID impl).
- **Demo:** **Gradio**. **Host:** **Hugging Face Spaces** (free, ML-idiomatic; GPU tier helps). Linked from hector-garza.com.
- **Tooling:** Python 3.12, single `pyproject.toml`, **ruff + pytest**, GitHub Actions CI (matches portfolio conventions). Optional: log training to TensorBoard / Weights & Biases.

---

## 7. ML substance (get it right — you'll be asked)

- **Generator:** project + reshape noise `z` → stack of **`ConvTranspose2d`** (fractional-strided) + BatchNorm + ReLU; `tanh` output to [-1, 1].
- **Discriminator:** stack of **`Conv2d`** (strided) + BatchNorm + **LeakyReLU**; sigmoid (or logits with BCEWithLogits). **No fully-connected layers** (DCGAN guideline).
- **Training loop (custom):** alternating steps — update D on real+fake, update G to fool D; **Adam, lr≈2e-4, betas=(0.5, 0.999)** (DCGAN defaults); weight init ~N(0, 0.02). Optional label smoothing.
- **Evaluation:** **FID** over epochs (the headline metric); D/G loss curves; a **fixed-noise grid** logged each epoch (diversity = no collapse); **nearest-neighbor** check vs. training images (rule out memorization).
- **Stability playbook:** what to try when D overpowers G (balance steps, lr, label smoothing, noise) — and document what actually helped.

---

## 8. Definition of Done

- [ ] **App** on a public URL (HF Space): click generate → a grid of synthetic samples; a tab shows the FID curve + fixed-noise progression + loss curves.
- [ ] **Honesty visible:** the training story (FID trend, progression) and a documented failure mode are shown — not just final cherry-picked samples.
- [ ] **`README.md`** — what/why, dataset provenance + license, the manufacturing-transfer note, one-command local train/run, links to live demo + `DECISIONS.md` + whiteboard.
- [ ] **`DECISIONS.md`** — the §1 template completed (rejected cherry-picking; accepted modest-resolution honest eval + bounded compute).
- [ ] **Whiteboard recording** linked from README and on hector-garza.com.
- [ ] **Tests pass:** G/D I/O shapes and a single train step (see `PLAN.md`).

---

## 9. Hosting / deployment
- **Hugging Face Spaces** (Gradio SDK); commit a trained checkpoint (or pull from a Release). The free GPU tier helps both training and snappy inference.
- Link from hector-garza.com; optionally embed the Space.
- Training is offline (a notebook/script + a GPU — Colab/Kaggle/HF are fine); the Space only does inference.

---

## 10. Repo bootstrap (how to start this as its own repo)

```bash
mkdir dcgan-image-generator && cd dcgan-image-generator
cp /path/to/08-dcgan-image-generator/SPEC.md .
cp /path/to/08-dcgan-image-generator/PLAN.md .
# seed: README.md, DECISIONS.md (template below), .gitignore (python + data/ + *.pt + runs/ + wandb/), LICENSE (MIT)

git init && git add -A && git commit -m "chore: scaffold dcgan-image-generator (spec + plan)"
git branch -M main
gh repo create cognitivefactory-hector/dcgan-image-generator --public --source=. --remote=origin --push
```

> PUBLIC repo. **Don't commit the dataset or large checkpoints** (gitignored; use a small checkpoint or a Release). No API keys needed. State the dataset license.

### `DECISIONS.md` starter
```markdown
# Decision Record — DCGAN Image Generator

## Situation
<GANs unstable (mode collapse, non-convergence); cherry-pick temptation; missing: proof it learned the distribution>

## Decision
<paper-faithful DCGAN (transposed/strided convs, BN, LeakyReLU, Adam betas), custom alternating loop, FID + fixed-noise grid; REJECTED cherry-picking, REJECTED chasing diffusion/SOTA>

## Risk
<mode collapse + cherry-picking-as-success; FID-over-epochs + fixed-noise grid + nearest-neighbor + stabilizers; ACCEPTED modest resolution + bounded compute>

## Change
<a defensible model: training curves, FID trend, failure modes shown; transfers to synthetic defect augmentation for vision QA; prevented a cherry-picked piece that collapses under questioning>

## Whiteboard session
- Recording: <link>
- The failure mode I hit and fixed: <…>
- Why FID, and its limits: <…>
- The manufacturing transfer: <…>
```

---

## 11. Open questions to resolve in the plan
- Dataset choice (anime faces vs. CelebA vs. flowers) + license.
- Resolution: 32×32 (cheap, fast) vs. 64×64 (nicer) given compute budget.
- FID implementation (torchmetrics `FrechetInceptionDistance` vs. a standard script).
- Where to train (Colab/Kaggle/HF GPU) and how the checkpoint reaches the Space.
