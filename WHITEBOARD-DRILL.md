# Whiteboard Drill — DCGAN Image Generator (design-stage)

> Rehearsal for the recorded whiteboard session. **The push** is me playing tough reviewer; **Defense** is the position that survives; **⚠ Your move** is what only you can answer once you've built/measured it. Fold the survivors into `DECISIONS.md`, then record.
> Scope: design-stage. Re-run after **M4** (the stability pass) with your real FID curve + a documented failure mode.

## Q1 — "Anyone can run a DCGAN tutorial. What did *you* actually understand?"
**The push:** You copied a notebook.
**Defense (survives):** I can walk the **alternating G/D gradient steps** (update D on real+fake, then update G to fool the updated D), and *why* the architecture is what it is: transposed convs to upsample noise→image, strided convs to downsample image→logit, BatchNorm to stabilize, LeakyReLU in D to keep gradients alive, no FC layers, Adam betas (0.5, 0.999) for GAN stability. Understanding *why each choice* exists is the difference from a copied tutorial.
**⚠ Your move:** Be ready to explain one architecture choice from memory at the board.

## Q2 (the killer) — "Every GAN mode-collapses. Show me yours — how did you detect and fight it?"
**The push:** If you didn't hit it, you didn't really train one.
**Defense (survives):** I instrumented for it from day one: a **fixed-noise grid every epoch** (if outputs lose diversity, that's collapse, visible immediately), **FID over epochs** (spikes/stalls flag it), and D/G loss curves (D→0 / G exploding). When it appeared I applied the playbook (balance D/G steps, tune LR, label smoothing, input noise) and recorded what actually helped. A clean run with no documented failure would be the suspicious result.
**⚠ Your move:** Capture an actual collapse (or near-collapse) + the fix — this is the centerpiece of the recording. Don't hide it; *feature* it.

## Q3 — "GANs are obsolete; diffusion won. Why DCGAN?"
**The push:** You're learning a dead technique.
**Defense (survives):** The goal is mastering **adversarial-training mechanics and training-instability diagnosis** within a real compute budget — transferable skills, and DCGAN is the cleanest vehicle for them. Diffusion is the SOTA destination, not the place to *learn* the failure modes. I'm explicit that this is a competence demonstration, not a claim of frontier novelty.

## Q4 — "You're eyeballing samples. Give me a number."
**The push:** 'Looks good' isn't evaluation.
**Defense (survives):** **FID** (Fréchet Inception Distance) tracked over epochs is the headline metric — it compares the statistics of generated vs. real images. I also know its limits: FID can be gamed, is sensitive to sample size and the Inception backbone, and doesn't measure diversity directly — which is why I pair it with the fixed-noise grid.
**⚠ Your move:** Have the FID-over-epochs curve in `assets/` and state your final FID with the sample size used.

## Q5 — "How do I know it isn't just memorizing training images?"
**The push:** Your 'samples' might be near-copies.
**Defense (survives):** A **nearest-neighbor check** — for generated samples, retrieve the closest training images; if generations are just near-duplicates, that's memorization, not generation. FID on a held-out split supports it. Latent interpolation (smooth transitions between noise vectors) is additional evidence it learned a manifold, not a lookup table.
**⚠ Your move:** Run and show the nearest-neighbor panel.

## Q6 — "An anime generator — how does this relate to your manufacturing pivot?"
**The push:** This is off-brand for an AI/manufacturing role.
**Defense (survives):** The dataset is a clean public *vehicle*; the **technique transfers directly**: GANs synthesize **defect images to augment training data for vision-based quality inspection** when real defect samples are rare — a genuine manufacturing problem I understand. I'm honest that the anime faces are for learning the method; the README states the transfer explicitly.
**⚠ Your move:** Be ready to say, in one sentence, how you'd apply this to synthetic defect augmentation on a line you've run.

## Verdict — SDRC after the drill
- **Holds:** honest-instrumentation-over-cherry-picking; the manufacturing transfer; bounded-compute scope.
- **Sharpen:** lead with **Q2** — *feature* the failure mode you fought (it's the credibility); have the FID curve + nearest-neighbor panel ready (Q4, Q5); rehearse the one-line manufacturing transfer (Q6).
- **Land this line in the room:** *"I trained an unstable model honestly — here's the FID trend, the fixed-noise progression, and the mode collapse I diagnosed and fixed — not a cherry-picked gallery."*
