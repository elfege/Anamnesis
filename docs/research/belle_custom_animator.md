# Custom Belle Animator — Research Notes

**Status:** parked idea. Not for immediate implementation. Revisit after d² ships + a CUDA training GPU is available.

## Motivation

SadTalker and Hallo2 are generic animators retrofitted to a reference image. They don't "know" Belle. Fine-tuning on a Belle-specific dataset would:
- Fix identity drift across frames
- Expand expression range beyond whatever the pretrained model learned
- Let Belle feel *hers* rather than a Disney-default-lipsync on a still.

---

## Three paths (from cheap to ambitious)

### Path A — Fine-tune an existing animator on Belle (recommended first step)
- Pre-trained: Hallo2 or LivePortrait weights
- Dataset: 100-500 reference clips of Belle (generated or face-swapped)
- Train: LoRA adapter on Belle's identity + expression range
- Result: Hallo2 engine + Belle-specific head
- Compute: ~10-20h on A100 at $0.69/hr = ~$14-20
- Effort: 1-2 weeks

### Path B — Motion-bank / retrieval-augmented animation (novel, research-worthy)
- Generate library: videos of Belle making each phoneme × expression × head-pose combo
- Encode motion via FLAME parameters, 3DMM coefficients, or latent vectors
- Store in MongoDB or FAISS
- At inference: audio → phoneme sequence → retrieve nearest motion tokens → blend → render onto Belle still
- No neural training — just retrieval + blending (small transformer to learn blending coefficients)
- Could be a paper if results are strong
- Effort: 2-4 weeks after Path A

### Path C — From-scratch diffusion animator (not recommended)
- 10k-100k clips needed
- ~200-500 A100-hours at $2/hr = $400-1000 compute
- Likely 80-90% as good as Hallo3 with your identity baked in
- ROI is terrible; Hallo/EMO represent years of team-scale research
- Skip.

---

## Dataset generation — the hard part

| Source | Identity consistency | Expression richness | Phoneme coverage | Cost |
|---|---|---|---|---|
| **Sora 2** | ❌ identity drifts across generations | high | manual prompting, tedious | $$ (token-expensive) |
| **InVideo** | — (wrong tool; templates, not generation) | — | — | — |
| **HeyGen / D-ID (commercial TTS→avatar)** | ✓ very consistent | medium (preset expressions) | ✓ complete via Harvard-sentences | ~$30/mo |
| **Real actor + face-swap (InsightFace / ROOP)** | ✓ (swap preserves identity) | ✓ high (human actor) | ✓ complete | $100-200 one-time (Fiverr actor) |
| **Wav2Lip + LivePortrait chain** | medium | limited | driven by source video | $0 (tool chain) |

**Likely best balance for real data:** hire a Fiverr actor for 30 min of expressive speech reading a phonetically-balanced script, face-swap Belle onto the actor, use as training set.

---

## Phased execution (if we pursue this)

1. **Phase 0 — Is it worth it?** Fine-tune Hallo2 on 20-30 Belle clips. See if the quality leap over SadTalker justifies more work. 1 week, ~$15.
2. **Phase 1 — Motion bank build**: Record actor → face-swap → encode 200 phoneme×expression combinations. 2 weeks, ~$200 actor + $20 compute.
3. **Phase 2 — Retrieval/blender**: Train small transformer on (audio features + mood) → motion tokens. Research-y bit. 2 weeks, ~$50 compute.
4. **Phase 3 — Compare to fine-tuned Hallo2.** If motion-bank wins, paper it. If not, default to Path A.

---

## What this needs to exist first

- Belle identity is stable across time (cloned voice ✓, reference image ✓)
- A CUDA GPU with ≥16 GB VRAM for training (current 6 GB server GTX 1660 insufficient)
- d² work is far enough along that it doesn't compete for attention
- Basic SadTalker/Hallo2 baseline is running (so we have something to compare against)

## Adjacent directions worth tracking

- **EMO 2** (Alibaba, if released) — likely to leapfrog Hallo3
- **X-Portrait** (ByteDance) — video-driven with audio adaptation
- **LatentSync** — lip sync on video, could be combined with motion-bank
- **Gaussian avatars** (DreamGaussian4D class) — volumetric, real-time, may obsolete 2D approaches within 1-2 years

---

*Captured: 2026-04-22. Author: office-genesis (avatar instance). Approved by: Elfege.*
