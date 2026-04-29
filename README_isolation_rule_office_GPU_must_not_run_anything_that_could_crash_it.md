# Isolation Rule — `office` GPU Must Not Run Anything That Could Crash It

**Status: ACTIVE.** Authored 2026-04-29 by user (Elfège), enforced via stop+disable of the relevant containers/services on `office` (192.168.10.110).

---

## The rule

The `office` machine has an **AMD Radeon RX 6800** (16 GB VRAM, ROCm).
Through repeated and reproducible failures it has demonstrated that the
ROCm stack on this card panics the kernel under VRAM pressure, taking the
whole machine down with it. Specifically:

- 2026-04-21 — three separate hard panics in one day during avatar
  SadTalker + XTTS work (MSG-116). Recovered with refactor — but the
  underlying instability did not go away.
- 2026-04-25 23:20 EDT — full kernel panic when Ollama tried to load a
  new model on top of an already-busy GPU. AMD memory manager log:
  `amdgpu: Freeing queue vital buffer ... queue evicted` ×5 followed by
  total system death.

**Until a CUDA replacement card is installed, no service on office may
load a model into VRAM in any unattended / auto-restarting fashion.**

## What's currently allowed

- Desktop graphics (the user's logged-in GNOME session)
- Manual `ollama` invocations the user starts and watches
- Anything that does not touch `/dev/dri/*` for compute purposes

## What's currently FORBIDDEN

- `restart: unless-stopped` containers that load models
- Systemd units that auto-start GPU services on boot
- The `avatar-worker-office` container (XTTS + SadTalker)
- The `anamnesis-trainer-office` container (Qwen QLoRA)
- Any background script that calls `ollama` or otherwise pins VRAM

## What was done to enforce it (2026-04-29)

```
ssh office
docker stop avatar-worker-office anamnesis-trainer-office
sudo systemctl disable anamnesis-trainer.service
```

Result: containers in `Exited (0)` state, systemd unit removed from
`multi-user.target.wants/`. Office now runs only desktop + ssh.

The orchestrator's failover chain (`AVATAR_WORKER_URL_2`,
`NANOGPT_URLS`) still lists office, but office workers won't respond,
so requests automatically fall through to `server` (CUDA, stable).

## When the rule lifts

- A CUDA card lands in office (e.g. RTX 3090 / 4090 / 5090) and the
  RX 6800 is removed or repurposed
- ROCm itself ships a fix that survives an evening of avatar work
  without panicking (very unlikely on this generation of hardware)

When either of those happens: amend this file to "ARCHIVED" and re-enable
the containers + systemd unit by reversing the commands above.

## Related

- `MSG-116` (intercom) — initial avatar SadTalker crash investigation
- Crash log evidence: 2026-04-25 23:20 EDT in
  `journalctl --boot=-1` on office, `amdgpu: Freeing queue vital buffer
  ... queue evicted` lines immediately preceding the panic.
- Bitter-Lesson reference doc (`docs/bitter_lesson/`) discusses why we
  defaulted to NVIDIA/CUDA for the δ² benchmarks — same reasoning.

## For future Claude instances reading this

Before doing **anything** that loads a model on office's GPU — even a
seemingly small Ollama call — check this file. If it still says
"ACTIVE", do not. Use `server` (CUDA, GTX 1660 SUPER 6GB) or RunPod
instead.

Treat this as a hard rule. The cost of getting it wrong is "user's
machine reboots, potentially loses unsaved work, definitely loses the
GPU work in progress."
