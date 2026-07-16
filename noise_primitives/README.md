# Noise primitives (thesis — Phase III channel layer)

This module is the **channel-noise layer from my PhD thesis** — the redesigned,extended successor to the disturbance implementations used in the papers. It provided the corruption machinery for calibrating the reliability-aware **NoisyGraph-RC** model: the RL coordination policy is trained under clean communication, and these primitives then inject a configurable mixture of channel impairments while only the reliability components (the autoencoder and a subset of GAT reliability parameters) are fine-tuned. The NoisyGraph models themselves are not included here — they will be released with their forthcoming papers.

## What each piece does

| Code | Impairment | Thesis validation scenario |
|---|---|---|
| `generate_noise(...)` | Additive Gaussian/uniform noise, with per-message apply probability (and legacy variant kept for paper parity) | 1A system-wide, 1B localized, 1C with healing |
| `snr_noise(...)` | SNR-based signal degradation (dB-parameterised), full or partial link application | 2 distance-dependent fading, 3 partial link fading |
| `drop_messages(...)` | Whole-message packet loss + partial message corruption | 4 |
| `GilbertElliotChannel` | Two-state bursty channel (good/bad states with transition probabilities; drop or noise in the bad state) | 5 Gilbert–Elliott bursty drops |
| `comm_delay_n_hops(...)` | Per-hop message delays with a bounded delay buffer | 6 (with reordering) |
| `jumble_messages_norm(...)`, `jumble_messages_realistic(...)` | Message reordering / collision mixing / stale-current overlap | 7 collision mixing, 8 overlap delay |
| `sample_noise_config(...)`, `apply_noise(...)`, `apply_calibration_noise(...)` | The **calibration curriculum sampler**: draws clean batches, single primitives, or compound configurations per batch and applies them to the communication tensor | — (training-time) |

## The validation schedule (thesis)

Each scenario is evaluated at three intensities (Light → Moderate → Severe):

| # | Type | Configuration | Parameters (L → M → S) |
|---|---|---|---|
| 0 | Clean | no noise | — |
| 1A | Additive | system-wide Gaussian | noise_level 0.2 → 0.5 → 0.8; apply_prob 1.0 |
| 1B | Additive | localized Gaussian | noise_level 0.2 → 0.5 → 0.8; apply_prob 0.4 |
| 1C | Additive | Gaussian with healing | noise_level 0.5 → 0.8; apply_prob 0.4; healing on |
| 2 | SNR | distance-dependent fading | 20 → 10 → 5 dB; apply_prob 1.0 |
| 3 | SNR | partial link fading | 20 → 10 → 5 dB; apply_prob 0.3 → 0.5 |
| 4 | Drops | packet loss + partial corruption | whole 0.05 → 0.15 → 0.30; partial 0.05 → 0.10 → 0.20 |
| 5 | Burst | Gilbert–Elliott bursty drops | p(G→B) 0.03 → 0.06 → 0.12; p(B→G) 0.2 → 0.3 → 0.5 |
| 6 | Delay | delay + local reordering | max_hop_delay 1 → 2 → 3; delay_prob 0.05 → 0.15 → 0.3; jumble_prob 0.01 → 0.03 → 0.05 |
| 7 | Jumbling | collision mixing | jumble_prob 0.02 → 0.05 → 0.1; mix_frac 0.2 → 0.35 → 0.5 |
| 8 | Jumbling | overlap delay (stale + current) | jumble_prob 0.01 → 0.03 → 0.05; overlap_α 0.6 → 0.75 → 0.9 |
| 10 | Combined | combined stress test | GE (p(G→B) 0.08, p(B→G) 0.25, bad-state noise σ 1.0) + SNR 5 dB + whole-drop 0.2 |

## The calibration curriculum (thesis)

`apply_calibration_noise` implements the training-time mixture: **20% clean batches**, **55% single primitives** (relative weights: additive Gaussian 35%, message drops 25%, SNR 10%, jumbling 10%, hop delays 10%, Gilbert–Elliott 10%, each with randomised parameters within published ranges), and **25% compound configurations** (delay + local reordering, or additive + partial drops). The
design intent: expose the reliability pathway to heterogeneous corruption without ever destabilising the frozen coordination policy.

Full parameter ranges, rationale and results are in the thesis (*Channel-Robust Emergent Communication: Mitigating Complex Network Distortions via Reliability-Aware Graph Attention*, Trinity College Dublin, 2026).
