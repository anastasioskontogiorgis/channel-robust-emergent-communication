#Channel-Robust Emergent Communication — CAV benchmark framework

*The experimental framework behind two publications on how state-of-the-art emergent-communication models for connected autonomous vehicles behave when communication stops being perfect: additive noise, message loss, jumbling and delays, across three difficulty levels of the standard unsignalised-intersection benchmark.*

**Papers**

- **CAV Unsignalised Intersection Crossing with Unreliable Emergent   Communication: A Comprehensive Analysis under Diverse Disruptions.** *European Journal of Artificial Intelligence*, ATT 2024 Special Issue — in final review, DOI to follow. *(The full study this framework implements: five models × four disturbance families × three difficulties.)*
- **CAV Unsignalised Intersection Crossing with Unreliable Emergent Communication.** *ATT'24 Workshop (Agents in Traffic and Transportation)* —   [open access, CEUR-WS](https://ceur-ws.org/Vol-3813/5.pdf). *(The initial study: four models, core disturbances, easy difficulty.)*

## Beyond the papers: the thesis noise layer

This repository also ships `noise_primitives/` — the **redesigned and extended channel-noise models from my PhD thesis**, a significant step beyond the
disturbance implementations used in the papers above. Where the paper-era code injects independent, per-round disturbances, the thesis layer models channels with memory, physics-grounded degradation, and compound corruption:

| Thesis primitive | What it models | Paper-era counterpart |
|---|---|---|
| Additive noise with apply-probability & healing | Localized / recoverable corruption, not just global σ | Global Gaussian/uniform |
| SNR-based fading (dB-parameterised) | Distance-dependent and partial-link signal degradation | — |
| Gilbert–Elliott channel | **Bursty** loss/noise with temporal memory (good/bad states) | Independent drops |
| Per-hop delay buffers | Multi-hop message staleness with bounded delay | Single-step delay |
| Realistic jumbling (collision mixing, stale–current overlap) | Physically motivated message mixing | Random permutation |
| Calibration curriculum sampler | Randomised single + compound corruption mixtures for training-time exposure | — |

These primitives powered the calibration of the reliability-aware **NoisyGraph-RC** model; the NoisyGraph models themselves will be released alongside their forthcoming papers. Full function-to-experiment mapping and the thesis validation/calibration schedules: [`noise_primitives/README.md`](noise_primitives/README.md). See [my research page](#) for the thesis story and results.

## What's here

| Path | What it is |
|---|---|
| `run_baselines.py` | The framework entry point: model selection, training, testing, noise injection, checkpointing, stats export — everything is a CLI flag. |
| `commNet.py`, `tar_comm.py`, `ga_comm.py`, `magic.py`, `models.py`, `gnn_layers.py` | The five evaluated models: CommNet, IC3Net, TarMAC(-IC3Net), GA-Comm, MAGIC. |
| `trainer.py`, `multi_processing.py`, `utils.py`, ... | RL training loop, parallel rollouts, and the paper-era disturbance implementations. |
| `ic3net-envs/` | The Traffic Junction environment (CommNet-lineage benchmark, via the IC3Net release — MIT licensed, see below). |
| `noise_primitives/` | Thesis channel-noise models + calibration sampler, with their own README. |
| `scripts/` | Runbooks: `train.sh`, `test_clean.sh`, `test_noise.sh` (single disturbance), `test_noise_sweep.sh` (full published schedule). |
| `Dockerfile` | The exact environment (Python 3.8 / torch 1.13.1) — the code runs exactly as it did for the papers. |

## Quickstart

The pinned stack is (`torch 1.13.1`, `gym 0.18.0`,
Python 3.8): the environment the published results were produced, it is containerised rather than upgraded for reproducibility:

```bash
docker build -t crec .
docker run -it --rm -v "$PWD/saved:/work/saved" -v "$PWD/data:/work/data" \
           -v "$PWD/logs:/work/logs" crec
```

Inside the container (or any native Python 3.8 environment with
`pip install -r requirements.txt && pip install -e ./ic3net-envs`):

```bash
# 1. Train (always on clean communication — the framework enforces it)
./scripts/train.sh gacomm easy            # MODEL: commnet|ic3net|tarcomm|gacomm|magic
                                          # Difficulty: easy|medium|hard

# 2. Evaluate on clean communication
./scripts/test_clean.sh gacomm easy

# 3. Evaluate under ONE disturbance of choice
./scripts/test_noise.sh gacomm easy gaussian 0.5
./scripts/test_noise.sh gacomm easy drops 0.3 0.4
./scripts/test_noise.sh gacomm easy delay 0.7 2

# 4. Or run the FULL published disturbance schedule in one command
./scripts/test_noise_sweep.sh gacomm easy
```

**Outputs.** The framework exports stats by default to `data/`, auto-naming the file from the run configuration (e.g. `gacomm_test_easy_gaussian0.5.txt`); the runbooks additionally tee console output to `logs/`. Both folders (and `saved/`) are created by the scripts — the framework itself expects `data/` to exist.

**Checkpoints.** Training appends `saved/traffic_junction/<model>/runN/model.pt`. Training each model in **easy → medium → hard order** reproduces the paper's convention `run1 = easy`, `run2 = medium`, `run3 = hard`, which the test scripts assume; override with the `RUN` argument/variable if your runs are
ordered differently.

## Training & testing settings (from the paper)

| Mode | Epochs | Difficulty | Add rate min–max | Agents | Curriculum |
|---|---|---|---|---|---|
| Train | 2000 | easy | 0.1 – 0.3 | 5 | 250 → 1250 |
| Train | 3000 | medium | 0.02 – 0.05 | 10 | 375 → 1875 |
| Train | 4000 | hard | 0.05 – 0.05 | 20 | — |
| Test | 1000 | easy | 0.1 – 0.3 | 5 | 125 → 625 |
| Test | 1000 | medium | 0.02 – 0.05 | 10 | 125 → 625 |
| Test | 1000 | hard | 0.05 – 0.05 | 20 | — |

These are encoded in `scripts/_common.sh`. For TarMAC on easy, the paper additionally evaluated three exploratory curriculum/epoch configurations; the
scripts encode the baseline configuration.

## The disturbance schedule

`test_noise_sweep.sh` reproduces the published testing schedule (`test_noise.sh` runs any single cell of it):

| Disturbance | Flags | Published settings |
|---|---|---|
| Gaussian noise | `--comm_constraints simple --noise_type gaussian --noise_level σ` | σ = 0.2, 0.5, 0.8 |
| Uniform noise | `--comm_constraints simple --noise_type uniform --noise_level u` | U[0, 0.5], U[0, 0.8] |
| Partial message loss | `--comm_constraints drops --drop_prob_part p` | p = 0.4, 0.7 |
| Whole message loss | `--comm_constraints drops --drop_prob_whole p` | p = 0.3, 0.6 |
| Combined loss | both drop flags jointly | (0.3, 0.4) and (0.6, 0.7) — effective 0.58 / 0.88 |
| Message jumbling | `--comm_constraints jumble --jumble_prob p` | p = 0.4, 0.7 |
| Message delays | `--comm_constraints delay --delay_prob p --delay_step 2` | p = 0.4, 0.7; max delay 2 |

## Attribution & license

The Traffic Junction environment in `ic3net-envs/` is the benchmark introduced
with CommNet and released with IC3Net (MIT license — retained verbatim in
`ic3net-envs/LICENSE.md`); the evaluated models are re-implementations /
adaptations of CommNet, IC3Net, TarMAC, GA-Comm and MAGIC as described in the
papers. Everything else: MIT, © 2026 Anastasios Kontogiorgis. If you use this
framework, please cite the papers above (`CITATION.cff`).
