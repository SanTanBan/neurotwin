# NeuroTwin: calibrated digital twins of neural organ-on-a-chip cultures

**Submission category: End-to-End System** (AI-driven digital twin: data input → mechanistic model → calibrated inference → actionable in-silico experiments)

> Living draft (v0.1, 2026-09-11). Items marked `[TBD]` are filled in as experiments complete; no number
> appears here before it has been measured.

- **Demo video:** [TBD]
- **Code repository:** [TBD — public Kaggle notebook + GitHub]
- **Interactive demo:** [TBD]

## Project summary

Neural organ-on-a-chip (OoC) platforms, from cortical networks grown on microelectrode arrays (MEAs) to
oriented circuits wired with axon-diode microchannels, are leading human-relevant alternatives to animal
testing for CNS research and drug development. Their readouts, however, are high-dimensional and purely
descriptive: an experimenter sees rasters and bursts, not mechanisms, and cannot ask the chip "what would
you do if…?".

**NeuroTwin** turns a neural-chip recording into a calibrated, mechanistic digital twin. A fast
event-driven spiking-network simulator models the chip itself: its electrode layout, excitatory and inhibitory
neurons, short-term synaptic depression, adaptation, distance-dependent wiring, and optionally
compartmentalized axon-diode geometry. Amortized simulation-based inference (neural posterior estimation),
trained once on a large bank of virtual cultures, maps any recording's electrophysiological signature to a
full posterior over nine interpretable biophysical parameters in milliseconds. The uncertainty is verified
by simulation-based calibration, and a model-misspecification check flags recordings the twin cannot
explain.

The reliability layer is not decoration: on 522 public recordings of developing rat cortical cultures
(Wagenaar, Pine & Potter 2006; 3,049 five-minute windows) it rejected our own first attempt. A bank of
81,664 simulated cultures covered young cultures but not mature bursting ones (75% of windows out of
support), and the posterior said so through explicit out-of-support and leakage flags. That diagnosis drove
a concrete fix — synapses with realistic decay kinetics and a prior concentrated where real cultures live —
rather than a confidently wrong biological claim. [TBD: round-2 results, developmental trajectories,
forecasting.]

A label-free imaging module closes the loop from images to function: virtual staining of nuclei and dead
cells from transmitted-light images of rat cortical cultures (CC-BY Google in-silico-labeling data),
benchmarked against the published model's own predictions on the same test wells, with calibrated
uncertainty on the readouts an experimenter uses. [TBD: results.] A twin of two-chamber oriented chips
recovers which way axons were engineered to grow — the defining feature of axon-diode devices — from the
onset lag between chambers, and maps how channel asymmetry becomes functional directionality. Everything
runs on CPU; code, trained models and an interactive app are public.

## Technical report (outline)

1. Problem & application scenario — neural OoC readouts, NAM context, why a mechanistic twin
2. Data — sources, licenses, processing, compliance
   - Wagenaar, Pine & Potter 2006 dense-culture MEA dataset (free with citation; 527 daily recordings)
   - [planned] Kapucu et al. 2022 comparative rat/hPSC MEA dataset (CC BY 4.0)
   - [planned] Google in-silico-labeling data, rat cortical conditions (CC BY 4.0)
3. Methods
   - 3.1 Chip simulator (LIF + Tsodyks–Markram STD + adaptation; 60-electrode layout; axon-diode compartments)
   - 3.2 Electrophysiological signature (16 summary statistics, identical for real and simulated data)
   - 3.3 Amortized inference (NPE with neural spline flows), calibration (SBC, TARP), misspecification test
   - 3.4 Developmental trajectories and forecasting
   - 3.5 Label-free structural module (virtual staining + uncertainty → structural priors)
   - 3.6 Oriented-circuit twin and in-silico chip design
4. Experiments & results — [TBD]
5. Reliability analysis & limitations — [TBD]
6. Impact — [TBD]
7. Reproduction — [TBD]
8. Team composition — [TBD: user to confirm members and backgrounds]
9. External resources & AI tools disclosure — sbi, numba, PyTorch; AI coding assistance (Claude) used for
   implementation, all results verified by the team.
