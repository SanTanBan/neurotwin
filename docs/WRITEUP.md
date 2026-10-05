# NeuroTwin: calibrated digital twins of neural organ-on-a-chip cultures

**Category: End-to-End System.** A chip recording goes in; out come a mechanistic twin of that chip, checks
that say when the twin can be trusted, and in-silico experiments whose answers carry their own uncertainty.

> Draft v0.3 (2026-10-05). `[TBD]` marks results still running and links that exist only at submission.

- **Demo video (about 4 min):** [TBD: YouTube link]
- **Code (Apache-2.0):** https://github.com/SanTanBan/neurotwin, and the Kaggle notebooks `nt-*` of
  https://www.kaggle.com/santanubanerjee9 that produced every number
- **Technical report (PDF):** https://github.com/SanTanBan/neurotwin/blob/main/docs/NeuroTwin_report.pdf
- **Worked example notebook:** https://www.kaggle.com/code/santanubanerjee9/nt-demo

## Summary

Neural organ-on-a-chip platforms record tens of thousands of spikes a minute, yet their readout is
descriptive: firing rates, bursts, synchrony. NeuroTwin turns such a recording into a calibrated, mechanistic
digital twin of the chip. A fast spiking-network simulator of the 60-electrode chip, with synaptic depression,
adaptation and fast and slow synaptic currents, generates tens of thousands of virtual cultures. Amortised
simulation-based inference (a neural spline flow) then maps any recording's 16-statistic signature to a
posterior over the culture's biophysical parameters in a fraction of a second on a CPU. A reliability layer —
calibration on held-out simulations, an out-of-support score, posterior leakage, and predictive checks against
the twin of a different recording — decides when the twin can be trusted, and says so when it cannot.

On 3,049 five-minute windows of public recordings from developing rat cortical cultures, that layer rejected
our first model (76% of windows outside the simulator's range) and drove a concrete fix. The second round is
calibrated (90% intervals cover 88–92% of true values), each twin reproduces its own recording far better than
another recording's twin (87% vs 49% of statistics inside the predictive interval), and the 11% of windows that
pass every check show maturation in mechanistic terms: within individual cultures, drive rises and grows more
heterogeneous while adaptation weakens. Run forward, a twin answers questions the experiment never asked —
halving excitation silences bursting in every posterior draw, while halving inhibition raises synchrony by an
amount the recording cannot pin down — and it reports which answer is firm. A label-free imaging module and a
two-chamber (axon-diode) twin extend the system to structure and to oriented circuits.

## What the system does

| Stage | What it does | Where |
|---|---|---|
| Chip simulator | 400 leaky integrate-and-fire neurons on the real 60-electrode layout; synaptic depression, spike-frequency adaptation, fast and slow synaptic currents, heterogeneous drive; optional two-chamber axon-diode geometry. About 10 CPU-seconds per five-minute culture | `src/neurotwin/sim.py` |
| Signature | 16 statistics of rates, bursts, network bursts, synchrony and burstiness, computed by the same code for real and simulated data | `features.py` |
| Amortised inference | Neural posterior estimation (sbi, neural spline flow) trained once per round on a simulation bank | `infer.py`, `nt-train*` |
| Reliability layer | Simulation-based calibration, TARP and interval coverage; out-of-support score; posterior leakage; posterior predictive checks against a mismatched twin | `nt-train2`, report §5 |
| In-silico experiments | Interventions run on posterior draws and on several random networks, each paired with an unmodified run on the same network and noise | `nt-demo`, `app/app.py` |
| Label-free imaging | U-Net virtual staining of nuclei and dead cells with predictive uncertainty and count intervals calibrated on validation wells, against the published 2018 model | `nt-vstain2` |
| Oriented chips | Posterior over forward and backward axon-crossing probabilities; recording-length design study | `nt-train2c`, `nt-design2c` |

## Key results

| Question | Answer (public data; every number reproduced by a notebook) |
|---|---|
| Is the uncertainty honest? | 90% intervals cover 0.88–0.92 of held-out simulations for all 11 parameters; TARP deviation −0.009 |
| Does the twin know when it does not apply? | Round 1: 76% of real windows flagged outside the simulator's range, so no biological claim was made. Round 2: 341 of 3,049 windows (11%) pass every check, spread across all ages |
| Is a twin specific to its recording? | 87% of statistics inside the own twin's 90% predictive interval vs 49% for another recording's twin; the own twin fits better for 15 of 16 recordings |
| What does maturation look like mechanistically? | Within 20 cultures followed over ≥ 3 days: background drive rises in 16, excitability spread in 16, adaptation weakens in 17 (4 of 6 trends survive a 5% false-discovery correction) |
| Can it predict an untested intervention? | Culture 2-2, DIV 20: halving excitation stops bursting in all 16 posterior draws and on all 8 networks; halving inhibition never lowers synchrony and raises it by more than 0.05 in 9 of 16 draws — a firm and an uncertain answer, reported as such |
| Round 3 (slow synaptic current) | [TBD] |
| Label-free imaging | [TBD: corrected ensemble vs the published model (pixel r 0.92 / 0.85; count error 4.8%)] |
| Oriented chips | From 5 minutes of recording the wiring direction is read correctly in 61.9% of simulated chips, with honest, wide intervals; [TBD: recording-length study] |

## Why it matters

Organ-chips are now part of the regulatory path for New Approach Methodologies (FDA Modernization Act 2.0,
2022; FDA roadmap, April 2025). A chip only helps a decision if its readout can be interpreted and trusted.
NeuroTwin adds mechanism (which biophysical change explains a recording), uncertainty (how sure, per
parameter), prediction (what an untested intervention would do) and a veto (machine-checkable flags when the
twin does not apply) — on ordinary CPUs, in under a second per recording once trained.

## Limitations

The model is a reduced spiking network, so its parameters are effective quantities, not measured cellular
constants; three of eleven parameters are not identified by the statistics; amortised posteriors are
calibrated on average and can be broader than a single recording warrants; network-burst duration is still
too short in the model; the oriented-chip results use simulated ground truth because no public axon-diode
recordings were found; and the in-silico predictions are untested because the public recordings contain no
pharmacology. Each point is quantified in the report (§5).

## Team

[TBD: members and disciplines, to be confirmed by the team.]

## Data, licences and AI tools

MEA recordings: Wagenaar, Pine & Potter (2006), BMC Neuroscience 7:11, free for research with citation,
downloaded at run time. Images: Christiansen et al. (2018), Cell 173:792, CC BY 4.0. Software: numpy, scipy,
pandas, matplotlib, numba, PyTorch, sbi, nflows. Implementation was assisted by an AI coding assistant
(Anthropic Claude); all results are produced by the public code and were checked by the team.
