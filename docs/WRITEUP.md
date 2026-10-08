# NeuroTwin: calibrated digital twins of neural organ-on-a-chip cultures

*Subtitle:* A chip recording goes in; a mechanistic twin of that chip comes out, with checks that say when to
trust it and in-silico experiments whose answers carry their own uncertainty.

## Category Declaration

**End-to-End System.** Data input (a microelectrode-array recording of a neural chip) → mechanistic chip model
→ calibrated amortised inference with reliability checks → in-silico experiments and chip-design answers.

## Demo Video

[YouTube link — 4 min 19 s]

## Code Repository Link

- **GitHub (Apache-2.0):** https://github.com/SanTanBan/neurotwin — entry script `python demo.py` (fits the
  worked example end to end on a CPU in about 10 minutes, with the trained posterior shipped in
  `assets/round3/`); README with the full notebook chain.
- **Kaggle notebook:** https://www.kaggle.com/code/santanubanerjee9/nt-demo (the worked example). Every other
  number comes from a Kaggle notebook whose exact script is in the repository (`kaggle/<name>/`, chain and
  runtimes in the report's §7), so each can be re-run as a Kaggle notebook.

## Project Summary

Neural organ-on-a-chip platforms record tens of thousands of spikes a minute, yet their readout is descriptive:
firing rates, bursts, synchrony. NeuroTwin turns such a recording into a calibrated, mechanistic digital twin
of the chip. A fast spiking-network simulator of the 60-electrode chip — with synaptic depression, adaptation,
and fast and slow synaptic currents — generates tens of thousands of virtual cultures. Amortised
simulation-based inference (a neural spline flow) then maps a recording's statistical signature to a posterior
over the culture's biophysical parameters in a fraction of a second on a CPU. A reliability layer — calibration
on held-out simulations, an out-of-support score, posterior leakage, and predictive checks against the twin of
a different recording — decides when the twin can be trusted, and says so when it cannot.

On 3,049 five-minute windows of public recordings from developing rat cortical cultures, that layer rejected
our first model (76% of windows outside the simulator's range) and drove two rounds of fixes. The final model
is calibrated (90% intervals cover 86–92% of true values), trusts itself on twice as many windows as the round
before (22%), and each twin reproduces its own recording far better than another recording's twin (91% vs 52%
of statistics inside the predictive interval). Within cultures followed over weeks, the twins describe
maturation in mechanistic terms — rising and more heterogeneous drive, faster-decaying adaptation — in two model
versions. Run forward, a twin answers questions the experiment never asked: halving excitation silences
bursting in every posterior draw, while halving inhibition usually raises synchrony by an amount the recording
cannot pin down — and the twin says which answer is firm. A label-free imaging module (nuclei-count error
1.8%, with calibrated intervals) and a two-chamber axon-diode twin extend the system to structure and to
oriented circuits.

## Technical Report

**Full report (PDF, about 18 pages):** https://github.com/SanTanBan/neurotwin/blob/main/docs/NeuroTwin_report.pdf
It covers the problem, data and licences, methods, experiments and results,
reliability and limitations, impact, reproduction, team and references. A condensed version follows.

### Data
- **MEA recordings:** Wagenaar, Pine & Potter (2006), BMC Neuroscience 7:11 — spontaneous activity of rat
  cortical cultures on 60-electrode arrays, recorded daily from 3 to 39 days in vitro; 3,049 five-minute
  windows from 522 recordings; free for research with citation; downloaded at run time.
- **Images:** Christiansen et al. (2018), Cell 173:792 — phase-contrast z-stacks with nuclear and dead-cell
  stains of rat cortical cultures (73 training, 8 test wells) and the published model's predictions; CC BY 4.0.
- **Simulations:** 151,680 virtual cultures over three rounds, plus 18,560 two-chamber chips.

### Methods
- **Chip simulator:** 400 leaky integrate-and-fire neurons on the real 60-electrode layout; distance-dependent
  wiring, short-term synaptic depression, spike-frequency adaptation, fast and slow (NMDA-like) synaptic
  currents, heterogeneous noisy drive; an optional two-chamber axon-diode geometry. Event-driven and
  numba-compiled: about 10 CPU-seconds per five-minute culture. Each model version reproduces the previous one
  exactly at its neutral setting (regression-tested).
- **Signature:** 16 statistics of rates, single-electrode and network bursts, synchrony and burstiness,
  computed by the same code for real and simulated data.
- **Inference:** neural posterior estimation with a neural spline flow (sbi), trained once per round; rounds 2
  and 3 concentrate simulations where real cultures live.
- **Reliability layer:** simulation-based calibration, TARP and interval coverage; a 10-nearest-neighbour
  out-of-support score; posterior leakage; posterior predictive checks against a mismatched twin.
- **In-silico experiments:** interventions on 16 posterior draws and on 8 random networks, each paired with an
  unmodified run on the same network and noise, so parameter uncertainty and network randomness stay apart.
- **Imaging:** a U-Net ensemble with heteroscedastic outputs; readouts (nuclei count, dead fraction) with
  intervals calibrated on validation wells.

### Experiments and results

| Question | Answer |
|---|---|
| Is the uncertainty honest? | 90% intervals cover 0.86–0.92 of held-out simulations for all 13 parameters; TARP deviation −0.008; all SBC rank tests pass |
| Does the twin know when it does not apply? | Round 1 flagged 76% of real windows as outside the simulator's range, so no biological claim was made from it; the final round passes 668 of 3,049 windows (22%, twice round 2's 11%) |
| Is a twin specific to its recording? | 91% of statistics inside the own twin's 90% predictive interval vs 52% for another recording's twin (16 recordings) |
| What does maturation look like mechanistically? | Within 20 cultures: excitability spread rises in 19, excitatory efficacy in 17, drive in 16; adaptation decays faster in 18. Four of eight trends survive a 5% false-discovery correction; drive, spread and faster adaptation also hold in the previous model version |
| Can it predict an untested intervention? | Culture 2-2 at 20 days: halving excitation stops bursting in all 16 posterior draws and on all 8 networks; halving inhibition raises synchrony in 9 of 16 draws and lowers it in 1 — a firm and an uncertain answer, reported as such |
| Label-free imaging (8 test wells) | Nuclei count error 1.8% (published model 4.8%), and the 90% count intervals hold in 8 of 8 wells; dead-cell fraction error 0.014 (published 0.011); pixel fidelity below the published model (r 0.77 vs 0.92) |
| Oriented (axon-diode) chips | The wiring direction is read correctly 59% of the time from 5 minutes of spontaneous activity and 65% from 40 minutes, with overstated pooled confidence: the design rule is to probe direction with stimulation, not to record longer |

### Reproduction
Every number comes from a public Kaggle notebook, each a single script built from `kaggle/<name>/template.py`
(the report's §7 lists the chain and runtimes). All run on free CPU sessions except the imaging ensemble (one
GPU session). `python demo.py` reproduces the worked example locally; seeds are fixed throughout.

### Limitations
The model is a reduced spiking network, so its parameters are effective quantities rather than measured
cellular constants; several parameters are weakly identified by the statistics (reported per parameter);
network-burst duration is still too short in the model; amortised posteriors are calibrated on average and can
be broader than a single recording warrants; the oriented-chip results use simulated ground truth because no
public axon-diode recordings were found; and the in-silico predictions are untested because the public
recordings contain no pharmacology. The report quantifies each point (§5).

## Optional Demo Link

https://www.kaggle.com/code/santanubanerjee9/nt-demo — the worked example as a Kaggle notebook (about three
minutes of computation on a free CPU session): fit, reliability verdict, predictive check and two in-silico
experiments. Locally: `python demo.py`; interactive app: `python app/app.py` (Gradio).

## Team

Santanu Banerjee (team leader and sole member) — Master of Science (by Research) in Operations Research,
Indian Institute of Technology Kharagpur, India.

## Data, licences and AI tools

MEA recordings: Wagenaar, Pine & Potter (2006), free for research with citation, downloaded at run time.
Images: Christiansen et al. (2018), CC BY 4.0. Software: numpy, scipy, pandas, matplotlib, numba, PyTorch,
sbi, nflows. Implementation was assisted by an AI coding assistant (Anthropic Claude); all results are produced
by the public code and were checked by the author.
