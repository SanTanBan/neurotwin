# Demo video script (target about 4:15, hard limit 5:00)

Narration is synthesised locally by `tools/make_video.py` (Windows speech, voice "Microsoft Zira"); each scene
lasts as long as its narration plus 0.6 s. Every number below is taken from docs/REPORT.md; `[TBD]` marks
results still running. Visuals are report figures (`docs/figures/`) and slides made by `tools/make_figures.py`
(`runs/figs/`). The scene list itself is `video/scenes.json`.

| # | Visual | Narration | ~s |
|---|---|---|---|
| 1 | `runs/figs/real_2-2-20.png` (a 45-minute recording) | "This is a living neural circuit on a chip: rat cortical neurons growing over sixty electrodes. Every dot is a spike. Every couple of minutes, the whole network fires at once." | 12 |
| 2 | `runs/figs/day1_overview.png` (development across 522 recordings) | "Labs summarise recordings like these with firing rates, burst counts and synchrony, and watch them change as cultures mature. These numbers describe. They cannot say why a network bursts, how sure we should be, or what it would do under a change nobody has tried." | 18 |
| 3 | `docs/figures/architecture.png` | "NeuroTwin turns a recording into a mechanistic digital twin: a simulated culture on the same chip, whose biophysical parameters are inferred from that recording, together with checks that say when the twin can be trusted." | 14 |
| 4 | slide `runs/figs/simulator.png` | "The simulator is the chip itself: four hundred model neurons on the same sixty-electrode layout, with synaptic depression, adaptation, fast and slow synaptic currents, and noisy drive. A five-minute culture costs about ten seconds of one processor core." | 16 |
| 5 | `docs/figures/round2_sbc.png` | "We simulate tens of thousands of virtual cultures and train a neural posterior estimator once. Fitting a new recording then takes a fraction of a second, and its uncertainty is honest: ninety-percent intervals contain the true value eighty-eight to ninety-two percent of the time." | 18 |
| 6 | `docs/figures/round1_coverage.png` | "Our first round failed, and the system said so. Three in four real recording windows lay outside anything the simulator could produce, and the twin flagged its own fits as extrapolation. A confident wrong answer is the real danger in this field." | 17 |
| 7 | slide `runs/figs/diagnosis.png` | "The diagnosis was concrete. Instantaneous synapses cannot sustain a two-second burst, and bursting cultures were rare among the simulations. We gave synapses realistic time courses, and concentrated the simulations where real cultures live." | 15 |
| 8 | `docs/figures/ppc_own_vs_mismatched.png` | "Now each twin reproduces its own recording: eighty-seven percent of its statistics fall inside the predicted range, against forty-nine percent for the twin of a different recording. And the twin still declines most mature recordings: only eleven percent of windows pass every check, and it says which." | 19 |
| 9 | `docs/figures/maturation.png` | "Across one hundred and thirty-four culture-days that pass every check, the twins describe maturation in mechanistic terms: within individual cultures, background drive and excitability spread rise, and spike-frequency adaptation weakens." | 13 |
| 10 | `runs/figs/demo_top.png` (top half of Figure 8) | "Here is one culture's twin, fitted to minutes twenty to twenty-five of its recording. Like the culture, it fires large, network-wide bursts. Because the twin is mechanistic, we can ask it questions the experiment never asked." | 15 |
| 11 | `docs/figures/demo_in_silico.png` | "Halve its excitation, and bursting stops in every version of the twin, on every simulated network: a confident prediction. Halve its inhibition, and synchrony never falls and often rises, by an amount this recording cannot pin down. The twin says so, instead of guessing. Both are experiments a lab can run on the same chip." | 23 |
| 12 | imaging figure `[TBD]` | "A label-free imaging module predicts nuclear and dead-cell stains from unstained images, with count intervals calibrated on validation wells: [TBD, the corrected run against the published 2018 model]." | 15 |
| 13 | `docs/figures/two_chamber.png` and `[TBD]` recording-length figure | "For oriented chips built with axon diodes, the twin reads which way the axons were engineered to grow. From five minutes of recording it gets the direction right only sixty-two percent of the time, and its intervals say so. [TBD: with forty minutes, …]. That is a design rule for chip experiments." | 22 |
| 14 | slide `runs/figs/public.png` | "Everything is public and runs on ordinary computers: the data, the notebooks that reproduce every number, and an app that fits a twin to your own recording." | 10 |
| 15 | `runs/figs/closing.png` | "NeuroTwin: from description to prediction for neural organs-on-chips, with uncertainty you can check." | 7 |

Estimated total about 4:15 including pauses. If the audio runs long, shorten scenes 2 and 7 first.

## Rules
- Every claim in the narration must match a number in the report; no rounding up.
- If a module is not finished in time, drop its scene rather than describe it vaguely.
- Scene 11 must keep both halves of the in-silico result: the firm one (excitation) and the uncertain one
  (inhibition).
