# Demo video script (target 4:30, hard limit 5:00)

Narration is synthesised by `tools/make_video.py`; each scene lasts as long as its narration. `[TBD]` marks
numbers to be filled from finished runs before recording. Visuals are existing figures or short screen
captures of the demo notebook and app.

| # | Visual | Narration | ~s |
|---|---|---|---|
| 1 | Real raster, culture 2-2 at DIV 20 (`runs/figs/day1_overview.png`, panel A) | "This is a living neural circuit on a chip: rat cortical neurons growing on sixty electrodes. Every dot is a spike. Twice a minute the whole network fires together." | 18 |
| 2 | Same raster with statistics overlaid | "Labs summarise this with firing rates, burst counts and synchrony. Useful numbers, but they only describe. They cannot tell you why the network bursts, how sure you should be, or what it would do under a change you have not tried." | 22 |
| 3 | Architecture diagram | "NeuroTwin turns the recording into a mechanistic digital twin: a simulated culture of the same chip, whose biophysical parameters are inferred from that recording, with calibrated uncertainty." | 20 |
| 4 | Simulator schematic: electrodes, E/I neurons, synapses | "The simulator is the chip itself: four hundred neurons on the same electrode layout, with synaptic depression, adaptation, synaptic kinetics and a noisy drive. It runs a five-minute culture in about two seconds on one core." | 22 |
| 5 | Bank + posterior training figure | "We simulate tens of thousands of virtual cultures, then train a neural posterior estimator once. After that, fitting any recording takes about a second." | 18 |
| 6 | `runs/figs/coverage.png` + reliability figure | "Our first attempt failed, and the system told us so. Three quarters of real recordings fell outside anything the simulator could produce, and the posterior flagged its own fits as extrapolation. A confident but wrong answer is the real danger in this field." | 28 |
| 7 | v3 model change + focused prior | "The diagnosis was concrete: instantaneous synapses cannot make a two-second burst, and bursting cultures occupied a corner of the prior. We gave synapses realistic decay kinetics and concentrated the simulations where real cultures live." | 24 |
| 8 | Round-2 calibration + PPC figures | "Round two: [TBD recovery/calibration numbers], and the fitted twin reproduces its own recording better than a twin fitted to a different recording." | 22 |
| 9 | `trajectories.png` | "Across [TBD] recordings of [TBD] cultures the twins show how these networks mature: [TBD trajectory finding]." | 24 |
| 10 | Demo notebook: in-silico experiment | "Because it is mechanistic, the twin answers questions the experiment did not. Halve the inhibition in silico and the network [TBD effect] — a prediction a lab can test on the same chip." | 22 |
| 11 | Virtual staining panel (real vs ours vs 2018 baseline) | "A label-free imaging module predicts nuclei and dead-cell stains from unstained images, with per-pixel uncertainty, [TBD comparison to the published baseline]. Structure feeds the twin without staining the culture." | 24 |
| 12 | Two-chamber design map | "For oriented chips built with axon diodes, the twin reads which way axons were engineered to grow from the delay between chambers, and maps how channel asymmetry becomes functional directionality." | 22 |
| 13 | Repo + notebooks + app | "Everything runs on CPU. The data is public, the notebooks reproduce every number, and the app fits a twin to your own recording." | 16 |
| 14 | Closing slide | "NeuroTwin: from description to prediction for neural organs-on-chips, with uncertainty you can check." | 12 |

Total ≈ 4:54 at the measured speaking rate; trim scenes 2, 7 and 13 if the recorded audio runs long.

## Recording notes
- Screen captures: demo notebook run (scene 10) and the app's three tabs (scene 13); keep each under 15 s.
- Every claim in the narration must match a number in the report; no rounding up.
- If a module is not finished in time, drop its scene rather than describe it vaguely.
