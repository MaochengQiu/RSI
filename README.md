# RSI weight-perturbation experiments

This repository contains small-scale experiments on gradient-free weight perturbation for language-model agents. The current scope keeps the agent harness fixed and studies two questions:

1. Can a perturbed single model outperform its base model on held-out tasks?
2. Can reward-weighted perturbations produce a useful shared-model update in an interactive environment?

## Experiments

| Directory | Environment | Main contents |
| --- | --- | --- |
| [`experiments/countdown`](experiments/countdown) | Countdown | Clean reconstruction and paired evaluation of a selected RandOpt candidate |
| [`experiments/webshop`](experiments/webshop) | WebShop | Perturbation sweeps, candidate selection, sealed-task evaluation, and analysis scripts |
| [`experiments/alfworld`](experiments/alfworld) | ALFWorld | Local perturbation tests and a small standard-ESOpt aggregation run |
| [`experiments/agentic-esopt-sudoku`](experiments/agentic-esopt-sudoku) | Sudoku | Evaluation of the released Agentic ESOpt checkpoint, public-log analysis, and short local ES diagnostics |

Each experiment directory contains its own setup notes, executable entry points, and result files. Large model checkpoints and raw runtime outputs are intentionally excluded from Git.

## Current evidence boundary

- Countdown produced one reconstructed candidate that outperformed the base model on 2,000 tasks not used for that selection.
- The selected WebShop candidate did not outperform the base model on the 60-task sealed set.
- The released Agentic ESOpt Sudoku checkpoint was substantially stronger than the base model in local evaluation; the full 100-generation training run was not reproduced locally.
- In the small ALFWorld ESOpt run, all four directions received the same development reward, so z-score aggregation produced a zero update.

These experiments use different models and protocols. Their results should be read separately rather than as one continuous training pipeline.

## Upstream projects

- [RandOpt / Neural Thickets](https://github.com/sunrainyg/RandOpt)
- [WebShop](https://github.com/princeton-nlp/WebShop)
- [ALFWorld](https://github.com/alfworld/alfworld)
- [Agentic ESOpt](https://github.com/zz1358m/Agentic-ESOpt)
- [HGM](https://github.com/metauto-ai/HGM)
