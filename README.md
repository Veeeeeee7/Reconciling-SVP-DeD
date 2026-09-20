# Reconciling Set-Valued Policy & Dead-End Discovery in Healthcare Reinforcement Learning

Code for the PSB 2027 paper _Reconciling Set-Valued Policy & Dead-End Discovery in Healthcare Reinforcement Learning: An Empirical Analysis_ (Li, Wu, Tang).

The paper studies whether two clinician-in-the-loop RL methods agree with each other: **Set-Valued Policies (SVP)**, which recommend every action within a relative near-optimality margin $\zeta$, and **Dead-End Discovery (DeD)**, which eliminates every action whose best-case probability of inevitable death exceeds an absolute threshold $\theta_D$. This repository reproduces every result and figure in the paper.

## Layout

```
lifegate_synthetic/   synthetic grid-world domain
  lifegate.py                    environment
  svp.py                         SVP and DeD solvers, near-greedy value iteration
  train_policies.py              hyperparameter sweep over (zeta, theta_D)
  cycle_stats.py                 how often the near-greedy iteration cycles
  visualize_*.py                 environment, policy-level and state-level figures
  summarize_state_level_stats.py per-state conflict frequency and IoU

mimic_sepsis/         real ICU sepsis cohort
  consistency.py                 solvers, cycle handling, conflict metrics
  experiments.py                 state-level analysis at the clinical setting
  additional_experiments.py      hyperparameter sweeps
  bootstrap.py                   bootstrap CIs over resampled ICU stays
  cycle_stats_mimic.py           cycle frequency across the zeta sweep
  reconciliation_eval.py         reconciliation ordering over the grid
  reconciliation_focal.py        reconciliation ordering at the clinical setting
  conflict_action_eval.py        conflicting actions vs. their SVP alternatives
  tier_mortality.py              one-step mortality of the four ordering tiers
  state_level_figure.py          DeD Q-values against the SVP policy
  visualizations.py              policy-level heatmaps and set-size curves

figures/              the figures as they appear in the paper
```

## Data

**No patient data is included in this repository.** The LifeGate experiments are fully self-contained and run out of the box. The MIMIC-III experiments require credentialed access to [MIMIC-III](https://physionet.org/content/mimiciii/) via PhysioNet.

The MIMIC scripts expect a preprocessed trajectory file at `mimic_sepsis/mimic_sepsis_data_2025/traj_shifted_train.csv` (and `_val`, `_test`) with one row per transition and the columns:

| column                          | meaning                                                                   |
| ------------------------------- | ------------------------------------------------------------------------- |
| `traj`                          | trajectory (ICU stay) id                                                  |
| `step`                          | index of the transition within the trajectory                             |
| `s:state`                       | discrete state, `0..749`; `750` = discharge, `751` = death                |
| `a:action`                      | discrete action, `0..24`, encoded as `5 * IV-fluid bin + vasopressor bin` |
| `r:reward`                      | terminal reward, `+1` discharge / `-1` death / `0` otherwise              |
| `s:next_state`, `a:next_action` | successor state and action                                                |
| `done`                          | terminal flag                                                             |

States are 750 K-means clusters and actions are 5x5 fluid/vasopressor bins, following Komorowski et al. (2018). Any pipeline producing this schema will work.

## Reproducing the results

```bash
pip install -r requirements.txt

cd lifegate_synthetic                 # no data required
python train_policies.py              # sweeps (zeta, theta_D) at each drag level
python visualize_policy_level.py      # policy-level heatmaps, drag comparison
python summarize_state_level_stats.py # per-state conflict frequency and IoU
python visualize_state_level.py       # state-level figures
python visualize_environment.py       # environment figure
python cycle_stats.py                 # cycle frequency

cd ../mimic_sepsis                    # requires the trajectory file described above
python consistency.py                 # run first: solves the policy grid and writes results/{svp,ded}_policies.npy
python additional_experiments.py      # 99 x 99 hyperparameter sweep
python experiments.py                 # state-level analysis
python bootstrap.py                   # bootstrap CIs over resampled ICU stays
python cycle_stats_mimic.py           # cycle frequency
python conflict_action_eval.py        # conflicting actions vs. their SVP alternatives
python tier_mortality.py              # four-tier one-step mortality
python reconciliation_eval.py         # reconciliation ordering over the grid
python reconciliation_focal.py        # reconciliation ordering at the clinical setting
python state_level_figure.py          # state-level figure
python visualizations.py              # policy-level heatmaps and set-size curves
```

Scripts write arrays to `results/` and figures to `figures/`; neither is committed. The hyperparameter sweeps are the expensive step and use multiprocessing where available.

## Citation

```bibtex
@inproceedings{li2027reconciling,
  title={Reconciling Set-Valued Policy \& Dead-End Discovery in Healthcare
         Reinforcement Learning: An Empirical Analysis},
  author={Li, Victor and Wu, Sixing and Tang, Shengpu},
  booktitle={Pacific Symposium on Biocomputing},
  year={2027}
}
```
