# TalkingData AdTracking - Ablation v2

**Window**: last 40,000,000 train rows (same window/features as the main run). **Protocol**: forward-chained folds; train = rows strictly before the early-stop slice, early-stop = last 12% of the pre-eval period, eval = a test-hour block. Train keeps all positives + 10% of negatives; ES/eval keep 30% of negatives (AUC is unbiased under uniform negative subsampling). Metric: AUC on eval; delta = mean paired per-block delta (12 time blocks/fold), 95% bootstrap CI; verdict requires CI to exclude 0 AND |delta| > noise floor (0.00024; seed sd 0.00012, 3 seeds).

## Folds

| fold | train_rows | es_rows | eval_rows | eval_pos |
|---|---|---|---|---|
| h9-10 | 1537163 | 618946 | 1898941 | 16886 |
| h13-14 | 2709061 | 1089039 | 2082017 | 16821 |

## Calibration (chosen on early-stop AUC only)

| use_hour | spw_full | es_auc | best_iter | healthy |
|---|---|---|---|---|
| True | 1.00000 | 0.97968 | 437 | True |
| True | 10.00000 | 0.97998 | 238 | True |
| True | 50.00000 | 0.97961 | 164 | True |
| False | 1.00000 | 0.97914 | 315 | True |
| False | 10.00000 | 0.97953 | 189 | True |
| False | 50.00000 | 0.97968 | 208 | True |

Chosen: use_hour=True, spw_full=10. 

## Health check

0 model(s) stopped before iteration 25 (flagged UNSTABLE in results.csv). All models trained past the floor, so deltas compare real models.

## Base

| variant | n_features | auc | delta | ci_lo | ci_hi | blocks_worse | best_iter | verdict | d_h9-10 | d_h13-14 |
|---|---|---|---|---|---|---|---|---|---|---|
| BASE | 49 | 0.98221 | 0.00000 | 0.00000 | 0.00000 | - | 320 | BASE | 0.00000 | 0.00000 |

## Leave-one-group-out (delta < 0 => the group helps)

| variant | n_features | auc | delta | ci_lo | ci_hi | blocks_worse | best_iter | verdict | d_h9-10 | d_h13-14 |
|---|---|---|---|---|---|---|---|---|---|---|
| minus group: raw | 44 | 0.98102 | -0.00118 | -0.00138 | -0.00098 | 24/24 | 307 | WORSE than base | -0.00112 | -0.00126 |
| minus group: counts | 37 | 0.98166 | -0.00052 | -0.00080 | -0.00023 | 17/24 | 287 | WORSE than base | -0.00056 | -0.00054 |
| minus group: uniques | 42 | 0.98180 | -0.00039 | -0.00059 | -0.00020 | 21/24 | 294 | WORSE than base | -0.00055 | -0.00028 |
| minus group: newness | 45 | 0.98250 | 0.00029 | 0.00014 | 0.00045 | 5/24 | 301 | BETTER than base | 0.00034 | 0.00024 |
| minus group: deltas | 31 | 0.97623 | -0.00586 | -0.00677 | -0.00491 | 24/24 | 264 | WORSE than base | -0.00787 | -0.00410 |
| minus group: ratios | 46 | 0.98249 | 0.00030 | 0.00011 | 0.00050 | 7/24 | 307 | BETTER than base | 0.00015 | 0.00040 |

## Only-group (raw + one group)

| variant | n_features | auc | delta | ci_lo | ci_hi | blocks_worse | best_iter | verdict | d_h9-10 | d_h13-14 |
|---|---|---|---|---|---|---|---|---|---|---|
| raw only | 5 | 0.96435 | -0.01793 | -0.02017 | -0.01567 | 24/24 | 452 | WORSE than base | -0.02286 | -0.01286 |
| raw + counts | 17 | 0.97370 | -0.00855 | -0.00958 | -0.00745 | 24/24 | 277 | WORSE than base | -0.01062 | -0.00639 |
| raw + uniques | 12 | 0.97233 | -0.00989 | -0.01124 | -0.00848 | 24/24 | 293 | WORSE than base | -0.01277 | -0.00699 |
| raw + newness | 9 | 0.96871 | -0.01354 | -0.01518 | -0.01188 | 24/24 | 340 | WORSE than base | -0.01731 | -0.00970 |
| raw + deltas | 23 | 0.97788 | -0.00431 | -0.00481 | -0.00382 | 24/24 | 377 | WORSE than base | -0.00520 | -0.00347 |
| raw + ratios | 8 | 0.96997 | -0.01207 | -0.01400 | -0.01011 | 24/24 | 426 | WORSE than base | -0.01625 | -0.00823 |

## Build-up

| variant | n_features | auc | delta | ci_lo | ci_hi | blocks_worse | best_iter | verdict | d_h9-10 | d_h13-14 |
|---|---|---|---|---|---|---|---|---|---|---|
| build-up 1: +counts | 17 | 0.97370 | -0.00855 | -0.00958 | -0.00745 | 24/24 | 277 | WORSE than base | -0.01062 | -0.00639 |
| build-up 2: +newness | 21 | 0.97462 | -0.00756 | -0.00848 | -0.00663 | 24/24 | 275 | WORSE than base | -0.00944 | -0.00573 |
| build-up 3: +uniques | 28 | 0.97564 | -0.00656 | -0.00746 | -0.00566 | 24/24 | 365 | WORSE than base | -0.00842 | -0.00472 |
| build-up 4: +deltas | 46 | 0.98249 | 0.00030 | 0.00011 | 0.00050 | 7/24 | 307 | BETTER than base | 0.00015 | 0.00040 |

## Feature-set variants

| variant | n_features | auc | delta | ci_lo | ci_hi | blocks_worse | best_iter | verdict | d_h9-10 | d_h13-14 |
|---|---|---|---|---|---|---|---|---|---|---|
| R-kernel set (no deltas/uniques) | 24 | 0.97527 | -0.00680 | -0.00783 | -0.00577 | 24/24 | 323 | WORSE than base | -0.00880 | -0.00509 |
| + r_iph_ip (drift-dropped) | 50 | 0.98223 | 0.00002 | -0.00014 | 0.00020 | 13/24 | 353 | no measurable difference | -0.00022 | 0.00027 |
| minus hour | 48 | 0.98198 | -0.00022 | -0.00039 | -0.00005 | 16/24 | 306 | tiny (below noise floor) | -0.00012 | -0.00035 |

## Training-setup ablations

| variant | n_features | auc | delta | ci_lo | ci_hi | blocks_worse | best_iter | verdict | d_h9-10 | d_h13-14 |
|---|---|---|---|---|---|---|---|---|---|---|
| spw=1 (full-data equiv.) | 49 | 0.98224 | 0.00005 | -0.00015 | 0.00025 | 14/24 | 468 | no measurable difference | 0.00022 | -0.00015 |
| spw=50 (full-data equiv.) | 49 | 0.98209 | -0.00011 | -0.00032 | 0.00011 | 16/24 | 231 | no measurable difference | -0.00025 | 0.00000 |
| shallow trees (7 leaves, depth 4) | 49 | 0.98189 | -0.00029 | -0.00049 | -0.00009 | 15/24 | 499 | WORSE than base | -0.00021 | -0.00042 |
| max_bin=255 | 49 | 0.98227 | 0.00006 | -0.00009 | 0.00020 | 10/24 | 279 | no measurable difference | 0.00007 | 0.00005 |
| main-run params (lr .08, mcs 200, bin 255, spw 50) | 49 | 0.98228 | 0.00008 | -0.00013 | 0.00029 | 9/24 | 264 | no measurable difference | 0.00007 | 0.00007 |
| train negatives kept = 0.03 | 49 | 0.98436 | 0.00005 | -0.00019 | 0.00029 | 5/12 | 496 | no measurable difference |  | 0.00006 |
| train negatives kept = 0.3 | 49 | 0.98373 | -0.00055 | -0.00075 | -0.00035 | 12/12 | 301 | WORSE than base |  | -0.00058 |

## Drop-one for the top permutation features

| variant | n_features | auc | delta | ci_lo | ci_hi | blocks_worse | best_iter | verdict | d_h9-10 | d_h13-14 |
|---|---|---|---|---|---|---|---|---|---|---|
| minus feature: dt_ip_app_dev_os_next1 | 48 | 0.97964 | -0.00254 | -0.00284 | -0.00221 | 24/24 | 262 | WORSE than base | -0.00298 | -0.00215 |
| minus feature: cnt_app | 48 | 0.98224 | 0.00003 | -0.00015 | 0.00022 | 13/24 | 359 | no measurable difference | 0.00012 | -0.00006 |
| minus feature: cnt_app_os | 48 | 0.98203 | -0.00016 | -0.00033 | 0.00002 | 17/24 | 305 | no measurable difference | -0.00013 | -0.00023 |
| minus feature: cnt_app_chan | 48 | 0.98211 | -0.00009 | -0.00025 | 0.00008 | 14/24 | 260 | no measurable difference | 0.00010 | -0.00029 |
| minus feature: device | 48 | 0.98170 | -0.00050 | -0.00068 | -0.00032 | 22/24 | 249 | WORSE than base | -0.00045 | -0.00057 |

## Importance

Group-level permutation (drop in AUC):

| group | perm_drop_auc | perm_drop_sd | n |
|---|---|---|---|
| counts | 0.08192 | 0.00028 | 12 |
| deltas | 0.05743 | 0.00042 | 18 |
| uniques | 0.01645 | 0.00019 | 7 |
| raw | 0.01082 | 0.00007 | 5 |
| newness | 0.00043 | 0.00002 | 4 |
| ratios | 0.00022 | 0.00002 | 3 |

Top 15 features (permutation drop in AUC, gain share):

| feature | group | perm_drop_auc | perm_drop_sd | gain_share |
|---|---|---|---|---|
| dt_ip_app_dev_os_next1 | deltas | 0.02465 | 0.00049 | 0.03738 |
| cnt_app | counts | 0.01194 | 0.00004 | 0.31504 |
| cnt_app_os | counts | 0.00631 | 0.00016 | 0.01283 |
| cnt_app_chan | counts | 0.00607 | 0.00009 | 0.06357 |
| app | raw | 0.00295 | 0.00006 | 0.01807 |
| device | raw | 0.00280 | 0.00006 | 0.04467 |
| cnt_ip_dev_os | counts | 0.00204 | 0.00007 | 0.04944 |
| dt_ip_app_dev_os_next2 | deltas | 0.00164 | 0.00013 | 0.00551 |
| nu_ip_channel | uniques | 0.00154 | 0.00005 | 0.15775 |
| nu_app_channel | uniques | 0.00137 | 0.00003 | 0.02240 |
| os | raw | 0.00126 | 0.00006 | 0.00457 |
| nu_ip_app | uniques | 0.00098 | 0.00004 | 0.13807 |
| channel | raw | 0.00098 | 0.00004 | 0.01503 |
| cnt_ip_dh | counts | 0.00072 | 0.00005 | 0.00168 |
| dt_ip_app_chan_next1 | deltas | 0.00057 | 0.00002 | 0.02446 |

## Takeaways (auto-generated; read together with the tables)

- Groups whose removal measurably hurts: deltas (-0.0059), raw (-0.0012), counts (-0.0005), uniques (-0.0004).
- Groups whose removal measurably helps (prune candidates): newness (+0.0003), ratios (+0.0003).
- Largest training-setup effects: train negatives kept = 0.3 (-0.0005), shallow trees (7 leaves, depth 4) (-0.0003), spw=50 (full-data equiv.) (-0.0001).
- Limit: one-day window => `hour` and day-level drift cannot be validated; features are computed over the whole window (as in the main pipeline), the absolute AUC is on negative-subsampled eval rows (unbiased).