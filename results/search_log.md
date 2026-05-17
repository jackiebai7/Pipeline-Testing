# Hyperparameter Search Log

## Configuration

| Parameter | Value |
|-----------|-------|
| Objective | Maximize ARI |
| Max iterations | 20 |
| Percentile range | 50-100 |
| Quantile range | 0.05-0.50 |
| Window size | 30 frames |
| Epochs per stage | 300 |
| Device | CUDA |

## Results

| Iter | Percentile | Quantile | Clusters | ARI | Status |
|-----:|-----------:|---------:|---------:|----:|--------|
|  1 |   82.0 | 0.0550 |       13 | 0.2691 | done |
|  2 |   63.8 | 0.0950 |        8 | 0.4247 | done |
|  3 |   86.8 | 0.1850 |        5 | 0.3984 | done |
|  4 |   94.6 | 0.0670 |       10 | 0.0548 | done |
|  5 |   71.1 | 0.0560 |       18 | 0.2343 | done |
|  6 |   60.9 | 0.1510 |        6 | 0.5761 | done **BEST** |
|  7 |   51.3 | 0.0900 |       11 | 0.5504 | done |
|  8 |   82.5 | 0.1590 |        6 | 0.4529 | done |
|  9 |   61.0 | 0.1680 |        5 | 0.2253 | done |
| 10 |   90.5 | 0.0510 |       20 | 0.1847 | done |
| 11 |   90.3 | 0.1900 |        4 | 0.1794 | done |
| 12 |   67.0 | 0.0810 |       12 | 0.3384 | done |
| 13 |   97.9 | 0.1170 |        4 | 0.0862 | done |
| 14 |   54.6 | 0.0690 |       12 | 0.4731 | done |
| 15 |   92.4 | 0.1710 |        5 | 0.1154 | done |
| 16 |   90.4 | 0.1960 |        5 | 0.2128 | done |
| 17 |   76.8 | 0.2450 |        5 | 0.2680 | done |
| 18 |   68.9 | 0.1600 |        6 | 0.3828 | done |
| 19 |   91.5 | 0.1740 |        5 | 0.2239 | done |
| 20 |   93.1 | 0.1650 |        - | - | running |

## Best Result

- **Iteration:** 6
- **Percentile:** 60.9
- **Quantile:** 0.1510
- **Clusters found:** 6
- **ARI:** 0.5761
