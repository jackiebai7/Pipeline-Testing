### Summary

============================================================
HYPERPARAMETER SEARCH COMPLETE
============================================================

Search log:
  Iter  1: percentile=  51.4, quantile=0.1025  →  7 clusters, ARI=0.4829
  Iter  2: percentile=  47.8, quantile=0.1223  →  5 clusters, ARI=0.5669
  Iter  3: percentile=  52.4, quantile=0.1677  →  5 clusters, ARI=0.5730
  Iter  4: percentile=  53.9, quantile=0.1087  →  11 clusters, ARI=0.6784
  Iter  5: percentile=  49.2, quantile=0.1030  →  8 clusters, ARI=0.6581
  Iter  6: percentile=  47.2, quantile=0.1505  →  6 clusters, ARI=0.7999
  Iter  7: percentile=  45.3, quantile=0.1199  →  6 clusters, ARI=0.7249
  Iter  8: percentile=  51.5, quantile=0.1545  →  5 clusters, ARI=0.8376
  Iter  9: percentile=  47.2, quantile=0.1589  →  6 clusters, ARI=0.8363
  Iter 10: percentile=  53.1, quantile=0.1006  →  7 clusters, ARI=0.6321
  Iter 11: percentile=  53.1, quantile=0.1698  →  4 clusters, ARI=0.7635
  Iter 12: percentile=  48.4, quantile=0.1155  →  7 clusters, ARI=0.5928
  Iter 13: percentile=  54.6, quantile=0.1337  →  5 clusters, ARI=0.5605
  Iter 14: percentile=  45.9, quantile=0.1097  →  8 clusters, ARI=0.5841
  Iter 15: percentile=  53.5, quantile=0.1604  →  6 clusters, ARI=0.7004
  Iter 16: percentile=  53.1, quantile=0.1730  →  6 clusters, ARI=0.6961
  Iter 17: percentile=  50.4, quantile=0.1973  →  6 clusters, ARI=0.4647
  Iter 18: percentile=  48.8, quantile=0.1552  →  6 clusters, ARI=0.5564
  Iter 19: percentile=  53.3, quantile=0.1619  →  6 clusters, ARI=0.7181
  Iter 20: percentile=  53.6, quantile=0.1577  →  6 clusters, ARI=0.5240

Best result: percentile=51.5, quantile=0.1545
  Clusters: 5, ARI: 0.8376

### Details

============================================================
PRE-STEP: Training Stage 1 model (shared across iterations)
============================================================
Extracted 2933 non-overlapping windows
Coverage: 87990/88000 frames (100.0%)
Training on fixed windows...
Epoch 10/300 — Loss: 0.711504
Epoch 20/300 — Loss: 0.695327
Epoch 30/300 — Loss: 0.684892
Epoch 40/300 — Loss: 0.462302
Epoch 50/300 — Loss: 0.427336
Epoch 60/300 — Loss: 0.400934
Epoch 70/300 — Loss: 0.388043
Epoch 80/300 — Loss: 0.368783
Epoch 90/300 — Loss: 0.352015
Epoch 100/300 — Loss: 0.328391
Epoch 110/300 — Loss: 0.382890
Epoch 120/300 — Loss: 0.320481
Epoch 130/300 — Loss: 0.304422
Epoch 140/300 — Loss: 0.415599
Epoch 150/300 — Loss: 0.374583
Epoch 160/300 — Loss: 0.299625
Epoch 170/300 — Loss: 0.422328
Epoch 180/300 — Loss: 0.292994
Epoch 190/300 — Loss: 0.381980
Epoch 200/300 — Loss: 0.282455
Epoch 210/300 — Loss: 0.376320
Epoch 220/300 — Loss: 0.280611
Epoch 230/300 — Loss: 0.401120
Epoch 240/300 — Loss: 0.275292
Epoch 250/300 — Loss: 0.410095
Epoch 260/300 — Loss: 0.293160
Epoch 270/300 — Loss: 0.324765
Epoch 280/300 — Loss: 0.270139
Epoch 290/300 — Loss: 0.333271
Epoch 300/300 — Loss: 0.268110

============================================================
PRE-STEP: Computing reconstruction loss signal (shared)
============================================================
Computed loss at 17594 positions
Loss stats — mean: 0.3138, std: 0.4036, max: 5.4347

============================================================
ITERATION 1/20  |  percentile=51.4, quantile=0.103
============================================================
Found 1841 transitions
Mean bout duration: 1.59s
Bout duration looks plausible
Created 1064 variable-length windows from 1842 segments
Window lengths — min: 30, max: 365, mean: 68.5
Retraining on variable-length windows...
Epoch 10/300 — Loss: 0.820757
Epoch 20/300 — Loss: 0.810221
Epoch 30/300 — Loss: 0.796956
Epoch 40/300 — Loss: 0.802842
Epoch 50/300 — Loss: 0.790301
Epoch 60/300 — Loss: 0.787845
Epoch 70/300 — Loss: 0.778309
Epoch 80/300 — Loss: 0.485667
Epoch 90/300 — Loss: 0.461270
Epoch 100/300 — Loss: 0.446026
Epoch 110/300 — Loss: 0.437166
Epoch 120/300 — Loss: 0.437784
Epoch 130/300 — Loss: 0.437282
Epoch 140/300 — Loss: 0.428869
Epoch 150/300 — Loss: 0.436859
Epoch 160/300 — Loss: 0.425791
Epoch 170/300 — Loss: 0.423833
Epoch 180/300 — Loss: 0.422949
Epoch 190/300 — Loss: 0.423655
Epoch 200/300 — Loss: 0.431154
Epoch 210/300 — Loss: 0.423143
Epoch 220/300 — Loss: 0.417439
Epoch 230/300 — Loss: 0.415291
Epoch 240/300 — Loss: 0.419355
Epoch 250/300 — Loss: 0.418001
Epoch 260/300 — Loss: 0.413705
Epoch 270/300 — Loss: 0.416915
Epoch 280/300 — Loss: 0.410385
Epoch 290/300 — Loss: 0.408643
Epoch 300/300 — Loss: 0.405377
Latents shape: (1064, 16)
Running UMAP...
c:\Users\fbai_\anaconda3\envs\condaenv312\Lib\site-packages\umap\umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
UMAP done. Shape: (1064, 2)
Estimated bandwidth: 0.1400
Found 7 clusters
Cluster sizes: [252 145 171 180 141 111  64]
Silhouette score: 0.8847
  → 7 clusters, ARI=0.4829
  ** NEW BEST (ARI=0.4829) **

============================================================
ITERATION 2/20  |  percentile=47.8, quantile=0.122
============================================================
Found 2017 transitions
Mean bout duration: 1.45s
Bout duration looks plausible
Created 1084 variable-length windows from 2018 segments
Window lengths — min: 30, max: 250, mean: 64.5
Retraining on variable-length windows...
Epoch 10/300 — Loss: 0.753743
Epoch 20/300 — Loss: 0.748617
Epoch 30/300 — Loss: 0.742303
Epoch 40/300 — Loss: 0.736412
Epoch 50/300 — Loss: 0.727155
Epoch 60/300 — Loss: 0.424244
Epoch 70/300 — Loss: 0.402098
Epoch 80/300 — Loss: 0.395056
Epoch 90/300 — Loss: 0.388514
Epoch 100/300 — Loss: 0.385952
Epoch 110/300 — Loss: 0.383294
Epoch 120/300 — Loss: 0.384070
Epoch 130/300 — Loss: 0.381542
Epoch 140/300 — Loss: 0.380386
Epoch 150/300 — Loss: 0.379136
Epoch 160/300 — Loss: 0.395307
Epoch 170/300 — Loss: 0.377541
Epoch 180/300 — Loss: 0.377150
Epoch 190/300 — Loss: 0.377692
Epoch 200/300 — Loss: 0.378066
Epoch 210/300 — Loss: 0.374365
Epoch 220/300 — Loss: 0.372815
Epoch 230/300 — Loss: 0.371270
Epoch 240/300 — Loss: 0.369215
Epoch 250/300 — Loss: 0.368623
Epoch 260/300 — Loss: 0.367092
Epoch 270/300 — Loss: 0.365415
Epoch 280/300 — Loss: 0.362869
Epoch 290/300 — Loss: 0.363616
Epoch 300/300 — Loss: 0.362811
Latents shape: (1084, 16)
Running UMAP...
c:\Users\fbai_\anaconda3\envs\condaenv312\Lib\site-packages\umap\umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
UMAP done. Shape: (1084, 2)
Estimated bandwidth: 0.2951
Found 5 clusters
Cluster sizes: [579 191 117 129  68]
Silhouette score: 0.9251
  → 5 clusters, ARI=0.5669
  ** NEW BEST (ARI=0.5669) **

============================================================
ITERATION 3/20  |  percentile=52.4, quantile=0.168
============================================================
Found 1781 transitions
Mean bout duration: 1.64s
Bout duration looks plausible
Created 1042 variable-length windows from 1782 segments
Window lengths — min: 30, max: 405, mean: 70.6
Retraining on variable-length windows...
Epoch 10/300 — Loss: 0.827306
Epoch 20/300 — Loss: 0.822134
Epoch 30/300 — Loss: 0.820712
Epoch 40/300 — Loss: 0.811883
Epoch 50/300 — Loss: 0.806282
Epoch 60/300 — Loss: 0.506116
Epoch 70/300 — Loss: 0.468329
Epoch 80/300 — Loss: 0.461460
Epoch 90/300 — Loss: 0.454349
Epoch 100/300 — Loss: 0.449017
Epoch 110/300 — Loss: 0.448047
Epoch 120/300 — Loss: 0.444715
Epoch 130/300 — Loss: 0.452002
Epoch 140/300 — Loss: 0.441332
Epoch 150/300 — Loss: 0.439950
Epoch 160/300 — Loss: 0.440520
Epoch 170/300 — Loss: 0.438153
Epoch 180/300 — Loss: 0.438603
Epoch 190/300 — Loss: 0.437586
Epoch 200/300 — Loss: 0.439952
Epoch 210/300 — Loss: 0.436901
Epoch 220/300 — Loss: 0.434813
Epoch 230/300 — Loss: 0.433793
Epoch 240/300 — Loss: 0.435558
Epoch 250/300 — Loss: 0.433580
Epoch 260/300 — Loss: 0.431953
Epoch 270/300 — Loss: 0.429777
Epoch 280/300 — Loss: 0.428828
Epoch 290/300 — Loss: 0.427987
Epoch 300/300 — Loss: 0.434003
Latents shape: (1042, 16)
Running UMAP...
c:\Users\fbai_\anaconda3\envs\condaenv312\Lib\site-packages\umap\umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
UMAP done. Shape: (1042, 2)
Estimated bandwidth: 0.3511
Found 5 clusters
Cluster sizes: [376 351 194  62  59]
Silhouette score: 0.8858
  → 5 clusters, ARI=0.5730
  ** NEW BEST (ARI=0.5730) **

============================================================
ITERATION 4/20  |  percentile=53.9, quantile=0.109
============================================================
Found 1704 transitions
Mean bout duration: 1.72s
Bout duration looks plausible
Created 999 variable-length windows from 1705 segments
Window lengths — min: 30, max: 500, mean: 74.3
Retraining on variable-length windows...
Epoch 10/300 — Loss: 0.842432
Epoch 20/300 — Loss: 0.837338
Epoch 30/300 — Loss: 0.839864
Epoch 40/300 — Loss: 0.828446
Epoch 50/300 — Loss: 0.824262
Epoch 60/300 — Loss: 0.611331
Epoch 70/300 — Loss: 0.487110
Epoch 80/300 — Loss: 0.455987
Epoch 90/300 — Loss: 0.446291
Epoch 100/300 — Loss: 0.445840
Epoch 110/300 — Loss: 0.449558
Epoch 120/300 — Loss: 0.450824
Epoch 130/300 — Loss: 0.441020
Epoch 140/300 — Loss: 0.440298
Epoch 150/300 — Loss: 0.433776
Epoch 160/300 — Loss: 0.433849
Epoch 170/300 — Loss: 0.435999
Epoch 180/300 — Loss: 0.433428
Epoch 190/300 — Loss: 0.432538
Epoch 200/300 — Loss: 0.431387
Epoch 210/300 — Loss: 0.432919
Epoch 220/300 — Loss: 0.432363
Epoch 230/300 — Loss: 0.427577
Epoch 240/300 — Loss: 0.428779
Epoch 250/300 — Loss: 0.427058
Epoch 260/300 — Loss: 0.422890
Epoch 270/300 — Loss: 0.423136
Epoch 280/300 — Loss: 0.421225
Epoch 290/300 — Loss: 0.433893
Epoch 300/300 — Loss: 0.426949
Latents shape: (999, 16)
Running UMAP...
c:\Users\fbai_\anaconda3\envs\condaenv312\Lib\site-packages\umap\umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
UMAP done. Shape: (999, 2)
Estimated bandwidth: 0.2217
Found 11 clusters
Cluster sizes: [267 225 197  66  62  58  41   7  33  23  20]
Silhouette score: 0.9296
  → 11 clusters, ARI=0.6784
  ** NEW BEST (ARI=0.6784) **

============================================================
ITERATION 5/20  |  percentile=49.2, quantile=0.103
============================================================
Found 1951 transitions
Mean bout duration: 1.50s
Bout duration looks plausible
Created 1081 variable-length windows from 1952 segments
Window lengths — min: 30, max: 250, mean: 65.8
Retraining on variable-length windows...
Epoch 10/300 — Loss: 0.772671
Epoch 20/300 — Loss: 0.765534
Epoch 30/300 — Loss: 0.761430
Epoch 40/300 — Loss: 0.756906
Epoch 50/300 — Loss: 0.698569
Epoch 60/300 — Loss: 0.438089
Epoch 70/300 — Loss: 0.421967
Epoch 80/300 — Loss: 0.420300
Epoch 90/300 — Loss: 0.416251
Epoch 100/300 — Loss: 0.413824
Epoch 110/300 — Loss: 0.413436
Epoch 120/300 — Loss: 0.412071
Epoch 130/300 — Loss: 0.410478
Epoch 140/300 — Loss: 0.409172
Epoch 150/300 — Loss: 0.406379
Epoch 160/300 — Loss: 0.405369
Epoch 170/300 — Loss: 0.404437
Epoch 180/300 — Loss: 0.401644
Epoch 190/300 — Loss: 0.400252
Epoch 200/300 — Loss: 0.401485
Epoch 210/300 — Loss: 0.398995
Epoch 220/300 — Loss: 0.397873
Epoch 230/300 — Loss: 0.398492
Epoch 240/300 — Loss: 0.397551
Epoch 250/300 — Loss: 0.399032
Epoch 260/300 — Loss: 0.395321
Epoch 270/300 — Loss: 0.394795
Epoch 280/300 — Loss: 0.393779
Epoch 290/300 — Loss: 0.394374
Epoch 300/300 — Loss: 0.390863
Latents shape: (1081, 16)
Running UMAP...
c:\Users\fbai_\anaconda3\envs\condaenv312\Lib\site-packages\umap\umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
UMAP done. Shape: (1081, 2)
Estimated bandwidth: 0.2368
Found 8 clusters
Cluster sizes: [301 191 191 107  73  97  62  59]
Silhouette score: 0.8761
  → 8 clusters, ARI=0.6581

============================================================
ITERATION 6/20  |  percentile=47.2, quantile=0.151
============================================================
Found 2040 transitions
Mean bout duration: 1.44s
Bout duration looks plausible
Created 1084 variable-length windows from 2041 segments
Window lengths — min: 30, max: 250, mean: 64.1
Retraining on variable-length windows...
Epoch 10/300 — Loss: 0.745163
Epoch 20/300 — Loss: 0.739979
Epoch 30/300 — Loss: 0.734235
Epoch 40/300 — Loss: 0.731099
Epoch 50/300 — Loss: 0.724934
Epoch 60/300 — Loss: 0.434156
Epoch 70/300 — Loss: 0.408801
Epoch 80/300 — Loss: 0.391328
Epoch 90/300 — Loss: 0.392553
Epoch 100/300 — Loss: 0.409741
Epoch 110/300 — Loss: 0.387551
Epoch 120/300 — Loss: 0.381699
Epoch 130/300 — Loss: 0.379063
Epoch 140/300 — Loss: 0.376264
Epoch 150/300 — Loss: 0.374771
Epoch 160/300 — Loss: 0.373953
Epoch 170/300 — Loss: 0.375956
Epoch 180/300 — Loss: 0.373029
Epoch 190/300 — Loss: 0.372601
Epoch 200/300 — Loss: 0.383823
Epoch 210/300 — Loss: 0.380194
Epoch 220/300 — Loss: 0.374378
Epoch 230/300 — Loss: 0.371488
Epoch 240/300 — Loss: 0.371363
Epoch 250/300 — Loss: 0.369912
Epoch 260/300 — Loss: 0.369823
Epoch 270/300 — Loss: 0.370999
Epoch 280/300 — Loss: 0.367940
Epoch 290/300 — Loss: 0.367910
Epoch 300/300 — Loss: 0.367964
Latents shape: (1084, 16)
Running UMAP...
c:\Users\fbai_\anaconda3\envs\condaenv312\Lib\site-packages\umap\umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
UMAP done. Shape: (1084, 2)
Estimated bandwidth: 0.3585
Found 6 clusters
Cluster sizes: [353 225 191 122 106  87]
Silhouette score: 0.9275
  → 6 clusters, ARI=0.7999
  ** NEW BEST (ARI=0.7999) **

============================================================
ITERATION 7/20  |  percentile=45.3, quantile=0.120
============================================================
Found 2112 transitions
Mean bout duration: 1.39s
Bout duration looks plausible
Created 1088 variable-length windows from 2113 segments
Window lengths — min: 30, max: 250, mean: 62.7
Retraining on variable-length windows...
Epoch 10/300 — Loss: 0.718948
Epoch 20/300 — Loss: 0.713542
Epoch 30/300 — Loss: 0.708541
Epoch 40/300 — Loss: 0.703525
Epoch 50/300 — Loss: 0.459130
Epoch 60/300 — Loss: 0.393380
Epoch 70/300 — Loss: 0.370488
Epoch 80/300 — Loss: 0.365064
Epoch 90/300 — Loss: 0.358939
Epoch 100/300 — Loss: 0.355956
Epoch 110/300 — Loss: 0.354410
Epoch 120/300 — Loss: 0.354439
Epoch 130/300 — Loss: 0.350888
Epoch 140/300 — Loss: 0.349971
Epoch 150/300 — Loss: 0.359191
Epoch 160/300 — Loss: 0.347638
Epoch 170/300 — Loss: 0.346043
Epoch 180/300 — Loss: 0.346413
Epoch 190/300 — Loss: 0.344692
Epoch 200/300 — Loss: 0.344600
Epoch 210/300 — Loss: 0.343626
Epoch 220/300 — Loss: 0.343083
Epoch 230/300 — Loss: 0.342441
Epoch 240/300 — Loss: 0.340779
Epoch 250/300 — Loss: 0.340381
Epoch 260/300 — Loss: 0.338537
Epoch 270/300 — Loss: 0.337181
Epoch 280/300 — Loss: 0.336627
Epoch 290/300 — Loss: 0.336249
Epoch 300/300 — Loss: 0.336690
Latents shape: (1088, 16)
Running UMAP...
c:\Users\fbai_\anaconda3\envs\condaenv312\Lib\site-packages\umap\umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
UMAP done. Shape: (1088, 2)
Estimated bandwidth: 0.2043
Found 6 clusters
Cluster sizes: [332 251 254 106  78  67]
Silhouette score: 0.9066
  → 6 clusters, ARI=0.7249

============================================================
ITERATION 8/20  |  percentile=51.5, quantile=0.154
============================================================
Found 1834 transitions
Mean bout duration: 1.60s
Bout duration looks plausible
Created 1064 variable-length windows from 1835 segments
Window lengths — min: 30, max: 365, mean: 68.6
Retraining on variable-length windows...
Epoch 10/300 — Loss: 0.810487
Epoch 20/300 — Loss: 0.803697
Epoch 30/300 — Loss: 0.809300
Epoch 40/300 — Loss: 0.799419
Epoch 50/300 — Loss: 0.794804
Epoch 60/300 — Loss: 0.485671
Epoch 70/300 — Loss: 0.460152
Epoch 80/300 — Loss: 0.450190
Epoch 90/300 — Loss: 0.444458
Epoch 100/300 — Loss: 0.452625
Epoch 110/300 — Loss: 0.434450
Epoch 120/300 — Loss: 0.439944
Epoch 130/300 — Loss: 0.435637
Epoch 140/300 — Loss: 0.431293
Epoch 150/300 — Loss: 0.433976
Epoch 160/300 — Loss: 0.447323
Epoch 170/300 — Loss: 0.440769
Epoch 180/300 — Loss: 0.430673
Epoch 190/300 — Loss: 0.427525
Epoch 200/300 — Loss: 0.431337
Epoch 210/300 — Loss: 0.427722
Epoch 220/300 — Loss: 0.425021
Epoch 230/300 — Loss: 0.422750
Epoch 240/300 — Loss: 0.424183
Epoch 250/300 — Loss: 0.423503
Epoch 260/300 — Loss: 0.425576
Epoch 270/300 — Loss: 0.426883
Epoch 280/300 — Loss: 0.424435
Epoch 290/300 — Loss: 0.417345
Epoch 300/300 — Loss: 0.423441
Latents shape: (1064, 16)
Running UMAP...
c:\Users\fbai_\anaconda3\envs\condaenv312\Lib\site-packages\umap\umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
UMAP done. Shape: (1064, 2)
Estimated bandwidth: 0.2808
Found 5 clusters
Cluster sizes: [325 190 259 226  64]
Silhouette score: 0.9415
  → 5 clusters, ARI=0.8376
  ** NEW BEST (ARI=0.8376) **

============================================================
ITERATION 9/20  |  percentile=47.2, quantile=0.159
============================================================
Found 2038 transitions
Mean bout duration: 1.44s
Bout duration looks plausible
Created 1083 variable-length windows from 2039 segments
Window lengths — min: 30, max: 250, mean: 64.2
Retraining on variable-length windows...
Epoch 10/300 — Loss: 0.744567
Epoch 20/300 — Loss: 0.741069
Epoch 30/300 — Loss: 0.734789
Epoch 40/300 — Loss: 0.729190
Epoch 50/300 — Loss: 0.722630
Epoch 60/300 — Loss: 0.420918
Epoch 70/300 — Loss: 0.434900
Epoch 80/300 — Loss: 0.393556
Epoch 90/300 — Loss: 0.386101
Epoch 100/300 — Loss: 0.383055
Epoch 110/300 — Loss: 0.379903
Epoch 120/300 — Loss: 0.379889
Epoch 130/300 — Loss: 0.372593
Epoch 140/300 — Loss: 0.372447
Epoch 150/300 — Loss: 0.369297
Epoch 160/300 — Loss: 0.369989
Epoch 170/300 — Loss: 0.367933
Epoch 180/300 — Loss: 0.367838
Epoch 190/300 — Loss: 0.366720
Epoch 200/300 — Loss: 0.366532
Epoch 210/300 — Loss: 0.364027
Epoch 220/300 — Loss: 0.362437
Epoch 230/300 — Loss: 0.362899
Epoch 240/300 — Loss: 0.360717
Epoch 250/300 — Loss: 0.359749
Epoch 260/300 — Loss: 0.358270
Epoch 270/300 — Loss: 0.359051
Epoch 280/300 — Loss: 0.357701
Epoch 290/300 — Loss: 0.366261
Epoch 300/300 — Loss: 0.357248
Latents shape: (1083, 16)
Running UMAP...
c:\Users\fbai_\anaconda3\envs\condaenv312\Lib\site-packages\umap\umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
UMAP done. Shape: (1083, 2)
Estimated bandwidth: 0.2865
Found 6 clusters
Cluster sizes: [352 191 250 114 113  63]
Silhouette score: 0.9448
  → 6 clusters, ARI=0.8363

============================================================
ITERATION 10/20  |  percentile=53.1, quantile=0.101
============================================================
Found 1739 transitions
Mean bout duration: 1.68s
Bout duration looks plausible
Created 1020 variable-length windows from 1740 segments
Window lengths — min: 30, max: 405, mean: 72.5
Retraining on variable-length windows...
Epoch 10/300 — Loss: 0.840150
Epoch 20/300 — Loss: 0.833973
Epoch 30/300 — Loss: 0.829680
Epoch 40/300 — Loss: 0.826184
Epoch 50/300 — Loss: 0.822719
Epoch 60/300 — Loss: 0.820338
Epoch 70/300 — Loss: 0.774875
Epoch 80/300 — Loss: 0.489755
Epoch 90/300 — Loss: 0.475183
Epoch 100/300 — Loss: 0.468079
Epoch 110/300 — Loss: 0.462497
Epoch 120/300 — Loss: 0.563824
Epoch 130/300 — Loss: 0.570998
Epoch 140/300 — Loss: 0.475158
Epoch 150/300 — Loss: 0.460896
Epoch 160/300 — Loss: 0.457721
Epoch 170/300 — Loss: 0.455072
Epoch 180/300 — Loss: 0.453788
Epoch 190/300 — Loss: 0.453100
Epoch 200/300 — Loss: 0.452527
Epoch 210/300 — Loss: 0.451355
Epoch 220/300 — Loss: 0.450545
Epoch 230/300 — Loss: 0.449163
Epoch 240/300 — Loss: 0.447454
Epoch 250/300 — Loss: 0.447004
Epoch 260/300 — Loss: 0.445596
Epoch 270/300 — Loss: 0.484065
Epoch 280/300 — Loss: 0.449955
Epoch 290/300 — Loss: 0.447325
Epoch 300/300 — Loss: 0.440743
Latents shape: (1020, 16)
Running UMAP...
c:\Users\fbai_\anaconda3\envs\condaenv312\Lib\site-packages\umap\umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
UMAP done. Shape: (1020, 2)
Estimated bandwidth: 0.2281
Found 7 clusters
Cluster sizes: [285 190 282  76  66  63  58]
Silhouette score: 0.9117
  → 7 clusters, ARI=0.6321

============================================================
ITERATION 11/20  |  percentile=53.1, quantile=0.170
============================================================
Found 1740 transitions
Mean bout duration: 1.68s
Bout duration looks plausible
Created 1021 variable-length windows from 1741 segments
Window lengths — min: 30, max: 405, mean: 72.4
Retraining on variable-length windows...
Epoch 10/300 — Loss: 0.838302
Epoch 20/300 — Loss: 0.831099
Epoch 30/300 — Loss: 0.825820
Epoch 40/300 — Loss: 0.822820
Epoch 50/300 — Loss: 0.817816
Epoch 60/300 — Loss: 0.489121
Epoch 70/300 — Loss: 0.461627
Epoch 80/300 — Loss: 0.452329
Epoch 90/300 — Loss: 0.448415
Epoch 100/300 — Loss: 0.447097
Epoch 110/300 — Loss: 0.448574
Epoch 120/300 — Loss: 0.443010
Epoch 130/300 — Loss: 0.442348
Epoch 140/300 — Loss: 0.440705
Epoch 150/300 — Loss: 0.439936
Epoch 160/300 — Loss: 0.441071
Epoch 170/300 — Loss: 0.438743
Epoch 180/300 — Loss: 0.436492
Epoch 190/300 — Loss: 0.437439
Epoch 200/300 — Loss: 0.435097
Epoch 210/300 — Loss: 0.433468
Epoch 220/300 — Loss: 0.431690
Epoch 230/300 — Loss: 0.431470
Epoch 240/300 — Loss: 0.428771
Epoch 250/300 — Loss: 0.433421
Epoch 260/300 — Loss: 0.428228
Epoch 270/300 — Loss: 0.427436
Epoch 280/300 — Loss: 0.428378
Epoch 290/300 — Loss: 0.424372
Epoch 300/300 — Loss: 0.424335
Latents shape: (1021, 16)
Running UMAP...
c:\Users\fbai_\anaconda3\envs\condaenv312\Lib\site-packages\umap\umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
UMAP done. Shape: (1021, 2)
Estimated bandwidth: 0.4318
Found 4 clusters
Cluster sizes: [365 260 249 147]
Silhouette score: 0.8357
  → 4 clusters, ARI=0.7635

============================================================
ITERATION 12/20  |  percentile=48.4, quantile=0.116
============================================================
Found 1986 transitions
Mean bout duration: 1.48s
Bout duration looks plausible
Created 1080 variable-length windows from 1987 segments
Window lengths — min: 30, max: 250, mean: 65.2
Retraining on variable-length windows...
Epoch 10/300 — Loss: 0.762220
Epoch 20/300 — Loss: 0.755873
Epoch 30/300 — Loss: 0.753925
Epoch 40/300 — Loss: 0.750155
Epoch 50/300 — Loss: 0.744600
Epoch 60/300 — Loss: 0.742098
Epoch 70/300 — Loss: 0.737094
Epoch 80/300 — Loss: 0.489943
Epoch 90/300 — Loss: 0.418709
Epoch 100/300 — Loss: 0.402959
Epoch 110/300 — Loss: 0.395166
Epoch 120/300 — Loss: 0.392322
Epoch 130/300 — Loss: 0.389840
Epoch 140/300 — Loss: 0.397635
Epoch 150/300 — Loss: 0.388709
Epoch 160/300 — Loss: 0.385421
Epoch 170/300 — Loss: 0.383408
Epoch 180/300 — Loss: 0.384801
Epoch 190/300 — Loss: 0.386479
Epoch 200/300 — Loss: 0.383516
Epoch 210/300 — Loss: 0.382893
Epoch 220/300 — Loss: 0.382178
Epoch 230/300 — Loss: 0.380799
Epoch 240/300 — Loss: 0.380542
Epoch 250/300 — Loss: 0.380743
Epoch 260/300 — Loss: 0.377038
Epoch 270/300 — Loss: 0.376754
Epoch 280/300 — Loss: 0.376305
Epoch 290/300 — Loss: 0.376271
Epoch 300/300 — Loss: 0.414637
Latents shape: (1080, 16)
Running UMAP...
c:\Users\fbai_\anaconda3\envs\condaenv312\Lib\site-packages\umap\umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
UMAP done. Shape: (1080, 2)
Estimated bandwidth: 0.2555
Found 7 clusters
Cluster sizes: [377 226 163 118  74  69  53]
Silhouette score: 0.8608
  → 7 clusters, ARI=0.5928

============================================================
ITERATION 13/20  |  percentile=54.6, quantile=0.134
============================================================
Found 1679 transitions
Mean bout duration: 1.74s
Bout duration looks plausible
Created 989 variable-length windows from 1680 segments
Window lengths — min: 30, max: 500, mean: 75.4
Retraining on variable-length windows...
Epoch 10/300 — Loss: 0.848420
Epoch 20/300 — Loss: 0.840790
Epoch 30/300 — Loss: 0.833481
Epoch 40/300 — Loss: 0.829983
Epoch 50/300 — Loss: 0.825478
Epoch 60/300 — Loss: 0.530014
Epoch 70/300 — Loss: 0.475754
Epoch 80/300 — Loss: 0.454779
Epoch 90/300 — Loss: 0.448548
Epoch 100/300 — Loss: 0.447021
Epoch 110/300 — Loss: 0.444716
Epoch 120/300 — Loss: 0.444630
Epoch 130/300 — Loss: 0.441711
Epoch 140/300 — Loss: 0.442690
Epoch 150/300 — Loss: 0.437496
Epoch 160/300 — Loss: 0.435192
Epoch 170/300 — Loss: 0.436652
Epoch 180/300 — Loss: 0.432675
Epoch 190/300 — Loss: 0.431306
Epoch 200/300 — Loss: 0.432007
Epoch 210/300 — Loss: 0.429204
Epoch 220/300 — Loss: 0.427829
Epoch 230/300 — Loss: 0.428376
Epoch 240/300 — Loss: 0.425129
Epoch 250/300 — Loss: 0.425244
Epoch 260/300 — Loss: 0.423653
Epoch 270/300 — Loss: 0.421890
Epoch 280/300 — Loss: 0.431308
Epoch 290/300 — Loss: 0.421458
Epoch 300/300 — Loss: 0.421413
Latents shape: (989, 16)
Running UMAP...
c:\Users\fbai_\anaconda3\envs\condaenv312\Lib\site-packages\umap\umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
UMAP done. Shape: (989, 2)
Estimated bandwidth: 0.2242
Found 5 clusters
Cluster sizes: [483 189 141 112  64]
Silhouette score: 0.8638
  → 5 clusters, ARI=0.5605

============================================================
ITERATION 14/20  |  percentile=45.9, quantile=0.110
============================================================
Found 2086 transitions
Mean bout duration: 1.41s
Bout duration looks plausible
Created 1084 variable-length windows from 2087 segments
Window lengths — min: 30, max: 250, mean: 63.4
Retraining on variable-length windows...
Epoch 10/300 — Loss: 0.727986
Epoch 20/300 — Loss: 0.723659
Epoch 30/300 — Loss: 0.718747
Epoch 40/300 — Loss: 0.714110
Epoch 50/300 — Loss: 0.710304
Epoch 60/300 — Loss: 0.707950
Epoch 70/300 — Loss: 0.691608
Epoch 80/300 — Loss: 0.391117
Epoch 90/300 — Loss: 0.374939
Epoch 100/300 — Loss: 0.369735
Epoch 110/300 — Loss: 0.361781
Epoch 120/300 — Loss: 0.360436
Epoch 130/300 — Loss: 0.358272
Epoch 140/300 — Loss: 0.357413
Epoch 150/300 — Loss: 0.355657
Epoch 160/300 — Loss: 0.356839
Epoch 170/300 — Loss: 0.367722
Epoch 180/300 — Loss: 0.350917
Epoch 190/300 — Loss: 0.348743
Epoch 200/300 — Loss: 0.348474
Epoch 210/300 — Loss: 0.347284
Epoch 220/300 — Loss: 0.346055
Epoch 230/300 — Loss: 0.345356
Epoch 240/300 — Loss: 0.345535
Epoch 250/300 — Loss: 0.341714
Epoch 260/300 — Loss: 0.341245
Epoch 270/300 — Loss: 0.342948
Epoch 280/300 — Loss: 0.340050
Epoch 290/300 — Loss: 0.338148
Epoch 300/300 — Loss: 0.337608
Latents shape: (1084, 16)
Running UMAP...
c:\Users\fbai_\anaconda3\envs\condaenv312\Lib\site-packages\umap\umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
UMAP done. Shape: (1084, 2)
Estimated bandwidth: 0.2614
Found 8 clusters
Cluster sizes: [255 251 201 113 105  59  50  50]
Silhouette score: 0.8910
  → 8 clusters, ARI=0.5841

============================================================
ITERATION 15/20  |  percentile=53.5, quantile=0.160
============================================================
Found 1726 transitions
Mean bout duration: 1.70s
Bout duration looks plausible
Created 1009 variable-length windows from 1727 segments
Window lengths — min: 30, max: 500, mean: 73.3
Retraining on variable-length windows...
Epoch 10/300 — Loss: 0.837957
Epoch 20/300 — Loss: 0.828790
Epoch 30/300 — Loss: 0.825930
Epoch 40/300 — Loss: 0.813386
Epoch 50/300 — Loss: 0.483173
Epoch 60/300 — Loss: 0.465211
Epoch 70/300 — Loss: 0.460025
Epoch 80/300 — Loss: 0.453068
Epoch 90/300 — Loss: 0.448280
Epoch 100/300 — Loss: 0.445764
Epoch 110/300 — Loss: 0.442979
Epoch 120/300 — Loss: 0.450155
Epoch 130/300 — Loss: 0.441668
Epoch 140/300 — Loss: 0.453844
Epoch 150/300 — Loss: 0.439976
Epoch 160/300 — Loss: 0.439400
Epoch 170/300 — Loss: 0.438061
Epoch 180/300 — Loss: 0.438472
Epoch 190/300 — Loss: 0.436081
Epoch 200/300 — Loss: 0.436407
Epoch 210/300 — Loss: 0.432374
Epoch 220/300 — Loss: 0.426769
Epoch 230/300 — Loss: 0.429649
Epoch 240/300 — Loss: 0.428344
Epoch 250/300 — Loss: 0.427854
Epoch 260/300 — Loss: 0.424456
Epoch 270/300 — Loss: 0.440066
Epoch 280/300 — Loss: 0.424697
Epoch 290/300 — Loss: 0.423032
Epoch 300/300 — Loss: 0.420871
Latents shape: (1009, 16)
Running UMAP...
c:\Users\fbai_\anaconda3\envs\condaenv312\Lib\site-packages\umap\umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
UMAP done. Shape: (1009, 2)
Estimated bandwidth: 0.2788
Found 6 clusters
Cluster sizes: [318 258 190 119  62  62]
Silhouette score: 0.9532
  → 6 clusters, ARI=0.7004

============================================================
ITERATION 16/20  |  percentile=53.1, quantile=0.173
============================================================
Found 1739 transitions
Mean bout duration: 1.68s
Bout duration looks plausible
Created 1020 variable-length windows from 1740 segments
Window lengths — min: 30, max: 405, mean: 72.5
Retraining on variable-length windows...
Epoch 10/300 — Loss: 0.837589
Epoch 20/300 — Loss: 0.833585
Epoch 30/300 — Loss: 0.829736
Epoch 40/300 — Loss: 0.825068
Epoch 50/300 — Loss: 0.820314
Epoch 60/300 — Loss: 0.509049
Epoch 70/300 — Loss: 0.476592
Epoch 80/300 — Loss: 0.461625
Epoch 90/300 — Loss: 0.452153
Epoch 100/300 — Loss: 0.449308
Epoch 110/300 — Loss: 0.459905
Epoch 120/300 — Loss: 0.441994
Epoch 130/300 — Loss: 0.440620
Epoch 140/300 — Loss: 0.438928
Epoch 150/300 — Loss: 0.439947
Epoch 160/300 — Loss: 0.438124
Epoch 170/300 — Loss: 0.436197
Epoch 180/300 — Loss: 0.435294
Epoch 190/300 — Loss: 0.437174
Epoch 200/300 — Loss: 0.432083
Epoch 210/300 — Loss: 0.431359
Epoch 220/300 — Loss: 0.429838
Epoch 230/300 — Loss: 0.436803
Epoch 240/300 — Loss: 0.431299
Epoch 250/300 — Loss: 0.428155
Epoch 260/300 — Loss: 0.428004
Epoch 270/300 — Loss: 0.426242
Epoch 280/300 — Loss: 0.427228
Epoch 290/300 — Loss: 0.425821
Epoch 300/300 — Loss: 0.425321
Latents shape: (1020, 16)
Running UMAP...
c:\Users\fbai_\anaconda3\envs\condaenv312\Lib\site-packages\umap\umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
UMAP done. Shape: (1020, 2)
Estimated bandwidth: 0.4239
Found 6 clusters
Cluster sizes: [211 277 182 138 103 109]
Silhouette score: 0.8730
  → 6 clusters, ARI=0.6961

============================================================
ITERATION 17/20  |  percentile=50.4, quantile=0.197
============================================================
Found 1901 transitions
Mean bout duration: 1.54s
Bout duration looks plausible
Created 1077 variable-length windows from 1902 segments
Window lengths — min: 30, max: 250, mean: 66.8
Retraining on variable-length windows...
Epoch 10/300 — Loss: 0.791037
Epoch 20/300 — Loss: 0.787015
Epoch 30/300 — Loss: 0.777722
Epoch 40/300 — Loss: 0.775903
Epoch 50/300 — Loss: 0.772034
Epoch 60/300 — Loss: 0.709037
Epoch 70/300 — Loss: 0.453590
Epoch 80/300 — Loss: 0.436762
Epoch 90/300 — Loss: 0.434339
Epoch 100/300 — Loss: 0.428481
Epoch 110/300 — Loss: 0.425505
Epoch 120/300 — Loss: 0.422516
Epoch 130/300 — Loss: 0.421565
Epoch 140/300 — Loss: 0.416168
Epoch 150/300 — Loss: 0.413285
Epoch 160/300 — Loss: 0.413065
Epoch 170/300 — Loss: 0.410198
Epoch 180/300 — Loss: 0.412145
Epoch 190/300 — Loss: 0.407288
Epoch 200/300 — Loss: 0.410032
Epoch 210/300 — Loss: 0.406184
Epoch 220/300 — Loss: 0.405887
Epoch 230/300 — Loss: 0.414898
Epoch 240/300 — Loss: 0.405715
Epoch 250/300 — Loss: 0.405038
Epoch 260/300 — Loss: 0.405256
Epoch 270/300 — Loss: 0.403521
Epoch 280/300 — Loss: 0.400743
Epoch 290/300 — Loss: 0.402089
Epoch 300/300 — Loss: 0.400943
Latents shape: (1077, 16)
Running UMAP...
c:\Users\fbai_\anaconda3\envs\condaenv312\Lib\site-packages\umap\umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
UMAP done. Shape: (1077, 2)
Estimated bandwidth: 0.4273
Found 6 clusters
Cluster sizes: [511 303  75  66  63  59]
Silhouette score: 0.9195
  → 6 clusters, ARI=0.4647

============================================================
ITERATION 18/20  |  percentile=48.8, quantile=0.155
============================================================
Found 1970 transitions
Mean bout duration: 1.49s
Bout duration looks plausible
Created 1080 variable-length windows from 1971 segments
Window lengths — min: 30, max: 250, mean: 65.5
Retraining on variable-length windows...
Epoch 10/300 — Loss: 0.766208
Epoch 20/300 — Loss: 0.761662
Epoch 30/300 — Loss: 0.758728
Epoch 40/300 — Loss: 0.751742
Epoch 50/300 — Loss: 0.748101
Epoch 60/300 — Loss: 0.731984
Epoch 70/300 — Loss: 0.438748
Epoch 80/300 — Loss: 0.422214
Epoch 90/300 — Loss: 0.413214
Epoch 100/300 — Loss: 0.409540
Epoch 110/300 — Loss: 0.405384
Epoch 120/300 — Loss: 0.402756
Epoch 130/300 — Loss: 0.402243
Epoch 140/300 — Loss: 0.396570
Epoch 150/300 — Loss: 0.395582
Epoch 160/300 — Loss: 0.396248
Epoch 170/300 — Loss: 0.391931
Epoch 180/300 — Loss: 0.391097
Epoch 190/300 — Loss: 0.389150
Epoch 200/300 — Loss: 0.387024
Epoch 210/300 — Loss: 0.387537
Epoch 220/300 — Loss: 0.385150
Epoch 230/300 — Loss: 0.382454
Epoch 240/300 — Loss: 0.382254
Epoch 250/300 — Loss: 0.379572
Epoch 260/300 — Loss: 0.379852
Epoch 270/300 — Loss: 0.379110
Epoch 280/300 — Loss: 0.383502
Epoch 290/300 — Loss: 0.376486
Epoch 300/300 — Loss: 0.375428
Latents shape: (1080, 16)
Running UMAP...
c:\Users\fbai_\anaconda3\envs\condaenv312\Lib\site-packages\umap\umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
UMAP done. Shape: (1080, 2)
Estimated bandwidth: 0.3846
Found 6 clusters
Cluster sizes: [340 365 125 114  73  63]
Silhouette score: 0.7933
  → 6 clusters, ARI=0.5564

============================================================
ITERATION 19/20  |  percentile=53.3, quantile=0.162
============================================================
Found 1731 transitions
Mean bout duration: 1.69s
Bout duration looks plausible
Created 1014 variable-length windows from 1732 segments
Window lengths — min: 30, max: 500, mean: 72.9
Retraining on variable-length windows...
Epoch 10/300 — Loss: 0.840673
Epoch 20/300 — Loss: 0.834986
Epoch 30/300 — Loss: 0.827627
Epoch 40/300 — Loss: 0.818956
Epoch 50/300 — Loss: 0.492132
Epoch 60/300 — Loss: 0.758266
Epoch 70/300 — Loss: 0.457483
Epoch 80/300 — Loss: 0.453469
Epoch 90/300 — Loss: 0.450226
Epoch 100/300 — Loss: 0.449344
Epoch 110/300 — Loss: 0.447387
Epoch 120/300 — Loss: 0.445510
Epoch 130/300 — Loss: 0.446728
Epoch 140/300 — Loss: 0.444826
Epoch 150/300 — Loss: 0.442884
Epoch 160/300 — Loss: 0.442205
Epoch 170/300 — Loss: 0.590021
Epoch 180/300 — Loss: 0.445772
Epoch 190/300 — Loss: 0.443071
Epoch 200/300 — Loss: 0.441635
Epoch 210/300 — Loss: 0.440303
Epoch 220/300 — Loss: 0.438007
Epoch 230/300 — Loss: 0.439861
Epoch 240/300 — Loss: 0.436666
Epoch 250/300 — Loss: 0.437244
Epoch 260/300 — Loss: 0.433806
Epoch 270/300 — Loss: 0.434797
Epoch 280/300 — Loss: 0.612272
Epoch 290/300 — Loss: 0.439485
Epoch 300/300 — Loss: 0.431961
Latents shape: (1014, 16)
Running UMAP...
c:\Users\fbai_\anaconda3\envs\condaenv312\Lib\site-packages\umap\umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
UMAP done. Shape: (1014, 2)
Estimated bandwidth: 0.3782
Found 6 clusters
Cluster sizes: [280 257 251 110  72  44]
Silhouette score: 0.8351
  → 6 clusters, ARI=0.7181

============================================================
ITERATION 20/20  |  percentile=53.6, quantile=0.158
============================================================
Found 1717 transitions
Mean bout duration: 1.71s
Bout duration looks plausible
Created 1006 variable-length windows from 1718 segments
Window lengths — min: 30, max: 500, mean: 73.7
Retraining on variable-length windows...
Epoch 10/300 — Loss: 0.839093
Epoch 20/300 — Loss: 0.839577
Epoch 30/300 — Loss: 0.832582
Epoch 40/300 — Loss: 0.826332
Epoch 50/300 — Loss: 0.541633
Epoch 60/300 — Loss: 0.478023
Epoch 70/300 — Loss: 0.472241
Epoch 80/300 — Loss: 0.451117
Epoch 90/300 — Loss: 0.451679
Epoch 100/300 — Loss: 0.446958
Epoch 110/300 — Loss: 0.448459
Epoch 120/300 — Loss: 0.447035
Epoch 130/300 — Loss: 0.443813
Epoch 140/300 — Loss: 0.444656
Epoch 150/300 — Loss: 0.444962
Epoch 160/300 — Loss: 0.445770
Epoch 170/300 — Loss: 0.441804
Epoch 180/300 — Loss: 0.440602
Epoch 190/300 — Loss: 0.438802
Epoch 200/300 — Loss: 0.437321
Epoch 210/300 — Loss: 0.434681
Epoch 220/300 — Loss: 0.436235
Epoch 230/300 — Loss: 0.431775
Epoch 240/300 — Loss: 0.433567
Epoch 250/300 — Loss: 0.430324
Epoch 260/300 — Loss: 0.430561
Epoch 270/300 — Loss: 0.428327
Epoch 280/300 — Loss: 0.424862
Epoch 290/300 — Loss: 0.426019
Epoch 300/300 — Loss: 0.420129
Latents shape: (1006, 16)
Running UMAP...
c:\Users\fbai_\anaconda3\envs\condaenv312\Lib\site-packages\umap\umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
UMAP done. Shape: (1006, 2)
Estimated bandwidth: 0.2705
Found 6 clusters
Cluster sizes: [495 184 127  63  75  62]
Silhouette score: 0.9232
  → 6 clusters, ARI=0.5240