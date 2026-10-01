# Full-cohort experiment reports

These aggregate reports contain no original image paths. Training used all 10,360 admitted unique images, frozen ImageNet ResNet50 and EfficientNet-B0 encoders, and separately trained heads for four label systems. `dataset-audit.json` records duplicate removal and group splits. The individual reports and `ensemble-metrics.json` contain measured held-out scores and confusion matrices. Checkpoint hashes bind these reports to the local model artifacts.

Ensemble balanced accuracy: Bethesda **32.5%**, SIPaKMeD **91.9%**, Herlev **69.0%**, custom binary **98.6%**. Bethesda is weak and Herlev is limited. The higher morphology/binary scores do not imply patient cancer-risk performance. Patient-level independence, transformed-duplicate removal, calibration, external validation, clinical fusion, and future-risk prediction have not been established.

`inference-smoke.json` is a functionality check on one held-out image per task, including actual mistakes. It is not an accuracy estimate or production latency benchmark. Model weights are available locally in `artifacts/models/` and bundled in `artifacts/cervisight-models.zip`; they are not committed to Git.
