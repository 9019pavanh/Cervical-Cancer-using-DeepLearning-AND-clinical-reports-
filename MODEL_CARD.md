# CerviSight research model card

## Intended use

Explore cervical cytology image classification and model explanations in a research demonstration. The project does not implement future cancer-risk prediction, clinical report interpretation, trained image–clinical fusion, cancer staging, or a patient-care decision system.

## Architecture

ResNet50 with ImageNet-1K V2 weights and EfficientNet-B0 with ImageNet-1K V1 weights. Each frozen image encoder has separate linear heads for Bethesda, SIPaKMeD, Herlev, and custom binary labels. Cross-entropy uses inverse training-class frequency weights; AdamW trains the heads. Validation loss selects the head checkpoint. Equal-weight averaging of softmax scores forms the ensemble. Both input pipelines resize RGB images to 224×224 and use ImageNet normalization. No backbone fine-tuning, clinical modality, or probabilistic calibration is claimed.

## Local cohort

The supplied five dataset directories contain overlapping collections. The audit admitted 16,429 labeled images, removed 6,069 pixel-identical copies, and retained 10,360 unique images. Zero conflicting exact-duplicate labels were found. 1,933 images were excluded: predominantly whole-field SIPaKMeD views plus an image with no recognized label. Non-image annotations are not training inputs.

| Task | Unique images | Label meaning |
|---|---:|---|
| Bethesda | 2,634 | ASCUS, HSIL, LSIL, NILM, SCC |
| SIPaKMeD | 4,049 | Five original cell morphology types |
| Herlev | 917 | Seven original cytology categories |
| Custom binary | 2,760 | Normal / abnormal labels |

The group split contains 7,268 training, 1,480 validation, and 1,612 test images. These are **approximate 70/15/15 group proportions**; group sizes produce different image proportions. Groups combine filename-based slide proxies with exact-image duplicate links. Patient identities and independence are not verified. Transformed duplicates, source bias, and patient overlap may remain; random holdout scores do not demonstrate prospective or external-site performance.

## Outputs and explanations

Outputs describe the selected task only. Morphology labels are not cancer stages, and an abnormal-image score is not future cancer risk. Model probabilities are uncalibrated research scores. The review flag uses score <0.7 or differing individual model argmax classes; this threshold is a heuristic without clinical validation. Grad-CAM highlights regions influential for the selected output and does not identify lesion boundaries, provide causal evidence, or validate correctness.

## Evaluation artifacts

`artifacts/models/*-metrics.json` records held-out per-class precision/recall/F1, confusion matrices and balanced accuracy. Individual reports include training/validation histories and selected cohort metadata. `ml.evaluate` evaluates equal-weight fusion against the same untouched test split using cached image features. Reports must be read together with the group-split limitations and training mode. Pilot runs are explicitly marked `pilot` and must not be represented as full-cohort experiments.

Full-cohort frozen-feature training completed locally with seed 42. Equal-weight ensemble results:

| Task | Held-out images | Balanced accuracy |
|---|---:|---:|
| Bethesda | 425 | 32.5% |
| SIPaKMeD | 631 | 91.9% |
| Herlev | 139 | 69.0% |
| Custom binary | 417 | 98.6% |

Bethesda performance is weak and Herlev performance is limited. These heads need further model development and cohort validation. Accuracy on SIPaKMeD or the custom binary task does not transfer to Bethesda categories or patient cancer risk. A real-checkpoint API smoke check successfully returned predictions and 224×224 Grad-CAM overlays for one held-out image per task; measured inference plus explanations took 584–761 ms on this local CPU. These four timings are functional measurements, not a production throughput benchmark. Both encoder feature passes plus head training took approximately 23 minutes in total, excluding the audit and environment setup.

## Training provenance and deployment

Model checkpoints include class ordering, architecture, training mode, seed, and a SHA-256 hash of the source manifest. Both ensemble checkpoints must use the same manifest and class definitions. The local training environment uses Python 3.12, PyTorch 2.3.1 CPU and torchvision 0.18.1. Model artifacts and original image data are excluded from Git; model deployment requires a separate artifact transfer. Dataset licenses and rights to redistribute derived weights must be reviewed before release.

Clinical or longitudinal prediction requires a new patient-linked cohort, a defined endpoint, patient-level validation, calibration, and external evaluation. The unpaired clinical CSV retained in the legacy repository is not used by this app.
