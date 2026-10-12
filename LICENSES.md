# Licenses of models, weights and third-party code

Two model families are in this repo. They have different licence obligations.

## Model v3 (FretNet): permissive, no Ultralytics

| Part | What | Licence | Source |
|---|---|---|---|
| Network + head code | `src/gro/fretnet/` (ResNet + FPN + per-keypoint heatmaps, windowed soft-argmax, confidence head), written for this repo | this repo's own code (no third-party code copied) | — |
| Training / export / eval code | `scripts/train_fretnet.py`, `scripts/export_fretnet.py`, `scripts/eval_v2.py` (the FretNet path), `gro.infer` | this repo's own code | — |
| Backbone architecture + loader | `torchvision.models.resnet50` | BSD-3-Clause | [torchvision LICENSE](https://github.com/pytorch/vision/blob/main/LICENSE) |
| Backbone init weights | torchvision `ResNet50_Weights.IMAGENET1K_V2` (`resnet50-11ad3fa6.pth`), ImageNet-1k classification | Published with torchvision under BSD-3-Clause | [torchvision models docs](https://pytorch.org/vision/stable/models.html), [weights](https://download.pytorch.org/models/resnet50-11ad3fa6.pth) |
| Trained v3 weights | `runs/fretnet/<name>/best.pt` (PC), CoreML export in the Jam Dashboard app | fine-tuned from the torchvision weights on our own labels; no AGPL code or weights involved | — |
| Runtime deps | PyTorch (BSD-3-Clause), torchvision (BSD-3-Clause), OpenCV (Apache-2.0), NumPy (BSD-3-Clause), coremltools (BSD-3-Clause), onnxruntime (MIT) | permissive | [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE), [OpenCV](https://github.com/opencv/opencv/blob/4.x/LICENSE), [coremltools](https://github.com/apple/coremltools/blob/main/LICENSE.txt), [onnxruntime](https://github.com/microsoft/onnxruntime/blob/main/LICENSE) |

Round 2 (global-context heads, `head="query"` / `SimCCNet` in `src/gro/fretnet/model.py`) adds no third-party code:
the transformer layers are `torch.nn` (BSD-3-Clause). The RTMPose-style SimCC head is our own re-implementation of the
idea (no MMPose code). The new backbone option:

| Part | What | Licence | Source |
|---|---|---|---|
| ConvNeXt-T architecture + loader | `torchvision.models.convnext_tiny` | BSD-3-Clause | [torchvision LICENSE](https://github.com/pytorch/vision/blob/main/LICENSE) |
| ConvNeXt-T init weights | torchvision `ConvNeXt_Tiny_Weights.IMAGENET1K_V1`, ImageNet-1k (trained by torchvision with its own recipe; ConvNeXt V1 itself is MIT) | Published with torchvision under BSD-3-Clause; same ImageNet caveat as below | [torchvision docs](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html), [ConvNeXt (MIT)](https://github.com/facebookresearch/ConvNeXt) |

Options checked for round 2 and not used (licences recorded for later):

| Option | Code | Pretrained weights | Why not used now |
|---|---|---|---|
| MMPose / RTMPose (CSPNeXt) | Apache-2.0 | Model-zoo weights released under the repo's Apache-2.0; CSPNeXt pose weights are trained on 7 public human-pose sets (COCO, AIC, CrowdPose, MPII, ...), which carry their own dataset terms | MMPose needs mmcv, whose prebuilt wheels stop well before the PC's torch 2.11 / cu128 (a source build was not attempted); the SimCC idea is re-implemented on torchvision instead |
| timm ConvNeXt / EfficientNet / RegNet | Apache-2.0 | Per-model: e.g. ConvNeXt V1 MIT; **ConvNeXt V2 ImageNet weights are CC-BY-NC (non-commercial), avoid**; others vary | timm is not installed on the PC; torchvision has the same backbones with BSD-3-Clause weights |
| RT-DETR (lyuwenyu/RT-DETR, HF `PekingU/rtdetr_*`) | Apache-2.0 | Apache-2.0 (COCO / Objects365 trained) | A detector plus keypoint head is a bigger port; its AIFI idea (self-attention on stride-32 tokens only) is used in `head="query"` |
| YOLOX (Megvii) | Apache-2.0 | Apache-2.0 (COCO trained) | Same: per-anchor regression like v1, not needed to test global context |

The v3 training, export and inference paths do not import `ultralytics` and do not load Ultralytics weights. The
data loader reads the YOLO-pose *label text format* produced by `scripts/export_yolo.py`. That is a plain file format;
no Ultralytics code is involved in reading it.

Caveat: torchvision states that pretrained weights "may have their own licenses or terms and conditions derived from
the dataset used for training". The ImageNet-1k terms of access cover non-commercial research use of the *images*.
Whether that reaches weights trained on them is legally unsettled. Every common ImageNet-pretrained backbone carries the
same caveat, including timm's Apache-2.0-tagged ones. If that matters for a release, re-check before shipping.

## Model v1 / v2 (YOLO11-pose): AGPL-3.0

| Part | Licence | Source |
|---|---|---|
| `ultralytics` package (training, export, `TorchBackend`, `scripts/train_yolo.py`, `scripts/eval_model.py`) | AGPL-3.0 (or Ultralytics Enterprise licence) | [ultralytics LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) |
| `yolo11s-pose.pt` / `yolo11n-pose.pt` COCO init weights, and every model fine-tuned from them (v1, v2_*) | AGPL-3.0 | [Ultralytics licensing](https://www.ultralytics.com/license) |

Shipping v1 (for example the CoreML `FretTrackerV1` in the Jam Dashboard desktop app) brings in the AGPL-3.0
obligations for the model: source availability for the work that includes it, including over a network. Alternatively,
an Ultralytics Enterprise licence removes them. Removing v1 from shipped builds leaves only the permissive v3 path.
