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

### Round 2b (2026-10-12): RTMPose-style and RT-DETR-based candidates, checked code AND weights

| Part | What we use | Code licence | Weights licence / terms | Source |
|---|---|---|---|---|
| RTMPose-style net | `src/gro/fretnet/rtmpose.py`: CSPNeXt backbone and RTMCC head (GAU + SimCC) re-implemented in plain PyTorch with MMDet's parameter names; no MMPose / mmcv / mmengine code is imported or copied | this repo's own code (the design follows the Apache-2.0 MMPose / MMDet papers and configs) | — | [MMPose (Apache-2.0)](https://github.com/open-mmlab/mmpose/blob/main/LICENSE), [RTMPose project](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) |
| CSPNeXt-m ImageNet init | `cspnext-m_8xb256-rsb-a1-600e_in1k-ecb3bbd9.pth` (md5 `13fdc36df67cbf319d345383c3b1e39b`), backbone keys only, loaded strictly | — | Released in the MMDetection model zoo (`download.openmmlab.com`), which is distributed under the repo's Apache-2.0; no separate weights licence is stated. Trained on ImageNet-1k: same ImageNet caveat as below | [MMDet RTMDet configs](https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet), [MMDet LICENSE](https://github.com/open-mmlab/mmdetection/blob/main/LICENSE) |
| RTMPose body checkpoints (checked, **not used**) | e.g. `rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth` | Apache-2.0 | Model zoo under Apache-2.0, but trained on Body7 = AI Challenger, COCO, CrowdPose, MPII, sub-JHMDB, Halpe, PoseTrack18. Several of these (AI Challenger, PoseTrack18, JHMDB) are released for non-commercial research only, so we use the ImageNet CSPNeXt instead | [RTMPose model zoo](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) |
| RT-DETR model code | HF `transformers` `RTDetrModel` (backbone ResNet-50-vd + hybrid encoder; decoder unused) | Apache-2.0 (transformers) | — | [transformers LICENSE](https://github.com/huggingface/transformers/blob/main/LICENSE), [lyuwenyu/RT-DETR (Apache-2.0)](https://github.com/lyuwenyu/RT-DETR) |
| RT-DETR weights | `PekingU/rtdetr_r50vd` (HF hub), COCO 2017 detection; its ResNet-50-vd was initialised from PaddleClas ImageNet weights (Apache-2.0) | — | Model card licence: `apache-2.0`. COCO annotations are CC-BY-4.0; the COCO images are Flickr photos under their own (mostly CC) licences. Same unsettled training-data question as ImageNet | [model card](https://huggingface.co/PekingU/rtdetr_r50vd) |
| New runtime dep | `transformers` (Apache-2.0), `safetensors` (Apache-2.0); only for the RT-DETR candidate | Apache-2.0 | — | — |

MMPose itself was not installed: its mmcv dependency has no wheels for the PC's torch 2.11 / cu128. The re-implementation
loads the MMDet checkpoint with `strict=True`, so the architecture matches the published one.

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

## Two-stage numbering (v7 experiments, 2026-10-12)

`src/gro/fretnet/numbering.py` and `scripts/train_numbering.py` use torchvision ResNet-18
(`ResNet18_Weights.IMAGENET1K_V1`, `resnet18-f37072fd.pth`) with our own spatial pooling/classification head,
rectification, augmentation and label reader. Stage 1 is the existing permissive FretNet checkpoint.
No Ultralytics imports or weights are used in this path. Code licence: torchvision BSD-3-Clause;
PyTorch BSD-3-Clause; OpenCV Apache-2.0; our new code is original to this repository.
The ImageNet weights retain the dataset-terms caveat recorded above; a BSD code licence alone does not
establish unrestricted rights to the pretraining images.
Sources checked: [torchvision licence](https://github.com/pytorch/vision/blob/main/LICENSE),
[ResNet-18 weight documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).

The smaller v7 variants use torchvision `mobilenet_v3_small` and
`MobileNet_V3_Small_Weights.IMAGENET1K_V1` (BSD-3-Clause code; same pretraining-data caveat).
The optional per-wire sequence head uses two original `torch.nn.TransformerEncoder` layers
and predicts absolute wire indices; it adds no external model code or weights.
[MobileNetV3 Small weight documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html).

## Landmark numbering (v8 experiment, 2026-10-12)

`src/gro/fretnet/landmark.py` uses the same torchvision `mobilenet_v3_small` architecture
and `MobileNet_V3_Small_Weights.IMAGENET1K_V1` initialization documented for v7 above.
The 1D localization head, explicit absent class, rectification target transforms, and
confidence-weighted wire voting are original repository code. Training and evaluation
(`scripts/train_landmark.py`, `scripts/landmark_eval.py`) add no third-party model code
or weights. Torchvision/PyTorch code is BSD-3-Clause and OpenCV is Apache-2.0; the same
ImageNet pretrained-weight/dataset-terms caveat above applies. Stage 1 and the optional
v7 offset-classifier vote retain their existing recorded provenance. No Ultralytics
code or weights are used by these training or inference paths.
