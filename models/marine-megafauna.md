# Marine Detect MegaFauna

An experimental upstream YOLOv8 detector for rays, sharks, and turtles.
Source: [Orange / Tēnaka Marine Detect](https://github.com/Orange-OpenSource/marine-detect/tree/587f45184ab0f04ed5a243c2625c441dfb2996a3).

The class labels in checkpoint order are `ray`, `shark`, and `turtle`.
The original file is `MegaFauna.pt`; its SHA-256 is recorded in the catalog.

The local FP32 export accepts `[1, 3, 640, 640]` RGB input and produces one
`[1, 7, 8400]` tensor. It passed ONNX graph validation and the app's CPU
inference check on 2026-11-01. A powerful GPU is recommended for evaluation
on a live stream; GPU performance has not been measured here.

The upstream project focuses on Malaysia/Indo-Pacific environments. Its
published accuracy measurements do not establish accuracy on Manafish ROV
footage. Evaluate this model on the intended environment before relying on it.

Weight redistribution terms remain unconfirmed. No download is hosted here.
See [the licensing notes](../NOTICE.md#marine-detect).
