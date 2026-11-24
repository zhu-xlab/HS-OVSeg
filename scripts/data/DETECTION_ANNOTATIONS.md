# Detection annotations for Object-level training

DIOR-R, DOTA-v2, and FAIR1M conversion now writes one annotation file for
every prepared image:

```text
<dataset-root>/
├── images/
│   └── <sample-id>.<image-extension>
├── annotations/
│   └── <sample-id>.json
├── classes.json
├── dataset.json
└── splits/
```

Each `annotations/<sample-id>.json` uses
`ovrs-detection-annotations-v1` and retains per-instance dataset class IDs,
class text, the horizontal `bbox_xyxy`, the original `oriented_bbox_xy` when
available, difficulty, and the original instance ID.

Training reads each image's annotation on demand and preserves every instance
region in memory. Native SAM3 logits are pooled inside each instance with
parameter-free masked LogSumExp attention. Correct prompts receive region BCE;
wrong in-batch prompts receive region-negative BCE. Low-weight outside/global
negative BCE limits spatial leakage. Generated SAM masks are never read.

For an existing converted dataset, export the per-image annotations without
loading SAM or regenerating masks:

```bash
python scripts/data/convert_dior_r_to_semantic.py --boxes-only
python scripts/data/convert_dota_v2_to_semantic.py --boxes-only
python scripts/data/convert_fair1m_to_semantic.py --boxes-only
```

Pass `--manifest`, `--config`, or `--output-dir` when the defaults do not match
the prepared dataset. The command also retains `detection_boxes.jsonl` as a
legacy compatibility file, while `dataset.json` is updated to prefer
`annotations/`.
