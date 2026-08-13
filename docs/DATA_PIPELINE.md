# EvoSeg Data Pipeline (Pixel-LLM / Sa2VA format)

All raw data lives OUTSIDE the repo under
`/9950backfile/chenjiahui/evo_artifacts/datasets/` (never committed).

## Image referring segmentation
`pixel_llm_data/ref_seg/{refcoco,refcoco+,refcocog}/`
- `instances.json` (COCO format) + `refs(unc|umd).p` (refer format)
- images via symlink `coco2014/train2014 -> evo_artifacts/datasets/coco2014/train2014`
- built by `projects/evoseg/tools/build_pixel_llm_refcoco.py` from local HF parquet
- sizes: refcoco 42,404 / refcocoplus 42,278 / refcocog 42,226 expressions

## gRefCOCO (GRES; faithfulness)
`pixel_llm_data/grefcoco/` (target-present only, finetune format, 19,979 entries)
- built by `build_pixel_llm_grefcoco.py`; 14,464 no-target refs kept for the
  faithful GRES dataset (custom, later stage)

## ReasonSeg
`pixel_llm_data/reason_seg/` (finetune format, 239 samples)
- built by `build_pixel_llm_reasonseg.py` (HF parquet -> jpg + polygons)

## Video RVOS (Sa2VA03RefVOS)
| dataset | expression_file | mask_file | image_folder | n_videos |
|---|---|---|---|---|
| MeViS | mevis_v2/train/meta_expressions_v2.json | mevis_v2/train/mask_dict.json | mevis_v2/train/JPEGImages | 1,662 |
| Ref-YT-VOS | ref_youtube_vos/extracted/meta_expressions/train/meta_expressions.json | pixel_llm_data/ref_youtube_vos/mask_dict.pkl | ref_youtube_vos/extracted/train/JPEGImages | 3,471 |
| Ryvos | pixel_llm_data/ryvos/meta_expressions.json | pixel_llm_data/ryvos/mask_dict.json | ref_youtube_vos/extracted/train/JPEGImages | 3,437 |
- builders: `build_pixel_llm_refytvos.py` (indexed-annotation RLE), `build_pixel_llm_ryvos.py`
- note: Sa2VA03RefVOS does NOT accept `serialize_data` kwarg

## VQA / anti-forgetting
`pixel_llm_data/llava/`
- `llava_instruct_150k.json` + `llava_images/` (81,479 symlinks -> COCO2017/train2017)
- LLaVADataset, 157,712 conversations; free (images already present)
- optional: LLaVA-1.5-665K (needs vg/gqa/ocr_vqa/textvqa, ~100GB download)

## Training configs
- SFT stage 1 (RefCOCO/+/g): `projects/sa2va/configs/evoseg/sa2va_qwen3_4b_refcoco_sft.py`
