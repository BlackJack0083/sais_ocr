# 线上提交历史

日期：2026-05-27

## 说明

- 本文档整理“已经提交到比赛平台”的版本演化。
- 以 [experiment_tracking.csv](/mnt/data/hejiakai/sais_ocr/reports/experiment_tracking.csv:1) 中 `scope=competition_submission` 的记录为主。
- 若某条线上结果来自聊天回传、仓库中缺少完整自动落盘信息，会明确标注为“用户回传”或“信息推断”。

## 已有线上分数的版本

| 时间 | run_name | detector | classifier | 分数 | 核心改动 | 结论 |
| --- | --- | --- | --- | ---: | --- | --- |
| 2026-05-21 12:09:52 | `paddleocr_baseline` | PaddleOCR built-in detector | PaddleOCR built-in recognizer | 0.000904 | 原始 PaddleOCR 端到端基线 | 现代 OCR 直接迁移到古文字任务基本失效 |
| 2026-05-21 17:07:01 | `yolo_plus_hust_pretrain_cls` | YOLO detector initial version | EfficientNet-B0 + ArcFace on HUST-OBC pretraining | 0.001859 | 从 PaddleOCR 切到 YOLO + 二阶段分类 | 有轻微提升，但域差极大，几乎不可用 |
| 2026-05-22 | `yolo_aug1_plus_unicode_cls_v2` | YOLOv8m detector aug1 | Competition-domain Unicode classifier v2 crops | 0.144921 | 分类器改为直接用比赛 XML 裁字训练；detector 用更克制的 aug1 | 第一次进入可用区间，证明“比赛域分类器”是关键 |
| 2026-05-22 | `yolo_olddet_plus_unicode_cls_v2` | YOLOv8m detector original best | Competition-domain Unicode classifier v2 crops | 0.153870 | 保留比赛域分类器，detector 从 aug1 回退到原始 best | old detector 比 aug1 更稳 |
| 2026-05-22 21:37:53 | `olddet_newcls_tuned_online_user_report` | YOLOv8m detector original best | Unicode classifier resumed e12 | 0.190546 | 分类器续训到 e12，并做推理阈值调参 | 前期主线最佳版本，首次到 0.19+ |
| 2026-05-23 19:57:05 | `group_lastpt_online_user_report_v1_inferred` | YOLOv8m detector original best | group-val retrain `last.pt` | 0.192495 | 切到无泄漏 group-val classifier 主线，参数较激进 | 线上略高于 0.1905；该条 detector/classifier 元信息为推断补档 |
| 2026-05-24 | `group_lastpt_online_user_report` | YOLOv8m detector original best | group-val retrain `last.pt` | 0.195015 | 沿用 group-val classifier 主线；打包默认参数为 `det_conf=0.08, det_iou=0.40, det_tta_mode=scale1536, cls_min_prob=0.38, cls_min_margin=0.07` | 当前已知线上最好，但 FP 明显偏高 |
| 2026-05-25 21:51:34 | `allboxes_longtailmix_online_user_report_v2` | YOLOv8m detector allboxes random-v1 | longtail mix v1 best | 0.186351 | detector 数据集改为 allboxes；classifier 改为 long-tail mix | allboxes + longtailmix 这版线上过预测明显，弱于 old detector 主线 |

## 已 push、待线上回分的版本

| 时间 | commit | detector | classifier | 说明 |
| --- | --- | --- | --- | --- |
| 2026-05-27 | `64d66dc` | YOLOv8m allboxes group 1536 e40 best | group-val + HUST035 + Chronicles-tail best | 当前待测新版本；我已把 `models/` 与默认 `DETECT_IMGSZ=1536` 同步到这条线 |

## 阶段总结

### 第一阶段：从 PaddleOCR 迁移到二阶段

- `paddleocr_baseline -> yolo_plus_hust_pretrain_cls`
- 关键变化：
  - 从 PaddleOCR 检测+识别切到 YOLO 检测 + EfficientNet 分类。
  - 分类器最初主要依赖 HUST-OBC。
- 结果：
  - 几乎没有真正解决问题，主要被域差拖垮。

### 第二阶段：比赛域分类器带来第一次质变

- `yolo_plus_hust_pretrain_cls -> yolo_aug1_plus_unicode_cls_v2 -> yolo_olddet_plus_unicode_cls_v2`
- 关键变化：
  - 用比赛 XML 裁字构建 Unicode 分类集。
  - 分类头直接输出 Unicode 字符。
  - detector 尝试 aug1，再回退到更稳的 old detector。
- 结果：
  - 线上从 `0.001859` 提升到 `0.153870`。

### 第三阶段：续训分类器 + 推理阈值调参

- `yolo_olddet_plus_unicode_cls_v2 -> olddet_newcls_tuned_online_user_report`
- 关键变化：
  - 分类器续训到 `e12`。
  - 做了一轮推理阈值调参。
- 结果：
  - 提升到 `0.190546`。

### 第四阶段：无泄漏 group-val 分类器

- `olddet_newcls_tuned_online_user_report -> group_lastpt_online_user_report_v1_inferred -> group_lastpt_online_user_report`
- 关键变化：
  - 分类切分改为 image-group 级别，减少泄漏。
  - 融入 HUST overlap 辅助样本。
  - 推理默认参数切到更高 recall 路线。
- 结果：
  - 线上最好来到 `0.195015`，但 FP 继续偏高。

### 第五阶段：allboxes detector 与 long-tail mix

- `group_lastpt_online_user_report -> allboxes_longtailmix_online_user_report_v2`
- 关键变化：
  - detector 训练时保留所有合法字框，不再只保留 `len(text)==1` 的框。
  - classifier 改成比赛 + HUST + Chronicles 的 long-tail mix。
- 结果：
  - 线上降到 `0.186351`，说明“allboxes 思路 + longtailmix v1”这版并不稳。

## 当前判断

- 已知线上最好版本仍是：
  - `group_lastpt_online_user_report`
  - `F1 = 0.195015`
- 当前最值得观察的新版本是：
  - `64d66dc`
  - `allboxes group 1536 e40 detector + chron-tail stable classifier`

