# 关键实验结果表

日期：2026-05-27

## 说明

- 这份表主要整理“对主线决策有影响”的关键实验。
- 不把所有扫参明细都抄进来，只保留决定过方向的版本。
- 分成三类：
  - 线上提交结果
  - 本地 oldsplit 关键结果
  - 本地 group split 关键结果

## 一、线上提交结果

| 时间 | 版本 | 分数 | TP | FP | FN | 备注 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 2026-05-21 12:09:52 | `paddleocr_baseline` | 0.000904 | 6 | 2102 | 11157 | 原始 PaddleOCR baseline |
| 2026-05-21 17:07:01 | `yolo_plus_hust_pretrain_cls` | 0.001859 | 24 | 14640 | 11139 | YOLO + HUST 预训练分类器，严重域差 |
| 2026-05-22 | `yolo_aug1_plus_unicode_cls_v2` | 0.144921 | 1871 | 12787 | 9292 | 比赛域分类器首次奏效 |
| 2026-05-22 | `yolo_olddet_plus_unicode_cls_v2` | 0.153870 | 1987 | 12677 | 9176 | old detector 比 aug1 更稳 |
| 2026-05-22 21:37:53 | `olddet_newcls_tuned_online_user_report` | 0.190546 | 2616 | 13679 | 8547 | 旧 detector + resumed e12 + 调参 |
| 2026-05-23 19:57:05 | `group_lastpt_online_user_report_v1_inferred` | 0.192495 | 2624 | 13476 | 8539 | group-val classifier 早期线上结果，信息部分推断 |
| 2026-05-24 | `group_lastpt_online_user_report` | 0.195015 | 2985 | 16465 | 8178 | 当前已知线上最好，但 FP 很高 |
| 2026-05-25 21:51:34 | `allboxes_longtailmix_online_user_report_v2` | 0.186351 | 2844 | 16516 | 8319 | allboxes random-v1 + longtailmix v1，过预测明显 |

## 二、本地 oldsplit 关键结果

### 2.1 oldsplit 主线候选

| 日期 | run_name | detector | classifier | F1 | TP | FP | FN | 备注 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 2026-05-24 | `infer_20260524_hustcls_best_oldsplit_afterfix` | YOLOv8m original best | HUST-overlap mix e20 best | 0.428718 | 4804 | 5549 | 7254 | HUST overlap 主线，后续被淘汰 |
| 2026-05-24 | `infer_20260524_groupfixbest_oldsplit_eval` | YOLOv8m original best | group-val fixbest best.pt | 0.478624 | 5413 | 5148 | 6645 | `fixbest` 不如 `last.pt` |
| 2026-05-24 | `infer_20260524_group_lastpt_oldsplit_eval` | YOLOv8m original best | group-val `last.pt` | 0.594925 | 6905 | 4250 | 5153 | group-val classifier 主线确立 |
| 2026-05-25 | `infer_20260525_chron_tail20_oldsplit_eval` | YOLOv8m original best | group-val + HUST035 + Chronicles-tail best | 0.603070 | 6973 | 4094 | 5085 | 在 old detector 上进一步提升 |
| 2026-05-26 | `infer_20260526_yolo1536e60_chrontail_oldsplit` | YOLOv8m 1536 e60 b4 | chron-tail stable classifier | 0.588257 | 6317 | 3102 | 5741 | 1536/e60 续训 detector 没带来收益 |
| 2026-05-26 | `infer_20260526_allboxes_group1536e40_chrontail_oldsplit` | YOLOv8m allboxes group 1536 e40 | chron-tail stable classifier | 0.640822 | 7488 | 3824 | 4570 | 当前本地 oldsplit 最优主线 |
| 2026-05-26 | `infer_20260526_yoloorig_longtailmix_v2_ob_oldsplit` | YOLOv8m original best | longtailmix v2 oracle+bronze | 0.513298 | 5848 | 4880 | 6210 | 新 longtail classifier 在 detector 噪声下退化 |
| 2026-05-27 | `infer_20260527_allboxes_group1536e40_longtailmix_v2_ob` | YOLOv8m allboxes group 1536 e40 | longtailmix v2 oracle+bronze | 0.552376 | 6341 | 4560 | 5717 | 更强 detector 也救不回这版 classifier |

### 2.2 oldsplit 的方向性结论

- `group-val last.pt` 明显优于早期 HUST-overlap mix。
- `chron-tail stable classifier` 比 `group-val last.pt` 再高一档。
- `longtailmix v2 oracle+bronze` 虽然分类 val acc 更高，但端到端 oldsplit 更差。
- `YOLO 1536 e60` 不是更强主线。
- `allboxes group 1536 e40 detector + chron-tail stable classifier` 是当前最强 oldsplit 组合。

## 三、本地 group split 关键结果

| 日期 | run_name | detector | classifier | F1 | TP | FP | FN | 备注 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 2026-05-24 | `infer_20260524_hustcls_groupval` | YOLOv8m original best | HUST-overlap mix e20 best | 0.679297 | 9430 | 4033 | 4871 | 早期 group split 主线 |
| 2026-05-24 | `infer_20260524_groupfixbest_groupval_eval` | YOLOv8m original best | group-val fixbest best.pt | 0.683664 | 9546 | 4079 | 4755 | 略好于 HUST-overlap，但不稳定 |
| 2026-05-24 | `infer_20260524_groupval_lastpt_eval` | YOLOv8m original best | group-val `last.pt` | 0.715928 | 10057 | 3737 | 4244 | 当前已记录的 group split 最优结果 |

### group split 的解读

- `group split` 分数普遍高于 `oldsplit`。
- 这是因为它本身不是线上代理集，而是另一种“无泄漏本地验证口径”。
- 因此：
  - `group split` 适合看模型是否学得更稳
  - `oldsplit` 更适合和历史线上主线保持连续对比
- 当前不能把 `group split` 上的 `0.7159` 直接当成线上预期。

## 四、关键失败实验

| 日期 | run_name | 结果 | 失败原因判断 |
| --- | --- | --- | --- |
| 2026-05-25 | `infer_20260525_rfdetr_oldsplit_base` | 0.376249 | RF-DETR recall 高，但 FP 爆炸，不适合作为当前主线 |
| 2026-05-25 | `infer_20260525_edgecrafter_base` | 0.542381 | EdgeCrafter 端到端弱于当前 YOLO 主线 |
| 2026-05-26 | `infer_20260526_yolo_paddle_rec` | 0.120538 | Paddle rec warm-start 未能适配当前古文字识别任务 |
| 2026-05-26 | `infer_20260526_yolo1536e60_chrontail_oldsplit` | 0.588257 | detector 仅靠更长续训和更大分辨率没有稳定提升 |
| 2026-05-26/27 | `longtailmix v2 oracle+bronze` 两条线 | 0.513298 / 0.552376 | classifier 在 GT crop 上更强，但对 detector 预测噪声不鲁棒 |

## 五、当前结论

- 当前已知线上最佳：
  - `group_lastpt_online_user_report`
  - `F1 = 0.195015`
- 当前本地 oldsplit 最佳：
  - `infer_20260526_allboxes_group1536e40_chrontail_oldsplit`
  - `F1 = 0.640822`
- 当前待线上验证版本：
  - commit `64d66dc`
  - `allboxes group 1536 e40 detector + chron-tail stable classifier`

