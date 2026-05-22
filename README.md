# SAIS OCR Inference

比赛推理入口已经改为：
- YOLO 检测模型负责找拓片中的单字框
- EfficientNet-B0 + ArcFace 分类模型负责识别框内古文字

## 输入输出

- 输入目录：`/saisdata`
- 输出文件：`/saisresult/prediction.json`
- 输出格式：

```json
{
  "图片ID": [
    {
      "bbox": [x, y, w, h],
      "text": "字"
    }
  ]
}
```

## 仓库内必须保留的文件

- `src/run_inference.py`
- `run.sh`
- `Dockerfile`
- `requirements.txt`
- `models/detector_best.pt`
- `models/classifier_best.pt`

## 模型文件说明

- `models/detector_best.pt`
  来自 YOLO 检测训练结果
- `models/classifier_best.pt`
  来自 EfficientNet-B0 + ArcFace 分类训练结果
  分类器的 `label_map` 直接使用 Unicode 字符作为类别名，推理时会直接输出这些字符

## Docker

当前 Dockerfile 默认基础镜像为：

```dockerfile
ARG BASE_IMAGE=nvcr.io/nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu22.04
```

如果后续你已经把 CUDA 基础镜像同步到了自己的阿里云 ACR，也可以在构建时覆盖这个 `BASE_IMAGE`。

## 本地测试

```bash
INPUT_DIR=./local_test/input \
OUTPUT_FILE=./local_test/result/prediction.json \
DETECTOR_WEIGHTS=./models/detector_best.pt \
CLASSIFIER_WEIGHTS=./models/classifier_best.pt \
python3 src/run_inference.py
```

当前默认推理参数已经调到一组更优的本地验证配置：
- `DETECT_CONF=0.12`
- `DETECT_IOU=0.45`
- `BOX_EXPAND_RATIO=0.00`
- `CLASSIFY_MIN_PROB=0.20`

如需覆盖，可通过环境变量传入。

## 说明

当前提交版分类器基于比赛训练集裁字重训，输出类别直接对应 Unicode 古文字字符。

实验与提交分数记录见：`reports/experiment_tracking.csv`
