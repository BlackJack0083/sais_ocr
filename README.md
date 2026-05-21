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
- `models/id_to_chinese.json`

## 模型文件说明

- `models/detector_best.pt`
  来自 YOLO 检测训练结果
- `models/classifier_best.pt`
  来自 EfficientNet-B0 + ArcFace 分类训练结果
- `models/id_to_chinese.json`
  用于把 HUST-OBC 类别 ID 转成最终提交所需的古文字字符

## 本地测试

```bash
INPUT_DIR=./local_test/input \
OUTPUT_FILE=./local_test/result/prediction.json \
DETECTOR_WEIGHTS=./models/detector_best.pt \
CLASSIFIER_WEIGHTS=./models/classifier_best.pt \
ID_TO_CHINESE_FILE=./models/id_to_chinese.json \
python3 src/run_inference.py
```

## 说明

分类模型训练时使用的是 HUST-OBC 的真实类别数 `1588`。其中部分合并类在 HUST-OBC 中对应多个近形字符；当前推理实现会使用该合并类的首个 ID 作为稳定输出字符。
