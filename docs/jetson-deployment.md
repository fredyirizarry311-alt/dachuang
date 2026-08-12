# Jetson 与 TensorRT 部署

## 目标软件栈

现存项目记录指向 Jetson Orin Nano 与 TensorRT 8.5。仓库中的 `infer_image.py` 使用 `num_bindings`、`binding_is_input` 和 `execute_v2` 等旧式 binding API；TensorRT 10 环境可能需要改为 tensor I/O API。

## 输入输出约束

- 输入：`1 × 3 × 224 × 224`，FP32。
- 图像：BGR 读取后转换 RGB，resize 224×224。
- 归一化：ImageNet mean `(0.485, 0.456, 0.406)`，std `(0.229, 0.224, 0.225)`。
- 输出：五类 logits，顺序必须与 `classes.txt` 一致。

## 文件准备

在 `jetson/` 下放置但不要提交：

```text
ship_risk_model.onnx
classes.txt
calibration_images/
```

校准图必须覆盖五类和实际部署光照，且不得直接用独立测试集做校准。

## 构建 INT8 Engine

```bash
cd jetson
python3 build_ship_risk_int8.py
```

输出：

```text
ship_risk_int8.cache
ship_risk_int8.engine
```

脚本同时设置 INT8 与 FP16 flag，由 TensorRT 为不支持 INT8 的层选择合适精度。

## 单图推理

```bash
python3 infer_image.py example.jpg --engine ship_risk_int8.engine
```

## 必须记录的验收数据

- Jetson 型号、JetPack、CUDA、cuDNN、TensorRT 版本。
- ONNX SHA-256、Engine SHA-256 和构建命令。
- 校准集版本、每类数量和来源隔离策略。
- FP32/FP16/INT8 的模型大小、单帧延迟、吞吐、峰值显存和功耗。
- PyTorch 与 TensorRT 的逐样本输出一致性及精度差异。

这些数据当前未形成可验证的公开记录，因此 v0.1.0 只声明“脚本已实现”，不声明具体速度、功耗或精度提升。

