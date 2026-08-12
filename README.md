# 智航“语义流”：面向船舶弱网的边缘多模态语义通信

本仓库整理自大学生创新训练项目，目标是在 NVIDIA Jetson 边缘端从船舶视频、语音和传感器数据中提取高价值险情语义，只传输结构化语义记录，而不是持续回传完整原始媒体。

当前发布版本为 **v0.1.0（2026-08-12）**。版本状态以 [`project-version.json`](project-version.json) 为机器可读基线，以 [`docs/version-status.md`](docs/version-status.md) 为人工审计基线。

## 当前实现状态

| 模块 | 状态 | 本版本证据 |
|---|---|---|
| MobileNetV4 五分类训练 | 已实现 | `vision/train_mobilenetv4.py` |
| 单图语义 token | 已实现 | `vision/predict_image.py` |
| 视频抽帧语义 token | 已实现 | `vision/predict_video.py` |
| Jetson TensorRT INT8 构建 | 已实现脚本，需在目标 Jetson 复验 | `jetson/build_int8.py` |
| Jetson TensorRT 单图推理 | 已实现脚本，需在目标 Jetson 复验 | `jetson/infer_image.py` |
| SenseVoice 音频语义 | 方案已确定，未发现可发布实现代码 | `audio/README.md` |
| TLV/CRC + ZeroMQ 传输 | 方案已确定，未发现可发布实现代码 | `communication/README.md` |
| 多模态融合与岸基 UI | 待实现 | `docs/roadmap.md` |

> 这里没有把项目计划书中的目标冒充为已完成结果。SenseVoice、ZeroMQ、TLV/CRC 和多模态融合仍明确列为后续版本工作。

## 系统链路

```text
摄像头 ──> MobileNetV4 ──> 视觉语义 token ──┐
麦克风 ──> SenseVoice（待集成）─────────────┤
传感器 ──> 阈值/时序分析（待集成）──────────┤
                                             ├─> 融合（待实现）
                                             └─> TLV + CRC + ZeroMQ（待实现）
                                                     └─> 岸基监控端
```

视觉事件类别固定为：

```text
normal_ship
fire
cabin_smoke
external_smoke
water_ingress
```

## 快速开始：视觉训练与推理

```bash
cd vision
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux: source .venv/bin/activate
pip install -r requirements.txt
python check_dataset.py
python train_mobilenetv4.py --epochs 15 --batch-size 16
python predict_image.py --image path/to/example.jpg
python predict_video.py --video path/to/example.mp4 --seconds 1
```

数据目录结构、当前数据版本及许可证注意事项见 [`docs/datasets.md`](docs/datasets.md)。仓库不包含训练图片、微信原图、模型权重、ONNX 或 TensorRT Engine。

## Jetson 部署

TensorRT 脚本位于 `jetson/`。目标流程是：

```text
PyTorch checkpoint -> ONNX -> FP16/INT8 TensorRT Engine -> 单图/帧推理
```

详细步骤及兼容性限制见 [`docs/jetson-deployment.md`](docs/jetson-deployment.md)。TensorRT Engine 与硬件、JetPack/TensorRT 版本强绑定，不能把其他机器生成的 Engine 当作通用文件提交。

## 语义 token 示例

```json
{
  "time": "00:00:05",
  "modality": "vision",
  "type": "ship_risk",
  "event": "fire",
  "event_cn": "火灾",
  "confidence": 0.91,
  "importance": 0.99,
  "source": "frame_0005.jpg"
}
```

## 文档索引

- [`docs/architecture.md`](docs/architecture.md)：系统架构与接口边界
- [`docs/datasets.md`](docs/datasets.md)：56,141 张最终数据的来源、拆分和风险
- [`docs/jetson-deployment.md`](docs/jetson-deployment.md)：Jetson/TensorRT 部署
- [`docs/version-status.md`](docs/version-status.md)：版本与代码一致性矩阵
- [`docs/roadmap.md`](docs/roadmap.md)：后续版本规划
- [`CHANGELOG.md`](CHANGELOG.md)：版本变更记录
- [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)：第三方数据与软件说明

## 数据与隐私

- 不提交训练集、视频、音频、微信原图或浏览器历史。
- 不提交模型权重、ONNX、TensorRT Engine 或校准缓存。
- 数据集许可证并不统一；使用或再分发前必须逐项复核。
- 当前私有仓库不等于已获得第三方数据再分发许可。

## 许可证

本仓库暂未授予统一的开源许可证。代码版权及后续开源方式需由项目成员与学校确认；第三方数据和依赖继续受各自许可证约束。

