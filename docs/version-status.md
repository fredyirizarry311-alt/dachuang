# 版本与实现状态

## v0.1.0 基线

快照日期：2026-08-12。

| 能力 | 文档状态 | 代码位置 | 可复现性结论 |
|---|---|---|---|
| 五分类模型训练 | 已实现 | `vision/train_mobilenetv4.py` | 可在具备数据集和依赖的 PC/GPU 环境运行 |
| 数据结构检查 | 已实现 | `vision/check_dataset.py` | 可运行 |
| 单图 PyTorch 推理 | 已实现 | `vision/predict_image.py` | 需要兼容 checkpoint |
| 视频抽帧 PyTorch 推理 | 已实现 | `vision/predict_video.py` | 需要兼容 checkpoint 和 OpenCV |
| TensorRT INT8 构建 | 脚本已实现 | `jetson/build_int8.py` | 必须在目标 Jetson、对应 TensorRT 版本复验 |
| 船舶风险模型 INT8 包装 | 已实现 | `jetson/build_ship_risk_int8.py` | 依赖同目录 ONNX 与校准图 |
| TensorRT 单图推理 | 脚本已实现 | `jetson/infer_image.py` | 使用旧式 binding API，TensorRT 10 可能需适配 |
| SenseVoice | 规划 | `audio/README.md` | 未发现本项目可发布推理代码、模型配置或测试记录 |
| TLV/CRC | 规划 | `communication/README.md` | 未发现可发布实现 |
| ZeroMQ | 规划 | `communication/README.md` | 未发现可发布实现 |
| 多模态融合 | 规划 | `docs/roadmap.md` | 未发现可发布实现 |

## 版本更新规则

每次版本更新必须同时完成：

1. 修改 `VERSION`。
2. 修改 `project-version.json` 中的版本、日期和状态列表。
3. 更新本文件中的实现矩阵。
4. 在 `CHANGELOG.md` 添加同版本记录。
5. 运行 `python tools/validate_release.py`。
6. 如果数据版本变化，同时更新 `docs/datasets.md` 的拆分计数和来源。

只有代码、测试证据和文档三者一致时，功能状态才能从“规划”改为“已实现”。Jetson 专属功能还应记录 JetPack、TensorRT、CUDA、设备型号、精度和时延后才能写成“已验证”。

