# 系统架构

## 目标

项目针对海上链路低带宽、高丢包和断续连接条件，把原始感知数据在 Jetson 边缘端压缩为少量、高价值的结构化事件信息。

## 分层设计

### 1. 感知层

- 视频：当前以 MobileNetV4 对采样帧做五分类。
- 音频：计划通过 SenseVoice 获取转写、关键词、情绪或声学事件信息。
- 环境：计划接入烟雾/气体等传感器并生成阈值事件。

### 2. 语义层

当前视觉脚本输出 JSON token，字段包括时间、模态、事件、置信度、重要度和来源。

建议后续统一 schema：

```json
{
  "schema_version": "1.0",
  "event_id": "uuid",
  "timestamp": "2026-08-12T11:00:00+08:00",
  "source_id": "jetson-01/camera-01",
  "modality": "vision",
  "event": "fire",
  "confidence": 0.91,
  "importance": 0.99,
  "location": "cabin-a",
  "payload": {}
}
```

当前代码尚未实现 `schema_version`、`event_id`、ISO 8601 时间或位置字段，因此这些字段属于 v0.2.0 目标，不属于 v0.1.0 已有接口。

### 3. 传输层（待实现）

计划使用 TLV + CRC 封装语义消息，通过 ZeroMQ PUSH/PULL 传输，并配置 HWM、本地缓存、断线重连和复网重发。

### 4. 岸基层（待实现）

岸基端解析和校验消息，融合视觉、音频和传感器事件，提供告警、态势展示和事件追溯。

## 关键一致性约束

- `classes.txt` 的次序、checkpoint 中的 `class_names` 和 TensorRT 输出索引必须一致。
- 训练、PyTorch 推理和 TensorRT 推理必须使用相同的 224×224 resize、RGB 通道和 ImageNet mean/std。
- TensorRT Engine 必须在目标设备和目标软件栈构建。
- 训练集按“源视频/源场景/原图组”划分，避免连续帧或同底图裁剪跨集合泄漏。

