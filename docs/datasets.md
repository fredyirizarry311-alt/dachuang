# 最终五分类数据说明

## 数据版本

最终磁盘数据快照包含 **56,141 张图片**：

| split | cabin_smoke | external_smoke | fire | normal_ship | water_ingress | 合计 |
|---|---:|---:|---:|---:|---:|---:|
| train | 12,277 | 352 | 23,187 | 8,932 | 240 | 44,988 |
| val | 611 | 90 | 3,757 | 1,506 | 60 | 6,024 |
| test | 505 | 0 | 4,624 | 0 | 0 | 5,129 |
| 合计 | 13,393 | 442 | 31,568 | 10,438 | 300 | 56,141 |

重要限制：当前 `test` 只覆盖 `cabin_smoke` 和 `fire`，不能据此声称完成了独立的五分类测试。训练脚本会在 test 目录存在时使用它，因此正式评估前应补齐五类 test，或显式记录评估只覆盖两个类别。

## 来源映射

| 最终类别 | 主要来源 | 最终数量 | 页面 | 来源关联风险 |
|---|---|---:|---|---|
| cabin_smoke | SmokeBench LQ | 9,975 | https://github.com/ncfjd/SmokeBench | 同监控场景关联风险；LQ/GT 为配准对 |
| cabin_smoke | Indoor Fire and Smoke YOLOv8 的 smoke 框裁剪 | 3,342 | https://www.kaggle.com/datasets/sinchanashivanand/indoor-fire-and-smoke-detection-with-yolov8 | 同原图多框裁剪；部分文件有 MP4/CCTV 痕迹 |
| cabin_smoke | 用户提供微信图片 | 76 | 无公开链接 | 来源授权和视频连续性未知 |
| external_smoke | PyroNear2025 random_smoke | 442 | https://huggingface.co/datasets/pyronear/pyro-sdis | 221 个源图组的原图/增强图对 |
| fire | PyroNear2025 | 26,938 | https://arxiv.org/abs/2402.05349 | 秒级时间戳和视频事件关联，连续/近邻帧风险高 |
| fire | Indoor Fire and Smoke YOLOv8 的 fire 框裁剪 | 3,592 | 同上 Kaggle 页面 | 同原图多框裁剪和视频抽帧风险 |
| fire | Factory Fire Detector | 1,038 | https://universe.roboflow.com/zikai-yu/factory-fire-detector-8mijx | 文件名含 `*_mp4-####`，视频近邻帧风险明确 |
| normal_ship | Game of Deep Learning: Ship datasets | 8,932 | https://www.kaggle.com/datasets/arpitjain007/game-of-deep-learning-ship-datasets | 未发现连续帧字段；含 6,252 train + 2,680 竞赛 test 图 |
| normal_ship | Boat-MNIST/早期验证集 | 1,506 | 原公开页未锁定 | 同一时间戳/底图的多坐标裁剪，场景泄漏风险高 |
| water_ingress | 9 张人工筛选微信原图 + SAM | 300 | 原图无公开链接；SAM: https://github.com/facebookresearch/segment-anything | 300 张派生图仅来自 9 张原图，同源依赖极强 |

## 许可证与再分发

- Indoor Fire and Smoke Kaggle 页面标注 CC0，但仍需检查第三方原始图权利。
- Factory Fire Detector 页面标注 CC BY 4.0。
- PyroNear 关联 Hugging Face 页面标注 Apache-2.0；必须按实际下载版本复核。
- Game of Deep Learning 页面显示数据库/内容条款；需按 ODbL/DbCL 具体条款处理。
- SmokeBench、本地 Boat-MNIST 和用户提供图片的完整许可证据尚未归档。

因此本仓库只记录数据来源，不再分发任何图片。

## 划分建议

下一版数据清洗应以 `source_group` 为单位重新分组划分，而不是逐图片随机划分：

- PyroNear：按视频或事件目录。
- Factory/Indoor：按原视频或裁剪前原图。
- Boat-MNIST：按坐标前的底图 ID。
- external_smoke：原图和增强图必须保持在同一 split。
- water_ingress：按 9 张源图隔离，不能让同一原图增强物跨 split。

