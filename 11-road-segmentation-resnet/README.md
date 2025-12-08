# 道路检测语义分割（ResNet骨干）

本文档给出一套工业级的道路场景语义分割方案，包括算法设计、训练与部署实践，以及可直接运行的 PyTorch 参考实现（`train.py`）。模型基于 ResNet-50 骨干的 DeepLabV3+，兼顾精度与部署效率。

## 1. 业务目标
- **场景**：车道线/道路区域分割，支持白天、夜晚、雨雪等多场景。
- **指标**：mIoU ≥ 0.85（典型路面数据），帧率 ≥ 20 FPS（TensorRT FP16，1080p）。
- **上线形态**：推理服务（ONNX/TensorRT）或端侧嵌入式（CUDA/ARM+NPU）。

## 2. 数据与标注
- **数据来源**：自采/众包/开源（BDD100K、Cityscapes 作为预训练）。
- **标注规范**：PNG 单通道 mask，255 为 ignore，0/1/2… 为语义 ID，背景必须占用 0。
- **划分策略**：8:1:1（train/val/test），同路段/天气尽量分散到不同子集，减少泄漏。
- **数据清洗**：
  - 剔除过曝/模糊/大雨雪不可用样本。
  - mask 连通域检查，确保道路区域闭合，忽略标注 ≤ 50 px 的碎片。

## 3. 数据增广（在线）
- 几何：随机水平翻转、随机尺度（0.5–2.0）、随机裁剪到 1024×512。
- 光照：颜色抖动（亮度/对比度/饱和度/色相），随机高斯模糊。
- 噪声/缺陷：随机遮挡（方框）、gamma 变换，夜晚/雨天合成（可选）。
- 归一化：Imagenet 均值方差，mask 不做归一化；保持 image/mask 同步变换。

## 4. 模型设计
- **骨干**：ResNet-50（可换 ResNet-101 获取更高精度）。
- **解码器**：DeepLabV3+（ASPP 空洞卷积 + 低层特征融合）。
- **输出**：`num_classes` 语义通道，默认 `ignore_index=255`。
- **预训练**：ImageNet 预训练；可用 Cityscapes 进行语义 warmup，再在道路数据上 finetune。
- **损失**：加权交叉熵 + Dice（可选）。示例代码默认加权 CE，便于处理类不平衡。
- **正则化**：Weight decay、Dropout（ASPP/decoder），同步 BN（多卡训练时建议）。

## 5. 训练策略
- **优化器**：SGD（momentum=0.9, weight_decay=1e-4）。
- **学习率**：poly decay（base 0.01，power 0.9，warmup 1000 iter）。
- **批大小**：单卡 4–8（依据显存），多卡使用 DistributedDataParallel。
- **训练时长**：30–80k iter；每 1k iter 评估一次 mIoU 并保存最优权重。
- **混合精度**：`torch.cuda.amp` 提升吞吐，减少显存。
- **类权重**：道路/背景占比悬殊时可设置 `class_weights`（示例代码支持）。

## 6. 推理与后处理
- TTA：多尺度 + 水平翻转（可选）。
- 后处理：条件随机场（CRF）或形态学平滑用于边界优化；部署端可改为轻量化的高斯滤波。
- 输出格式：PNG mask，或转换为多边形用于后续规划模块。

## 7. 工程结构
```
11-road-segmentation-resnet/
├── README.md          # 本设计文档（当前文件）
├── requirements.txt   # 训练依赖
└── train.py           # 可运行的训练/评估/导出脚本
```

## 8. 快速开始
1. 安装依赖：`pip install -r requirements.txt`
2. 准备数据：目录结构如下
   ```
   dataset/
     train/images/*.jpg
     train/masks/*.png
     val/images/*.jpg
     val/masks/*.png
   ```
3. 训练示例（单卡）：
   ```bash
   python train.py \
     --data-root dataset \
     --num-classes 2 \
     --save-dir runs/exp1
   ```
4. 评估：`python train.py --data-root dataset --num-classes 2 --save-dir runs/exp1 --eval-only --checkpoint runs/exp1/best.pt`
5. 导出 ONNX：`python train.py --data-root dataset --num-classes 2 --save-dir runs/exp1 --export-onnx runs/exp1/model.onnx`

## 9. 关键实现亮点
- **严格同步的数据增广**：对图像与 mask 使用相同的随机参数，避免标签漂移。
- **工业化训练循环**：断点重训、最优权重保存、日志与 mIoU 计算齐备。
- **可复用配置**：超参均可通过命令行配置，适配不同显存/数据规模。
- **部署友好**：导出 ONNX → TensorRT；保持算子兼容性（避免动态图层）。

## 10. 后续优化建议
- 使用 **ResNet-101** 或 **Swin-T** 骨干获取更高精度。
- 将 **Dice/Focal** 与 CE 组合，进一步抑制前景/背景不平衡。
- 引入 **蒸馏**（teacher：大模型，student：轻量模型）提升端侧性能。
- 引入 **自动混合精度 + 梯度累积**，在小显存卡上使用较大 batch。
- 结合 **稀疏/量化**（INT8、GPTQ、AWQ）以满足更严苛的延时/功耗需求。

更多细节请查看 `train.py` 源码，其中包含完整的训练、评估与导出流程。
