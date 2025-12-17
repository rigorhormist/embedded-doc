# BERT 句子成分分割示例

本示例展示如何基于中文 **BERT** 以及现有标注数据完成简单的句子成分（短语）分割。代码使用 Hugging Face Transformers 的 `Trainer` 接口，准备好 CoNLL 标注数据后即可训练、验证并在新句子上进行预测。

## 环境准备

```bash
pip install -r requirements.txt
```

> 说明：示例依赖 `bert-base-chinese`，会在首次运行时自动下载模型权重。

## 数据格式

数据采用 **CoNLL** 列式格式：每行包含 `token<TAB>label`，句子之间用空行隔开。仓库提供了 `data/sample.conll` 作为最小示例：

```text
# tokenslabels
我B-NP
爱B-VP
自然B-NP
语言I-NP
处理I-NP

今B-NP
天I-NP
的O
天气B-NP
真B-ADJP
好I-ADJP

你B-NP
能B-VP
给B-VP
我B-NP
推荐B-VP
书B-NP
吗O
```

其中标签使用 BIO 方案区分 **名词短语 (NP)**、**动词短语 (VP)** 等句子成分，你可以根据自己的标注体系自由扩展。

## 快速体验

运行以下命令将在示例数据上微调 1 个 epoch，并对一条自定义句子做预测：

```bash
python chunk_segmentation.py \
  --data-file data/sample.conll \
  --output-dir outputs/demo-checkpoint \
  --num-epochs 1 \
  --demo
```

- 训练日志与模型权重保存在 `outputs/demo-checkpoint`。
- 预测结果以 **token-标签** 的形式输出，便于检查分割效果。

## 自定义训练

- 准备自己的 CoNLL 数据并通过 `--data-file` 传入；
- `--model-name` 可切换到其他 BERT 变体，例如 `hfl/chinese-bert-wwm-ext`；
- `--num-epochs`、`--learning-rate`、`--batch-size` 等参数均可在脚本中调整；
- 若已经有训练好的模型，可将 `--output-dir` 指向对应目录，然后仅使用 `--predict` 做推理。

## 推理示例

```bash
python chunk_segmentation.py --predict "请把最新的项目进展发我" --output-dir outputs/demo-checkpoint
```

上述命令会加载 `outputs/demo-checkpoint` 下保存的权重，使用 BERT 做句子成分分割并打印结果。
