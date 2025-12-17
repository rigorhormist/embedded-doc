"""
使用中文 BERT 进行句子成分（短语）分割的示例脚本。

主要功能：
1. 从 CoNLL 列式数据读取 token 和标签。
2. 使用 BERT 对标签做对齐，并基于 Trainer 完成微调。
3. 支持将训练好的模型用于新句子的句法成分预测。

示例运行：
python chunk_segmentation.py \
  --data-file data/sample.conll \
  --output-dir outputs/demo-checkpoint \
  --num-epochs 1 \
  --demo
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from seqeval.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)


@dataclass
class SentenceExample:
    """保存一个句子对应的 tokens 与标签列表。"""

    tokens: List[str]
    labels: List[str]


class ConllReader:
    """读取 CoNLL 格式的句子成分标注数据。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.sentences: List[SentenceExample] = []
        self.label_list: List[str] = []
        self._read()

    def _read(self) -> None:
        sentences: List[SentenceExample] = []
        current_tokens: List[str] = []
        current_labels: List[str] = []

        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    if current_tokens:
                        sentences.append(
                            SentenceExample(
                                tokens=current_tokens.copy(),
                                labels=current_labels.copy(),
                            )
                        )
                        current_tokens.clear()
                        current_labels.clear()
                    continue
                try:
                    token, label = stripped.split("\t")
                except ValueError as exc:
                    raise ValueError(
                        f"行格式错误：{stripped}，请确保使用 <token>\t<label>"
                    ) from exc
                current_tokens.append(token)
                current_labels.append(label)

        if current_tokens:
            sentences.append(SentenceExample(tokens=current_tokens, labels=current_labels))

        labels = sorted({label for sent in sentences for label in sent.labels})
        if "O" in labels:
            labels.remove("O")
            labels.insert(0, "O")

        self.sentences = sentences
        self.label_list = labels

    def train_test_split(self, test_ratio: float = 0.2) -> Tuple[List[SentenceExample], List[SentenceExample]]:
        split = int(len(self.sentences) * (1 - test_ratio))
        return self.sentences[:split], self.sentences[split:]


class BertChunkingDataset(torch.utils.data.Dataset):
    """将原始 tokens/labels 对齐到 BERT 子词后得到可供训练的数据集。"""

    def __init__(
        self,
        examples: Sequence[SentenceExample],
        tokenizer: AutoTokenizer,
        label_to_id: dict,
    ) -> None:
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id

    def __len__(self) -> int:  # pragma: no cover - 简单包装
        return len(self.examples)

    def __getitem__(self, idx: int):
        example = self.examples[idx]
        tokenized = self.tokenizer(
            example.tokens,
            is_split_into_words=True,
            truncation=True,
            return_offsets_mapping=True,
        )

        labels: List[int] = []
        word_ids = tokenized.word_ids()
        previous_word_id = None
        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)  # 忽略特殊符号
            elif word_id != previous_word_id:
                labels.append(self.label_to_id[example.labels[word_id]])
            else:
                # 子词延用同一个标签，保持 BIO 一致性
                labels.append(self.label_to_id[example.labels[word_id]])
            previous_word_id = word_id

        tokenized["labels"] = labels
        return tokenized


@dataclass
class ChunkingConfig:
    data_file: Path
    output_dir: Path
    model_name: str = "bert-base-chinese"
    num_epochs: int = 3
    learning_rate: float = 5e-5
    batch_size: int = 8
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_steps: int | None = None
    demo: bool = False
    predict_sentence: str | None = None


class ChunkingPipeline:
    def __init__(self, cfg: ChunkingConfig) -> None:
        self.cfg = cfg
        self.reader = ConllReader(cfg.data_file)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.label_to_id = {label: idx for idx, label in enumerate(self.reader.label_list)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}

    def _build_datasets(self):
        train_examples, eval_examples = self.reader.train_test_split(test_ratio=0.25)
        train_dataset = BertChunkingDataset(train_examples, self.tokenizer, self.label_to_id)
        eval_dataset = BertChunkingDataset(eval_examples, self.tokenizer, self.label_to_id)
        return train_dataset, eval_dataset

    def _compute_metrics(self, p):  # Trainer 会传入 EvalPrediction
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_labels: List[List[str]] = []
        true_predictions: List[List[str]] = []

        for prediction, label in zip(predictions, labels):
            preds: List[str] = []
            refs: List[str] = []
            for p_id, l_id in zip(prediction, label):
                if l_id == -100:
                    continue
                preds.append(self.id_to_label[p_id])
                refs.append(self.id_to_label[l_id])
            true_predictions.append(preds)
            true_labels.append(refs)

        return {
            "f1": f1_score(true_labels, true_predictions),
            "accuracy": accuracy_score(true_labels, true_predictions),
        }

    def train(self) -> None:
        train_dataset, eval_dataset = self._build_datasets()

        model = AutoModelForTokenClassification.from_pretrained(
            self.cfg.model_name,
            num_labels=len(self.reader.label_list),
            id2label=self.id_to_label,
            label2id=self.label_to_id,
        )

        args = TrainingArguments(
            output_dir=str(self.cfg.output_dir),
            per_device_train_batch_size=self.cfg.batch_size,
            per_device_eval_batch_size=self.cfg.batch_size,
            learning_rate=self.cfg.learning_rate,
            num_train_epochs=self.cfg.num_epochs,
            weight_decay=self.cfg.weight_decay,
            warmup_ratio=self.cfg.warmup_ratio,
            evaluation_strategy="steps" if len(eval_dataset) > 0 else "no",
            logging_steps=1,
            save_strategy="no",
            max_steps=self.cfg.max_steps,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if len(eval_dataset) > 0 else None,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForTokenClassification(self.tokenizer),
            compute_metrics=self._compute_metrics if len(eval_dataset) > 0 else None,
        )

        trainer.train()

        trainer.save_model(self.cfg.output_dir)
        self.tokenizer.save_pretrained(self.cfg.output_dir)

        if len(eval_dataset) > 0:
            metrics = trainer.evaluate()
            metrics_path = self.cfg.output_dir / "metrics.json"
            metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
            print(f"评估结果已保存：{metrics_path}")

    def predict(self, sentence: str) -> List[Tuple[str, str]]:
        if not self.cfg.output_dir.exists():
            raise FileNotFoundError(
                f"未找到训练好的模型目录：{self.cfg.output_dir}，请先运行训练或指定已存在的模型路径。"
            )

        model = AutoModelForTokenClassification.from_pretrained(self.cfg.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.output_dir)

        tokens = list(sentence)
        encoded = tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
        )
        with torch.no_grad():
            logits = model(**encoded).logits
        predictions = torch.argmax(logits, dim=-1)[0].tolist()

        word_ids = encoded.word_ids(0)
        results: List[Tuple[str, str]] = []
        for token_id, word_id in zip(predictions, word_ids):
            if word_id is None:
                continue
            label = model.config.id2label[token_id]
            results.append((tokens[word_id], label))

        merged = self._merge_sub_tokens(results)
        for token, label in merged:
            print(f"{token}\t{label}")
        return merged

    @staticmethod
    def _merge_sub_tokens(pairs: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
        merged: List[Tuple[str, str]] = []
        for key, group in itertools.groupby(pairs, key=lambda x: x[0]):
            labels = [label for _, label in group]
            most_freq = max(set(labels), key=labels.count)
            merged.append((key, most_freq))
        return merged


def parse_args() -> ChunkingConfig:
    base_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(description="基于 BERT 的句子成分分割")
    parser.add_argument(
        "--data-file",
        type=Path,
        default=base_dir / "data" / "sample.conll",
        help="CoNLL 数据路径",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir / "outputs" / "bert-chunk",
        help="模型输出目录",
    )
    parser.add_argument("--model-name", type=str, default="bert-base-chinese", help="预训练 BERT 名称或路径")
    parser.add_argument("--num-epochs", type=int, default=3, help="训练 epoch 数")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--batch-size", type=int, default=8, help="batch 大小")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="权重衰减系数")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="学习率 warmup 比例")
    parser.add_argument("--max-steps", type=int, default=None, help="可选的最大训练步数（优先于 epoch）")
    parser.add_argument("--demo", action="store_true", help="使用示例数据做快速训练与预测")
    parser.add_argument("--predict", dest="predict_sentence", type=str, default=None, help="仅做推理时输入的句子")
    args = parser.parse_args()
    return ChunkingConfig(**vars(args))


def main() -> None:
    cfg = parse_args()
    pipeline = ChunkingPipeline(cfg)

    if cfg.demo or cfg.predict_sentence is None:
        print("\n===== 开始训练 =====")
        pipeline.train()

    if cfg.predict_sentence:
        print("\n===== 推理 =====")
        pipeline.predict(cfg.predict_sentence)
    elif cfg.demo:
        print("\n===== 示例推理 =====")
        demo_sentence = "请把最新的项目进展发我"
        pipeline.predict(demo_sentence)


if __name__ == "__main__":
    main()
