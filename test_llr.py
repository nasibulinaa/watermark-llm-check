#!/usr/bin/env python3
"""Эквивалентность по-токенного LLR (detect.py) референс-реализации
openstamp/src/llr.py (length_normalized_llr).

Референс работает с полными logits (B, T, V): log_softmax -> gather по
label-позициям -> сумма / длина. detect.py имеет только по-токенные
logprobs сервера (т.е. уже собранные log_softmax) и суммирует их,
пропуская первый текст-токен (labels = input_ids[:, 2:]). Тест
проверяет, что результаты совпадают.
"""

import os
import sys

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "openstamp", "src"))
from llr import length_normalized_llr  # noqa: E402


def main():
    torch.manual_seed(0)
    B, k, V = 2, 40, 1000
    T = k + 1  # позиция 0 — BOS, не считается
    logits_base = torch.randn(B, T, V)
    logits_marked = torch.randn(B, T, V)
    input_ids = torch.randint(0, V, (B, T))
    attention_mask = torch.ones(B, T)

    ref = length_normalized_llr(logits_base, logits_marked,
                                input_ids, attention_mask)

    lp_base = torch.log_softmax(logits_base.float(), dim=-1)
    lp_marked = torch.log_softmax(logits_marked.float(), dim=-1)
    for b in range(B):
        # по-токенно: j = 1..k-1, label = input_ids[b, j+1]
        # (сдвиг референса: позиция 0 — BOS)
        labels = input_ids[b, 2:]
        lb = lp_base[b, 1:-1].gather(1, labels.unsqueeze(-1)).squeeze(-1)
        lm = lp_marked[b, 1:-1].gather(1, labels.unsqueeze(-1)).squeeze(-1)
        n = len(labels)  # = k - 1 посчитанных токенов
        ours = (lm.sum() - lb.sum()) / n
        assert abs(ours.item() - ref[b].item()) < 1e-4, \
            (b, ours.item(), ref[b].item())

    print(f"OK: по-токенный LLR (detect.py) == openstamp/src/llr.py "
          f"({B} последовательностей, k={k})")


if __name__ == "__main__":
    main()
