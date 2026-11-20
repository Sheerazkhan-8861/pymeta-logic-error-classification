#!/usr/bin/env python3
"""Dynamic-padding, assistant-only collator for PyMETA hierarchical training."""

from typing import Any

import torch


class HierarchicalDataCollator:
    """
    Expected item:
      {"messages": [system, user, assistant]}

    Prompt and padding labels are -100. Assistant response tokens are supervised.
    No truncation is ever performed. Sequences beyond max_length are rejected.
    """

    def __init__(self, tokenizer: Any, max_length: int | None = 1536):
        self.tokenizer = tokenizer
        self.max_length = max_length

        if tokenizer.pad_token_id is None:
            raise ValueError(
                "Tokenizer has no pad_token_id. Configure padding first."
            )

        if not getattr(tokenizer, "chat_template", None):
            raise ValueError("Tokenizer chat_template is missing.")

        if max_length is not None and (
            not isinstance(max_length, int) or max_length < 1
        ):
            raise ValueError("max_length must be a positive integer or None.")

        self.pad_token_id = tokenizer.pad_token_id
        self.label_pad_token_id = -100

    def _tokenize(self, text: str) -> list[int]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
        )
        return encoded["input_ids"]

    def _prepare_example(self, example: dict) -> tuple[list[int], list[int]]:
        if not isinstance(example, dict):
            raise TypeError("Each example must be a dictionary.")

        messages = example.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            raise ValueError("Each example must contain exactly three messages.")

        expected_roles = ["system", "user", "assistant"]
        if any(not isinstance(message, dict) for message in messages):
            raise ValueError("Every message must be a dictionary.")

        actual_roles = [message.get("role") for message in messages]
        if actual_roles != expected_roles:
            raise ValueError(f"Invalid message roles: {actual_roles}")

        if any(not isinstance(message.get("content"), str) for message in messages):
            raise ValueError("Every message content must be a string.")

        prompt_text = self.tokenizer.apply_chat_template(
            messages[:2],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_ids = self._tokenize(prompt_text)
        input_ids = self._tokenize(full_text)

        prompt_length = len(prompt_ids)
        sequence_length = len(input_ids)

        if prompt_length == 0:
            raise ValueError("Empty prompt token sequence.")

        if sequence_length <= prompt_length:
            raise ValueError("Assistant response contains no tokens.")

        if input_ids[:prompt_length] != prompt_ids:
            raise ValueError(
                "Prompt/full tokenization boundary mismatch. "
                "Check the tokenizer chat template."
            )

        if self.max_length is not None and sequence_length > self.max_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds the configured "
                f"maximum of {self.max_length}; truncation is prohibited."
            )

        labels = [-100] * prompt_length + input_ids[prompt_length:]

        if len(labels) != len(input_ids):
            raise RuntimeError("Input and label lengths differ.")

        if not any(token != -100 for token in labels):
            raise ValueError("No supervised assistant tokens found.")

        return input_ids, labels

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Received an empty batch.")

        prepared = [self._prepare_example(example) for example in features]
        max_batch_length = max(len(input_ids) for input_ids, _ in prepared)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for input_ids, labels in prepared:
            padding_length = max_batch_length - len(input_ids)

            batch_input_ids.append(
                input_ids + [self.pad_token_id] * padding_length
            )
            batch_attention_mask.append(
                [1] * len(input_ids) + [0] * padding_length
            )
            batch_labels.append(
                labels + [self.label_pad_token_id] * padding_length
            )

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }
