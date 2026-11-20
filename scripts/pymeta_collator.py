import torch

def pymeta_collator(records):
    batch_input_ids = []
    batch_attention_masks = []
    batch_labels = []

    for record in records:
        messages = record["messages"]

        prefix_text = tokenizer.apply_chat_template(
            messages[:2],
            tokenize=False,
            add_generation_prompt=True
        )

        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        prefix_ids = tokenizer(
            prefix_text, add_special_tokens=False
        )["input_ids"]

        full_ids = tokenizer(
            full_text, add_special_tokens=False
        )["input_ids"]

        prefix_length = len(prefix_ids)
        suffix_ids = full_ids[prefix_length:]

        assert full_ids[:prefix_length] == prefix_ids
        assert len(full_ids) <= MAX_LENGTH
        assert len(suffix_ids) == 9
        assert tokenizer.eos_token_id in suffix_ids
        assert suffix_ids[-2] == tokenizer.eos_token_id
        assert suffix_ids[-1] == 198  # Verified trailing newline

        labels = [-100] * prefix_length + suffix_ids.copy()

        batch_input_ids.append(full_ids)
        batch_attention_masks.append([1] * len(full_ids))
        batch_labels.append(labels)

    batch_max_length = max(len(x) for x in batch_input_ids)

    for i in range(len(batch_input_ids)):
        padding_length = batch_max_length - len(batch_input_ids[i])

        batch_input_ids[i] += [tokenizer.pad_token_id] * padding_length
        batch_attention_masks[i] += [0] * padding_length
        batch_labels[i] += [-100] * padding_length

    return {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(
            batch_attention_masks, dtype=torch.long
        ),
        "labels": torch.tensor(batch_labels, dtype=torch.long)
    }
