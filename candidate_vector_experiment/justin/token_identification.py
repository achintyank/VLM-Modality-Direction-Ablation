# 2. Token identification
#
# Two different problems here:
# - Image tokens are easy to find — Qwen2-VL marks them with a fixed `<|image_pad|>` token id,
#   so a simple `input_ids == image_token_id` mask works regardless of what else is in the sequence.
# - Caption tokens are just regular text, indistinguishable from the prompt text by token id.
#   To find *where the caption sits* in the tokenized sequence, we locate the caption substring in
#   the rendered chat text via its character offsets, then map those to token indices using the
#   tokenizer's offset mapping (requires a fast tokenizer — Qwen2's is fast by default).
#
# Verify in Colab: confirm `processor.tokenizer.is_fast` is `True`, and that
# `chat_text.find(caption)` actually finds the caption unmodified inside the rendered template —
# some chat templates escape or alter text, which would break the offset lookup.

import torch

from model_setup import model, processor


def get_image_token_id(model, processor) -> int:
    if hasattr(model.config, "image_token_id"):
        return model.config.image_token_id
    return processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")


def get_image_mask(input_ids: torch.Tensor, image_token_id: int) -> torch.Tensor:
    """Boolean mask over a single sequence (seq_len,) marking image-token positions."""
    return input_ids == image_token_id


def get_caption_token_mask(chat_text: str, caption: str, tokenizer) -> torch.Tensor:
    """
    Boolean mask over the tokenized chat_text marking which tokens fall inside
    the caption substring, via character-offset alignment.
    """
    char_start = chat_text.find(caption)
    if char_start == -1:
        raise ValueError("Caption text not found verbatim in rendered chat template — "
                          "check for template escaping before trusting this mask.")
    char_end = char_start + len(caption)

    encoding = tokenizer(chat_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding["offset_mapping"]

    mask = torch.zeros(len(offsets), dtype=torch.bool)
    for i, (start, end) in enumerate(offsets):
        if start == end:
            continue  # special/padding token with no span
        if start >= char_start and end <= char_end:
            mask[i] = True
    return mask


image_token_id = get_image_token_id(model, processor)
print(f"image_token_id = {image_token_id}")
print(f"tokenizer is fast: {processor.tokenizer.is_fast}")
