# 1. Model setup

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from device import DEVICE

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"


def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    ).to(DEVICE)
    model.eval()
    return model, processor


model, processor = load_model()
print("Model loaded.")
