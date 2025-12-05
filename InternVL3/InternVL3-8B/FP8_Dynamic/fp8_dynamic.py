import torch
from transformers import AutoModel, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Load model.
model_id = "/root/workspace/models/InternVL3-8B"
model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)


# Recipe
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["re:.*lm_head","re:mlp1.*", "re:.*vision_model.*"]
)

# Perform oneshot
oneshot(
    model=model,
    recipe=recipe,
)


# Save to disk compressed.
SAVE_DIR = "/root/workspace/models/InternVL3-8B-FP8-DYNAMIC"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)