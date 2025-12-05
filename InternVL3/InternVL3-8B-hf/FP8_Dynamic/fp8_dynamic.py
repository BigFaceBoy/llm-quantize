import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Load model.
model_id = "/root/workspace/models/InternVL3-8B-hf"
model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)


recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["re:.*lm_head","re:.*multi_modal_projector.*", "re:.*vision_tower.*"]
)

# Perform oneshot
oneshot(
    model=model,
    recipe=recipe,
)


# Save to disk compressed.
SAVE_DIR = "/root/workspace/models/InternVL3-8B-hf-FP8-DYNAMIC"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

