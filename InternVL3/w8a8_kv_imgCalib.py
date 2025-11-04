import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.awq import AWQModifier, AWQMapping
from llmcompressor.utils import dispatch_for_generation
import base64
from io import BytesIO

# Load model.
model_id = "/root/workspace/models/InternVL3-8B-hf"
model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained(model_id)

# Load datasets
DATASET_ID = "/root/workspace/datasets/flickr30k"
DATASET_SPLIT = "test[:512]"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048
ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42)

def preprocess_and_tokenize(example):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": example["image"]
                },
                {
                    "type": "text", 
                    "text":  "What does the image show?"
                },
            ],
        }
    ]

    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    return inputs

ds = ds.map(preprocess_and_tokenize, remove_columns=ds.column_names, writer_batch_size=10)

def data_collator(batch):
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}

# Recipe
recipe = """
quant_stage:
    quant_modifiers:
        QuantizationModifier:
            ignore: ["re:.*lm_head",  "re:.*vision_tower.*",  "re:.*multi_modal_projector.*"]
            config_groups:
                group_0:
                    weights:
                        num_bits: 8
                        type: float
                        strategy: tensor
                        dynamic: false
                        symmetric: true
                    input_activations:
                        num_bits: 8
                        type: float
                        strategy: tensor
                        dynamic: false
                        symmetric: true
                    targets: ["Linear"]
            kv_cache_scheme:
                num_bits: 8
                type: float
                strategy: tensor
                dynamic: false
                symmetric: true
"""


# Perform oneshot
oneshot(
    model=model,
    tokenizer=model_id,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    trust_remote_code_model=True,
    data_collator=data_collator
)

# Save to disk compressed.
SAVE_DIR = "/root/workspace/models/InternVL3-8B-hf-FP8-W8A8-FP8-KV-flickr30k"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)