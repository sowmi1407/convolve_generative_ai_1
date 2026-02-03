import torch
import json
import re
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer
)
from huggingface_hub import login
from .utils import config, dbg  # Import config and debug function
from prompts import qwen_prompt

login(token=config['hf_token'])

class QwenVLStampSignatureExtractor:
    def __init__(
        self,
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        device="cuda"
    ):
        self.device = device

        # Processor for vision + language
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        # Load Qwen 3 VL model
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).eval()

    def extract(self, image_path):
        image = Image.open(image_path).convert("RGB")
        prompt = qwen_prompt  # Use the imported prompt

        # -------------------------
        try:
            print("[DEBUG] Applying chat template")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.processor(
                text=text,
                images=image,
                return_tensors="pt"
            ).to(self.device)

            print("[DEBUG] Processor completed")
        except Exception as e:
            print(f"[ERROR] Processor failed: {e}")
            raise

        # -------------------------
        # MODEL GENERATION
        # -------------------------
        try:
            print("[DEBUG] Starting model.generate()")
            torch.cuda.synchronize()

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    return_dict_in_generate=True,
                    output_scores=True
                )

            torch.cuda.synchronize()
            print("[DEBUG] model.generate() finished")
        except Exception as e:
            print(f"[ERROR] model.generate failed: {e}")
            raise

        # -------------------------
        # DECODE OUTPUT
        # -------------------------
        try:
            print("[DEBUG] Decoding output")
            generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]

            output_text = self.processor.decode(
                generated_ids,
                skip_special_tokens=True
            ).strip()

            print("[DEBUG] Raw model output:")
            print(output_text)
        except Exception as e:
            print(f"[ERROR] Decoding failed: {e}")
            raise

        # -------------------------
        # CONFIDENCE COMPUTATION
        # -------------------------
        try:
            print("[DEBUG] Computing token confidence")
            probs = []

            for i, token_id in enumerate(generated_ids):
                step_logits = outputs.scores[i][0]
                step_probs = torch.softmax(step_logits, dim=-1)
                probs.append(step_probs[token_id].item())

            avg_confidence = sum(probs) / len(probs) if probs else 0.0
            print(f"[DEBUG] Avg confidence: {avg_confidence:.4f}")
        except Exception as e:
            print(f"[ERROR] Confidence computation failed: {e}")
            avg_confidence = 0.0

        # -------------------------
        # JSON EXTRACTION
        # -------------------------
        try:
            print("[DEBUG] Extracting JSON")
            match = re.search(r"\{[\s\S]*\}", output_text)

            if not match:
                print("[ERROR] No JSON found in output")
                raise RuntimeError("JSON extraction failed")

            result = json.loads(match.group(0))
            result["_internal_conf"] = round(avg_confidence, 4)

            print("[DEBUG] Parsed JSON:")
            print(result)
        except Exception as e:
            print(f"[ERROR] JSON parsing failed: {e}")
            raise

        print("[DEBUG] extract() completed successfully")
        return result