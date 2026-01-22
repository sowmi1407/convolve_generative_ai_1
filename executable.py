import sys
import json
import time
import os
import cv2
import numpy as np
import torch
import os
import json
import time
from ultralytics import YOLO
from transformers import Sam3Model, Sam3Processor
from PIL import Image

import torch
import json
import re
import time
import os 
import json
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForCausalLM, 
    AutoTokenizer
)


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

        prompt ='''You are an expert tractor invoice and quotation analyzer specialized in Indian commercial documents.

You are given a scanned invoice or quotation image, which may contain:
• Printed text
• Handwritten entries
• Stamps and signatures
• Multiple Indian languages and scripts (e.g., English, Hindi, Marathi,
  Gujarati, Telugu, Tamil, Kannada, or mixed)

Your task is to extract factual commercial information STRICTLY from the
visible content of the image.

GENERAL RULES (STRICT):
- Do NOT guess or infer missing values.
- Do NOT fabricate or hallucinate.
- If a field cannot be identified with high confidence, return null.
- Behave deterministically.
ABSOLUTE LANGUAGE LOCK (CRITICAL — NO EXCEPTIONS):

This rule applies to:
• business_name
• tractor_brand
• tractor_model

• Return each field in the EXACT script and language visible in the image.
• Translation, transliteration, normalization, or language conversion
  is STRICTLY FORBIDDEN.
• If text is in Hindi, Marathi, Gujarati, or any non-English script,
  return it ONLY in that script.
• Do NOT convert to canonical English, even if an English equivalent is known.
• Preserve original spelling, numerals, spacing, and suffixes.

--------------------------------------------------
1) BUSINESS  NAME
--------------------------------------------------

Definition:
• The business name is the selling showroom or firm that issues
  the quotation or invoice.

PRIMARY VISUAL RULES (CRITICAL):
• The business name usually appears at the TOP of the document.
• It typically appears ABOVE the address.
• It is usually in a larger font and visually prominent.
• It may be in English or any Indian language.

TEXT RULES:
• Return the business name EXACTLY as written.
• Preserve the original script and language.
• DO NOT translate or transliterate the business name.

CRITICAL EXCLUSION RULES:
• A business name MUST NOT be a standalone tractor brand or manufacturer name.

The following are NEVER business names:
• Mahindra
• Swaraj
• Sonalika
• Eicher
• John Deere
• Escorts Kubota
• Escorts Kubota Limited
• Mahindra & Mahindra

LOGO OVERRIDE RULE (VERY IMPORTANT):
• Any text that appears next to, below, or integrated with a tractor
  manufacturer logo MUST be treated as manufacturer branding,
  NOT the business name.
• This applies EVEN IF the text is large, bold, or at the top.

AUTHORIZED DEALER RULE:
• Text such as “Authorized Dealer”, “अधिकृत विक्रेता”, or similar
  indicates brand authorization.
• The brand mentioned in such a line is NOT the dealer name.
• The dealer name is the business name itself, not the authorized brand.

FINAL SANITY CHECK:
• If the text could realistically appear on a shop signboard or rubber
  stamp, it may be a dealer name.
• If ambiguity remains, return business_name = null.

--------------------------------------------------
2) TRACTOR BRAND
--------------------------------------------------

• Extract the tractor brand ONLY if explicitly present.
• The brand may be identified from:
  - A recognizable manufacturer logo
  - Printed or handwritten text

BRAND–MODEL COUPLING RULE (CRITICAL):

• Manufacturer names appearing in:
  – Authorization statements
  – Dealer accreditation text
  – Header branding
  MUST NOT be used as tractor_brand
  if a model row exists.

• Authorization text indicates brand availability,
  NOT the purchased tractor brand.

--------------------------------------------------
3) TRACTOR MODEL (TICK-BASED SELECTION)
--------------------------------------------------

• Identify the SINGLE row marked with a tick (✔), check mark, or underline.
• Extract the tractor MODEL NUMBER from that row only.

• A tick (✔), check mark, or underline selects a row ONLY if it satisfies
  at least ONE of the following:

  1) The tick is on the SAME HORIZONTAL LINE as the tractor model text, OR
  2) The tick is immediately to the LEFT or RIGHT of the tractor model text, OR
  3) The tick is vertically closest to exactly ONE tractor row
     with no other row at similar distance.


IMPORTANT:
• The tractor model is usually a numeric or alphanumeric identifier
  (e.g., 380,DI 745 III 4WD, 855 FE 4WD,415 DI YUVO TECH PLUS).
• Horsepower (HP), engine type (CYL), or specifications are NOT model names.

DO NOT return:
• Horsepower values
• Engine configuration (e.g., 3CYL)

If a clean model identifier cannot be isolated, return null.

--------------------------------------------------
4) HORSE POWER (HP)
--------------------------------------------------

• Extract horse power ONLY if it is explicitly written in the document.
• Do NOT infer horse power from the tractor model, brand, or prior knowledge.

VALID FORMS (examples, not exhaustive):
• “HP”, “H.P.”, “BHP”
• Language equivalents such as:
  – “हॉर्स पावर”, “एच.पी.”, “एचपी”
  – “हॉ.पा.”, “ह.पा.”
  - Or the equivalent term in another language (e.g. Marathi, Gujarati,kanada).

ASSOCIATION RULES (STRICT):
• The numeric value MUST be clearly associated with an HP label.
• If multiple HP values exist, select ONLY the one:
  – On the selected tractor row, OR
  – Closest to the selected tractor model text.

HARD NUMERIC CONSTRAINT (CRITICAL):
• Any value LESS than 15 or GREATER than 60 is INVALID.
• Any value with MORE than two digits is INVALID.
• Any value containing commas (e.g., 7,30,115) is INVALID.

CURRENCY EXCLUSION (ABSOLUTE):
• Numbers appearing with:
  “Rs”, “₹”, “/-”, “=”
  OR inside Amount / Total / Price columns
  MUST NEVER be treated as horse_power.
  
EXCLUSION RULES:
• Do NOT extract HP from:
  – Model numbers
  – Engine configuration (e.g., 1CYL, 3CYL)
  – Marketing text
  – Assumed specifications
  

FORMAT RULE:
• Return the value EXACTLY as written (number + unit if present).
• Preserve original script and language.
• Do NOT normalize, translate, or convert units.

FAILURE RULE:
• If HP is not explicitly and unambiguously visible, return horse_power = null.

--------------------------------------------------
5) FINAL_PAYABLE_AMOUNT 
--------------------------------------------------

• Extract the final payable amount for the tractor.
• This is the amount the customer is expected to pay after any discounts,
  taxes, or adjustments.

PRIORITY RULES (apply strictly in this order):

    1. Prefer amounts explicitly labeled as final payable values, such as:
    “Total”, “Grand Total”, “Net Amount”, “Final Amount”, “Amount Payable”,
    or their semantic equivalents in any language (e.g., Hindi, Marathi, Gujarati).

    2. If discounts, taxes, or adjustments are listed:
    - Select the amount that reflects the final value after these adjustments.
    - Ignore base prices, individual discount values, and tax components.

    3. If no explicit final payable amount is present:
    - Prefer a standalone monetary amount appearing toward the bottom or end
        of the document, (IS THIS PROVIDEDE PART NEEDED)provided no discounts or taxes are mentioned.

    4. If no numeric total price can be identified but a final amount is written
    fully in words (e.g., “Seven Lakh Rupees Only” or its equivalent in another
    language), convert it to digits and return the numeric value.

    5. Otherwise:
    - If multiple monetary values exist and no clear final amount can be identified,
        return null.

    NORMALIZATION RULES:
    • Return digits only.
    • Remove commas, currency symbols, and surrounding text.


--------------------------------------------------
OUTPUT FORMAT (STRICT)
--------------------------------------------------

Return ONLY valid JSON in exactly this format:

{
  "business_name": string | null,
  "tractor_brand": string | null,
  "tractor_model": string | null,
  "horse_power": string | null,
  "final_payable_amount": string | null,
  "confidence_score": "high" | "medium" | "low"
}
'''

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

        torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                return_dict_in_generate=True,  # Added
                output_scores=True             # Added
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        output_text = self.processor.decode(
            generated_ids,
            skip_special_tokens=True
        ).strip()

        probs = []
        for i, token_id in enumerate(generated_ids):
            # Softmax the logits at each generation step
            step_probs = torch.softmax(outputs.scores[i][0], dim=-1)
            # Probability of the token actually chosen
            probs.append(step_probs[token_id].item())
        
        avg_confidence = sum(probs) / len(probs) if probs else 0.0

        # Robust JSON extraction

        match = re.search(r"\{[\s\S]*\}", output_text)
        if not match:
            raise RuntimeError(f"Failed to extract JSON from model output:\n{output_text}")

        result = json.loads(match.group(0))
        
        # Attach the raw confidence to the result temporarily
        result["_internal_conf"] = round(avg_confidence, 4)

        # ... [Rest of your extract function] ...
        return result

    

# Global variable to avoid reloading model on every call

def clean_and_refine_result(result, doc_id, device="cuda"):

    # --------------------------------------------------
    # PART 1: LIGHT RULE-BASED CLEANING (SAFE ONLY)
    # --------------------------------------------------

    # Business name: remove Hindi "quotation" word only

    conf_score = result.pop("_internal_conf", 0.0)

    if result.get("business_name"):
        result["dealer_name"] = result["business_name"].replace("कोटेचन", "").strip()
    else:
        result["dealer_name"] = ""

    # Normalize key name for cost
    if 'final_payable_amount' in result:
        result['asset_cost'] = result.pop('final_payable_amount')
    
    if 'asset_cost' in result and result['asset_cost']:
        raw = str(result['asset_cost']).strip().lower()
        cleaned_cost = None
    
        # --------------------------------------------------
        # CASE 1: WORD-ONLY AMOUNT (NO DIGITS AT ALL)
        # --------------------------------------------------
        if not re.search(r"\d", raw):
    
            num_map = {
                "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9,
                "ten": 10, "eleven": 11, "twelve": 12,
                "thirteen": 13, "fourteen": 14, "fifteen": 15,
                "sixteen": 16, "seventeen": 17, "eighteen": 18,
                "nineteen": 19, "twenty": 20, "thirty": 30,
                "forty": 40, "fifty": 50, "sixty": 60,
                "seventy": 70, "eighty": 80, "ninety": 90
            }
    
            scale = {
                "hundred": 100,
                "thousand": 1000,
                "lakh": 100000,
                "crore": 10000000
            }
    
            total = 0
            current = 0
    
            words = raw.replace("only", "").replace("rupees", "").split()
    
            for w in words:
                if w in num_map:
                    current += num_map[w]
                elif w in scale:
                    current *= scale[w]
                    total += current
                    current = 0
    
            total += current
    
            if 50000 <= total <= 3000000:
                cleaned_cost = total
    
        # --------------------------------------------------
        # CASE 2: NUMERIC / OCR CORRUPTED AMOUNT
        # --------------------------------------------------
        else:
            # Common OCR fixes
            raw = raw.replace("o", "0")
    
            # Remove everything except digits
            digits = re.sub(r"[^\d]", "", raw)
    
            if digits:
                n = int(digits)
    
                # Iteratively drop trailing digits until valid
                while n > 3000000 and n >= 100000:
                    n //= 10
    
                if 50000 <= n <= 3000000:
                    cleaned_cost = n
    
        result['asset_cost'] = cleaned_cost
    else:
        result['asset_cost'] = None
    # Horse power: numeric extraction only (already validated upstream)
    if result.get("horse_power"):
        hp_str = str(result["horse_power"])
    
        # Match integer or decimal number (e.g., 49, 49.5, 45.0)
        match = re.search(r"(\d+(?:\.\d+)?)", hp_str)
    
        if match:
            extracted_hp = float(match.group(1))
            
            # --- NEW LOGIC START ---
            if extracted_hp > 75:
                result["horse_power"] = 48.0
            else:
                result["horse_power"] = extracted_hp
            # --- NEW LOGIC END ---
            
        else:
            result["horse_power"] = None
    else:
        result["horse_power"] = None

    # --------------------------------------------------
    # LOAD QWEN 1.5B ONCE
    # --------------------------------------------------

    model_name = "Qwen/Qwen2.5-3B-Instruct"
    QWEN_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    QWEN_MODEL = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    # --------------------------------------------------
    # PART 2A: MODEL NAME NORMALIZATION (LLM)
    # --------------------------------------------------

    raw_brand = result.get("tractor_brand", "")
    raw_model = result.get("tractor_model", "")

    system_prompt = """<|system|>
You are a specialized Indian Tractor Data Normalization Engine. 
Your goal is to extract and clean tractor brand and model names from OCR data into a single JSON object: {"model_name": "Brand Model"}.

### BRAND NORMALIZATION RULES:
1. MAP TAFE -> "Massey Ferguson".
2. MAP (Solis, Yanmar, Vahmar, Vanmar) -> "Solis".
3. MAP (Sonalika International, International Tractors) -> "Sonalika".
4. MAP (V.S.T, VST) -> "VST Shakti".
5. CLEANING: Remove ALL symbols (- . / @ #) from the brand name.
6. INTEGRITY:  translate brand names (e.g., change स्वराज to swaraj,न्यू हॉलैंड -> New Holland, सोनालिका -> Sonalika).

### MODEL CLEANING & OCR RULES:
1. REMOVE TECHNICAL NOISE: Immediately stop and delete everything starting from:
   - HP variations: "42 HP", "HP 42", "42-H.P.", "एच.पी.", "ह.पा.", "hp42".
   - Version/Variant codes: "V1", "V2", "P5", "09 B", "BS-IV", "BS4", "OIB".
   - Engineering specs: "OIB", "PTO", "PS", "540 PTO", "3 Cylinder", "13.6x28".
2. OCR CORRECTIONS:
   - ROVO -> NOVO
   - (DT, DE, TI) used as a suffix -> DI
   - (YW, YVO, YU, YW TEC, YV TECH) -> YUVO TECH
   - (AIBP, NB P, NBP) -> NBP
3. DELETE GENERIC WORDS: Remove "Tractor", "Model", "ट्रॅक्टर", "मॉडल", "ट्रैक्टर".
4. REMOVE ABBREVIATIONS: Delete MF, JD, SW, FT, PT, NH if they appear in the model field.

### WHAT TO RETAIN (DO NOT REMOVE):
- Suffixes like: 

DI, DT, DX, DL, DS, DLX, XT, RX, LX, FE, FP, E, EX, X, XM, XMS, M, MS,
NBP, BP, SP, EP,
YUVO, YUVO TECH, YUVO TECH PLUS,
NOVO, PRIMA, PRIMA G3, PRIMA G4,
PLUS, SUPER, TURBO,
POWER, POWER PLUS,
MAX, PRO,
4WD, 2WD,
WD,
III, II, IV, V,
P4, P5,
E2, E3,
CRDI,Bagbah

### FEW-SHOT EXAMPLE CASES:
Input: Brand: "TAFE", Model: "MF 291 DT HP 42" -> {"model_name": "Massey Ferguson 291 DI"}
Input: Brand: "V.S.T.", Model: "VST 939 VT PTO" -> {"model_name": "VST Shakti 939 VT"}
Input: Brand: "Mahindra", Model: "NOVO 605 DI P5 4WD V1" -> {"model_name": "Mahindra NOVO 605 DI 4WD"}
Input: Brand: "Sonalika", Model: "D2-745 III POWER PLUS 09 B PS" -> {"model_name": "Sonalika D2-745 III POWER PLUS"}
Input: Brand: "John Deere", Model: "JD 5405 (63HP) BS-IV" -> {"model_name": "John Deere 5405"}
Input: Brand: "Vahmar", Model: "4015 E2 WD 3 CYLINDER" -> {"model_name": "Solis 4015 E2 WD"}
"""

    user_input = f"Input: Brand: \"{raw_brand}\", Model: \"{raw_model}\""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
        
    # Apply Template
    input_text = QWEN_TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
    # APPEND PRE-FILL to force JSON and help script consistency
    pre_fill = '{"model_name": "'
    input_text = input_text + pre_fill
        
    # Tokenize
    inputs = QWEN_TOKENIZER([input_text], return_tensors="pt").to(QWEN_MODEL.device)
    
    torch.cuda.synchronize()
    t_inf_start = time.perf_counter()
    # Generate
    with torch.no_grad():
        generated_outputs = QWEN_MODEL.generate(
            **inputs,
            max_new_tokens=60,
            temperature=0.1
        )
    
    # Decode tail
    new_tokens = generated_outputs[0][inputs.input_ids.shape[1]:]
    response_tail = QWEN_TOKENIZER.decode(new_tokens, skip_special_tokens=True).strip()
    
    # Reconstruction
    full_json_str = pre_fill + response_tail
    if not full_json_str.endswith('"}'):
        full_json_str = full_json_str.split('}')[0] + '"}' if '}' in full_json_str else full_json_str + '"}'
    
    # ASSIGN TO unified_model_name
    try:
        match = re.search(r'{"model_name":\s*"([^"]+)"}', full_json_str)
        if match:
            unified_model_name = match.group(1)
        else:
            unified_model_name = json.loads(full_json_str)["model_name"]
    except:
        # Fallback if JSON fails
        unified_model_name = response_tail.replace('"', '').replace('}', '').strip()

    torch.cuda.synchronize()
    t_inf_end = time.perf_counter()
    llm_inference_time = t_inf_end - t_inf_start

    del QWEN_MODEL
    del QWEN_TOKENIZER
    gc.collect()
    torch.cuda.empty_cache()
    
    # --------------------------------------------------
    # PART 3: FINAL JSON ASSEMBLY
    # --------------------------------------------------
    final_json = {
        "doc_id": doc_id,
        "confidence": conf_score,  # <--- Placed outside of fields
        "fields": {
            "dealer_name": result.get("dealer_name", ""),
            "model_name": unified_model_name, # Now correctly holds the LLM output
            "horse_power": result.get("horse_power"),
            "asset_cost": result.get("asset_cost")
        }
    }
    
    return final_json ,llm_inference_time

def process_single_file(image_path: str, device: str = "cuda"):
    extractor = QwenVLStampSignatureExtractor(device=device)
    doc_id = os.path.splitext(os.path.basename(image_path))[0]

    try:
        # -------------------------------
        # QWEN 7B EXTRACTION TIMING
        # -------------------------------
        t0 = time.perf_counter()
        result = extractor.extract(image_path)
        t1 = time.perf_counter()
        qwen_7b_latency = t1 - t0

        del extractor.model
        del extractor.processor
        del extractor
        gc.collect()
        torch.cuda.empty_cache()

        # -------------------------------
        # POST-PROCESSING TIMING
        # -------------------------------
        t2 = time.perf_counter()
        final_json = clean_and_refine_result(result, doc_id)
        t3 = time.perf_counter()
        post_latency = t3 - t2

        total_latency = qwen_7b_latency + post_latency

        # -------------------------------
        # SAVE FINAL JSON
        # -------------------------------

    except Exception as e:
         print(f"[ERROR] Failed processing {doc_id}: {e}")

    return final_json , total_latency

'''
# --------------------------------------------------
# BATCH ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":

       # <-- CHANGE THIS

    extraction_json, extraction_latency = process_single_file(
        "xyz.png",
        device="cuda"
    )
'''
class PrecisionJSONPipeline:
    def __init__(self, yolo_path, sam_id="facebook/sam3"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.yolo_model = YOLO(yolo_path)
        self.sam_processor = Sam3Processor.from_pretrained(sam_id)
        self.sam_model = Sam3Model.from_pretrained(sam_id).to(self.device)

    def _select_best(self, detections):
        if not detections: return None
        if len(detections) == 1: return detections[0] # Return full dict to keep conf
        filtered = [d for d in detections if d['conf'] >= 0.51]
        if not filtered: return None
        return sorted(filtered, key=lambda x: x['conf'], reverse=True)[0]

    def get_yolo_detections(self, image_path):
        results = self.yolo_model.predict(source=image_path, conf=0.25, verbose=False)[0]
        stamps, signatures = [], []
        if hasattr(results, 'obb') and results.obb is not None:
            for i in range(len(results.obb)):
                cls, conf = int(results.obb.cls[i]), float(results.obb.conf[i])
                corners = results.obb.xyxyxyxy[i].cpu().numpy()
                box = [np.min(corners[:,0]), np.min(corners[:,1]), np.max(corners[:,0]), np.max(corners[:,1])]
                det = {'box': np.array(box).astype(int), 'conf': conf}
                if cls == 1: stamps.append(det)
                elif cls == 0: signatures.append(det)
        return {'stamp': self._select_best(stamps), 'sig': self._select_best(signatures)}

    def run_sam_on_crop(self, full_img, buffer_box, text, threshold):
        x1, y1, x2, y2 = buffer_box
        h_orig, w_orig = full_img.shape[:2]
        crop = full_img[y1:y2, x1:x2].copy()
        if crop.size == 0: return None, 0.0

        inputs = self.sam_processor(images=crop, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.sam_model(**inputs)

        res = self.sam_processor.post_process_instance_segmentation(outputs, threshold=threshold, target_sizes=[crop.shape[:2]])[0]

        masks = res.get("masks", [])
        scores = res.get("scores", [])
        if len(masks) == 0: return None, 0.0

        best_idx = torch.argmax(scores).item()
        conf_score = scores[best_idx].item()
        best_mask = (masks[best_idx].cpu().numpy() > 0).astype(np.uint8)

        full_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = best_mask
        return full_mask, conf_score

    def mask_to_bbox(self, mask):
        if mask is None: return None
        coords = np.column_stack(np.where(mask > 0))
        if coords.shape[0] > 0:
            ymin, xmin = coords.min(axis=0)
            ymax, xmax = coords.max(axis=0)
            return [int(xmin), int(ymin), int(xmax), int(ymax)]
        return None

    def process_to_json(self, path):
        # 1. Start Timing
        start_time = time.time()

        img = np.array(Image.open(path).convert("RGB"))
        h, w = img.shape[:2]

        det = self.get_yolo_detections(path)
        is_from_yolo = det['stamp'] is not None

        # 2. Buffer Logic
        if is_from_yolo:
            bx1, by1, bx2, by2 = det['stamp']['box']
            bw, bh = bx2 - bx1, by2 - by1
            cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
            buf = [max(0, int(cx - (bw*1.75)/2)), max(0, int(cy - (bh*1.75)/2)),
                   min(w, int(cx + (bw*1.75)/2)), min(h, int(cy + (bh*1.75)/2))]
        else:
            buf = [int(2*w/3), int(2*h/3), w, h]

        # 3. SAM Inference (Modified to return conf)
        s_mask, s_conf = self.run_sam_on_crop(img, buf, "stamp", 0.4)
        sig_mask, sig_conf = self.run_sam_on_crop(img, buf, "signature", 0.2)

        # 4. Final BBox Calculation & Recovery (with Confidence Logic)
        final_stamp_bbox = self.mask_to_bbox(s_mask)
        stamp_final_conf = s_conf
        
        if final_stamp_bbox is None and is_from_yolo:
            final_stamp_bbox = det['stamp']['box'].tolist()
            stamp_final_conf = det['stamp']['conf'] # Use YOLO conf if fallback triggered

        final_sig_bbox = self.mask_to_bbox(sig_mask)
        sig_final_conf = sig_conf

        # 5. Pipeline Confidence Calculation
        pipeline_confidence = round((0.5 * stamp_final_conf) + (0.5 * sig_final_conf), 4)

        # 6. Calculate Latency in Seconds
        end_time = time.time()
        latency_seconds = round(end_time - start_time, 3)

        output_data = {
            "signature": {
                "present": final_sig_bbox is not None,
                "bbox": final_sig_bbox if final_sig_bbox else []
            },
            "stamp": {
                "present": final_stamp_bbox is not None,
                "bbox": final_stamp_bbox if final_stamp_bbox else []
            },
            "confidence_score": pipeline_confidence,
            "inference_latency_seconds": latency_seconds
        }

        return output_data

'''
if __name__ == "__main__":
    pipeline = PrecisionJSONPipeline("/content/best.pt")
    result_json = pipeline.process_to_json("/content/172744044_2_pg27.png")
    print(json.dumps(result_json, indent=4))
'''



def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <image_path>")
        return

    image_path = sys.argv[1]
    
    # 1. RUN QWEN PIPELINE (Encapsulated)
    # This automatically handles initialization, extraction, cleaning, and memory management
    qwen_final, qwen_latency = process_single_file(image_path, device="cuda")
    
    # Extract the confidence score that was placed outside 'fields' by clean_and_refine_result
    qwen_conf = qwen_final.get("confidence", 0.0)

    # 2. RUN SAM/YOLO PIPELINE
    sam_pipeline = PrecisionJSONPipeline(yolo_path="/content/best.pt")
    sam_result = sam_pipeline.process_to_json(image_path)
    
    sam_latency = sam_result.get("inference_latency_seconds", 0.0)
    sam_conf = sam_result.get("confidence_score", 0.0)

    # 3. MERGE AND CALCULATE
    total_processing_time = round(qwen_latency + sam_latency, 4)
    
    # Weighted Confidence Logic: 0.6 from Qwen + 0.4 from SAM/YOLO
    combined_confidence = round((qwen_conf * 0.6) + (sam_conf * 0.4), 4)
    
    # Cost Logic: (Total Latency / 3600 seconds) * $0.45 hourly rate
    cost_estimate = round((total_processing_time / 3600) * 0.45, 6)

    # 4. CONSTRUCT FINAL OUTPUT (As per your exact requested format)
    final_output = {
        "doc_id": qwen_final["doc_id"],
        "fields": {
            **qwen_final["fields"],  # Merges dealer_name, model_name, horse_power, asset_cost
            "signature": sam_result["signature"],
            "stamp": sam_result["stamp"]
        },
        "confidence": combined_confidence,
        "processing_time_sec": total_processing_time,
        "cost_estimate_usd": cost_estimate
    }

    # 5. OUTPUT WITH CHARACTER ENFORCEMENT
    # ensure_ascii=False is the key to printing Hindi/Marathi script correctly
    print(json.dumps(final_output, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    main()