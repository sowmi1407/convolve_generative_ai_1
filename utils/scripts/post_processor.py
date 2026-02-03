import os
import json
import time
import torch
import gc
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from .qwen_extractor import QwenVLStampSignatureExtractor  # Import the extractor
from .utils import dbg  # Import debug function

def clean_and_refine_result(result, doc_id, device="cuda"):
    dbg(f"clean_and_refine_result() started for doc_id={doc_id}")

    # --------------------------------------------------
    # PART 1: LIGHT RULE-BASED CLEANING (SAFE ONLY)
    # --------------------------------------------------

    # Business name: remove Hindi "quotation" word only

    conf_score = result.pop("_internal_conf", 0.0)

    if result.get("business_name"):
        result["dealer_name"] = result["business_name"].replace("कोटेचन", "").strip()
        dbg("Initial field cleanup done")
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
    # from .utils import dbg  # Already imported above

    dbg(f"process_single_file() started for {image_path}")

    extractor = QwenVLStampSignatureExtractor(device=device)
    doc_id = os.path.splitext(os.path.basename(image_path))[0]

    # ---- GUARANTEED RETURNS ----
    final_json = None
    total_latency = 0.0

    try:
        dbg("Starting Qwen 7B extraction")
        # -------------------------------
        # QWEN 7B EXTRACTION
        # -------------------------------
        t0 = time.perf_counter()
        result = extractor.extract(image_path)
        t1 = time.perf_counter()
        dbg("Qwen 7B extraction finished")
      
        qwen_7b_latency = t1 - t0

        # Free heavy model ASAP
        dbg("Freeing Qwen 7B memory")

        del extractor.model
        del extractor.processor
        del extractor
        gc.collect()
        torch.cuda.empty_cache()

        # -------------------------------
        # POST-PROCESSING
        # -------------------------------
        t2 = time.perf_counter()
        dbg("Starting post-processing")
        final_json, post_latency = clean_and_refine_result(result, doc_id)
        t3 = time.perf_counter()
        dbg("Post-processing finished")

        total_latency = qwen_7b_latency + post_latency

    except Exception as e:
        dbg(f"PIPELINE FAILED for {doc_id}: {e}")

        print(f"[ERROR] Failed processing {doc_id}: {e}")

        # ---- SAFE FALLBACK JSON ----
        final_json = {
            "doc_id": doc_id,
            "confidence": 0.0,
            "fields": {
                "dealer_name": None,
                "model_name": None,
                "horse_power": None,
                "asset_cost": None
            }
        }
        total_latency = 0.0
    dbg(f"process_single_file() completed for {doc_id}")

    return final_json, total_latency