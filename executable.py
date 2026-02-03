import sys
import json
from utils.scripts.post_processor import process_single_file  # Import from post_processor.py
from utils.scripts.sam_pipeline import PrecisionJSONPipeline  # Import from sam_pipeline.py

def main():
    if len(sys.argv) < 2:
        print("Usage: python executable.py <image_path>")
        return

    image_path = sys.argv[1]
    
    # 1. RUN QWEN PIPELINE (Encapsulated)
    # This automatically handles initialization, extraction, cleaning, and memory management
    qwen_final, qwen_latency = process_single_file(image_path, device="cuda")
    
    # Extract the confidence score that was placed outside 'fields' by clean_and_refine_result
    qwen_conf = qwen_final.get("confidence", 0.0)

    # 2. RUN SAM/YOLO PIPELINE
    sam_pipeline = PrecisionJSONPipeline(yolo_path="./utils/Models/stamp_signature_detector/best.pt")
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
        bnb_config = BitsAndBytesConfig(
