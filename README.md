# Extract–Refine–Segment (ERS) Pipeline for Document Intelligence

This repository contains the official implementation for the **IDFC Convolve 4.0 Challenge**.

The **Extract–Refine–Segment (ERS) Pipeline** is a modular, multi-stage document intelligence architecture designed to process Indian tractor loan quotations. The system is robust to real-world challenges such as layout volatility, multilingual text, handwritten artifacts, stamps/signatures, and varying image quality.

---
## 🧩 Pipeline Overview

The ERS system consists of **two parallel branches** that operate independently and are merged into a final structured output:

### 1. Text / Information Extraction Branch
- Uses **vision-language and text-only large language models (LLMs)**.
- Extracts structured semantic fields such as dealer name, tractor brand/model, horsepower, and payable amount.
- Performs post-extraction normalization and cleanup to handle noisy OCR, spelling variations, and domain-specific naming inconsistencies.

### 2. Visual Detection & Segmentation Branch
- Uses **computer vision models** to detect and localize visual elements.
- Identifies and segments **stamps and signatures** using object detection followed by instance segmentation for pixel-level accuracy.

### 3. Merge & Consolidation
- Outputs from both branches are merged into a **single JSON schema**.
- Includes **confidence scoring** and **latency measurements**.
- Confidence weighting favors semantic extraction while validating with visual evidence.

---

## 🧠 Model Stack (High-Level)

### Text Branch
- **Qwen2.5-VL-7B-Instruct**  
  Primary vision-language model for image-level semantic extraction.

- **Qwen2.5-3B-Instruct**  
  Lightweight text-only model for post-processing, normalization, and rule-based refinement.

### Visual Branch
- **YOLO (custom fine-tuned)**  
  Detects bounding boxes for stamps and signatures.

- **SAM 3 (Segment Anything Model)**  
  Refines YOLO detections with precise instance segmentation.

> **Execution Order**
> - Text Branch: Qwen VL 7B → Qwen 3B  
> - Visual Branch: YOLO → SAM 3  
> - Final merge performed in the main pipeline with weighted confidence aggregation.

---

## 📦 Submission Structure

```text
submission.zip
│
├── executable.py          # Main inference entry point
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── Report.pdf             # Detailed technical report
├── sample_output/
│   └── result.json        # Example of final consolidated output schema
└── utils/
    ├── best.pt            # Fine-tuned YOLOv8s weights (stamp/signature detection)
    └── config.json        # Hugging Face access token
```

---

## ⚙️ Installation

Install all required dependencies using:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

After installing the required dependencies, run the pipeline using the following command:

```bash
python executable.py IMAGE_PATH
```

### Arguments

- `IMAGE_PATH`  
  Absolute or relative path to the input document image (e.g., scanned tractor loan quotation).

### Example

```bash
python executable.py sample_images/loan_quote.jpg
```

### Output

- The pipeline performs end-to-end **extraction, refinement, and segmentation**.
- The final consolidated output is generated in structured JSON format.
- A reference output schema is available at:

```
sample_output/result.json
```

---

## 🔒 Gated Model Access (SAM 3)

This pipeline utilizes **Segment Anything Model 3 (SAM 3)**, which is hosted on a **gated Hugging Face repository**.

### Access Requirement

Before execution, ensure that the usage terms for SAM 3 have been accepted at:

```
https://huggingface.co/facebook/sam3
```

### Authentication Strategy

For evaluation convenience, an **authorized Hugging Face access token** has been pre-configured in:

```
utils/config.json
```

> **Important Note**  
> Hardcoding access tokens is **not recommended** for production environments due to security best practices.  
> This approach has been **intentionally adopted** to ensure a **frictionless evaluation experience**, removing the need for:
> - Manual environment variable configuration  
> - CLI-based Hugging Face authentication  
> - Additional setup steps  

---

## 🧠 Memory Management Strategy (16 GB VRAM Constraint)

The pipeline is explicitly optimized to operate within a **16 GB VRAM** limit (e.g., NVIDIA T4), requiring careful GPU memory orchestration.

### 1. Sequential Resource Cycling

To avoid GPU Out-of-Memory (OOM) errors, models are loaded and unloaded sequentially:

- **Load & Extract**  
  The **Qwen-2.5-7B-VL** model is loaded to perform primary semantic extraction.

- **Explicit Deallocation**
  ```python
  del model
  ```

- **Cache Clearance**
  ```python
  torch.cuda.empty_cache()
  ```

- **Re-Initialization**  
  After memory is flushed, **YOLOv8** and **SAM 3** are initialized for geometric attestation (stamps, signatures, layout validation).

---

### 2. Hardware Scalability

While sequential loading guarantees compatibility with 16 GB GPUs, it introduces additional latency.

- **Recommended Hardware:**  
  24 GB VRAM GPU (e.g., NVIDIA A5000)

- **Observed Improvement:**  
  End-to-end inference time reduces from approximately **29 seconds to 9 seconds**.

---


## 📄 Additional Documentation

For detailed information on the **architecture**, **end-to-end pipeline**, **cost analysis**, and **system diagrams**, please refer to the report provided in this repository.
