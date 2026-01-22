# Extract-Refine-Segment (ERS) Pipeline for Document Intelligence

[cite_start]This repository contains the official implementation for the **IDFC Convolve 4.0 Challenge**[cite: 3, 9]. [cite_start]The **ERS Pipeline** is a modular, multi-stage architecture designed to handle the complexities of Indian tractor loan quotations, including layout volatility, multilingual scripts, and varying image quality[cite: 8, 9, 18].

---

##  Submission Structure
# Extract-Refine-Segment (ERS) Pipeline for Document Intelligence

This repository contains the official implementation for the **IDFC Convolve 4.0 Challenge**. The **ERS Pipeline** is a modular, multi-stage architecture designed to handle the complexities of Indian tractor loan quotations, including layout volatility, multilingual scripts, and varying image quality.

---

## 📂 Submission Structure
```text
submission.zip
│
├── executable.py          # Main inference script (entry point)
├── requirements.txt       # Environment dependencies
├── README.md              # Project documentation (this file)
├── Report.pdf             # Detailed Technical Report
├── sample_output/         
│   └── result.json        # Example of the final consolidated schema
└── utils/                 # Core logic and model assets
    └──  best.pt           # Fine-tuned YOLOv8s weights for Stamp/Signature detection
    └── config.json.       # Hugging face token


# Install dependencies
pip install -r requirements.txt    


## Gated Model Access (SAM 3)

This pipeline utilizes **Segment Anything Model 3 (SAM 3)**, which is hosted on a **gated Hugging Face repository**.

### Access Requirement
- Ensure that you have accepted the usage terms at:  
  **https://huggingface.co/facebook/sam3**

### Authentication Strategy
- For the convenience of the evaluation committee, an **authorized Hugging Face access token** has been pre-configured in:

### Important Note
> While hardcoding access tokens is **not recommended** for production environments due to security best practices, this approach has been **intentionally adopted** here.

- This design choice allows evaluators to run the pipeline **seamlessly**, without requiring:
- Manual environment variable setup
- CLI-based Hugging Face authentication
- Additional configuration steps

The goal is to ensure a **frictionless evaluation experience** while maintaining functional access to the gated SAM 3 model.


## Memory Management (16GB VRAM Strategy)

[cite_start]The pipeline is specifically optimized to run within the strict **16GB VRAM** envelope of a single NVIDIA T4 GPU[cite: 148, 321].

### **1. Sequential Resource Cycling**
[cite_start]To avoid **Out-of-Memory (OOM)** issues, we implement a strict sequential lifecycle for model assets:
* [cite_start]**Load & Extract:** The **Qwen-2.5-7B-VL** model is loaded into memory to perform the primary semantic extraction task[cite: 86, 152].
* [cite_start]**Explicit Deletion:** Once the extraction is complete, the model is **explicitly deleted** from the GPU memory (`del model`).
* [cite_start]**Cache Clearing:** We trigger `torch.cuda.empty_cache()` to ensure all allocated VRAM is released back to the system.
* [cite_start]**Re-Initialization:** Only after the memory is flushed do we initialize the **YOLOv8** and **SAM 3** stacks for geometric attestation[cite: 113, 152].

### **2. Hardware Scalability**
[cite_start]While this manual management ensures stability on 16GB hardware, it introduces a latency bottleneck due to repeated loading/unloading cycles[cite: 113, 173]. 
* [cite_start]**Recommendation:** This issue is easily overcome by using a **24GB VRAM GPU (e.g., NVIDIA A5000)**.
* [cite_start]**Benefit:** Increased VRAM allows all models (Qwen-7B, Qwen-3B, YOLO, and SAM 3) to remain resident in memory simultaneously, reducing total processing time from **29 seconds to 9 seconds**[cite: 149, 150].



