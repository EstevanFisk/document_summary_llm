# DocChat: Serverless Agentic RAG with Docling & LangGraph 🐥

> **A "Serverless-First" document analysis agent that uses advanced layout-aware parsing (Docling) and graph-based orchestration (LangGraph) to perform deep research on complex technical reports.**

[![Deployed on Modal](https://img.shields.io/badge/Deployed_on-Modal-green?style=for-the-badge&logo=modal)](https://modal.com)
[![Powered by Docling](https://img.shields.io/badge/Parsing-Docling-blue?style=for-the-badge)](https://github.com/DS4SD/docling)
[![Orchestrated with LangGraph](https://img.shields.io/badge/Agents-LangGraph-orange?style=for-the-badge)](https://langchain-ai.github.io/langgraph/)

---

## 🏗️ Architecture

This project is not just a wrapper around OpenAI. It is a full-stack engineering demonstration of how to build, optimize, and deploy a heavy computer-vision RAG pipeline on serverless infrastructure.

### The Core Pipeline
1.  **Ingestion (The "Vision" Layer):** Uses **Docling** to render PDFs as images first, then extracts text with layout awareness. This prevents the common "headers merging with body text" issue found in standard PyPDF parsing.
2.  **Orchestration (The "Brain"):** A **LangGraph** state machine that doesn't just "retrieve and generate." It:
    * **Grades** retrieved documents for relevance.
    * **Rewrites** queries if the initial retrieval is poor.
    * **Loops** (Agentic behavior) until a satisfactory answer is found.
3.  **Infrastructure (The "Metal"):** Deployed on **Modal**, utilizing:
    * **Serverless GPUs (T4)** for OCR and Embeddings.
    * **Custom Container Images** with pinned Linux system libraries (`libGL`, `fonts-liberation`).
    * **Concurrency Management** to handle multiple users without burning GPU credits.

---

## 🚀 Live Demo
**Try the app here:** [INSERT YOUR MODAL URL HERE]

*(Note: The app runs on a cold-start architecture to save costs. Please allow 30-60 seconds for the first request as the container spins up.)*

---

## 🛠️ Tech Stack & Engineering Decisions

| Component | Technology | Why I Chose It |
| :--- | :--- | :--- |
| **Parsing** | **Docling** (IBM) | Standard parsers break on multi-column PDFs. Docling uses computer vision to understand layout before extraction. |
| **Agents** | **LangGraph** | Standard chains are too linear. Graphs allow for cyclical logic (e.g., "Check answer -> Bad? -> Retry"). |
| **Compute** | **Modal** | AWS/GCP are overkill for this. Modal allows defining infrastructure *in code* (IaC) and billing by the second. |
| **UI** | **Gradio (Pinned)** | Rapid prototyping interface. Pinned to v5.16+ to handle Pydantic serialization conflicts. |
| **OCR** | **RapidOCR** | Lightweight ONNX-based OCR that runs efficiently on CPU/Low-tier GPU when vector fonts fail. |

---

## ⚡ Challenges & Solutions

### 1. The "Linux Slim" Font Problem
**Challenge:** The app worked perfectly on Windows but produced "gibberish" tokens on Modal.
**Root Cause:** The `debian-slim` container lacks standard fonts. Docling couldn't render the PDF text layer and fell back to OCR, which failed without fonts.
**Solution:** Custom image build step installing `fonts-liberation` and `libgl1` at the OS level.
```python
# modal_app.py
image = modal.Image.debian_slim().apt_install("libgl1", "fonts-liberation")
```


### 2. "Dependency Hell" (Pydantic vs. Gradio)
**Challenge:** `Docling` requires Pydantic v2.10+ (for strict schemas), but `Gradio` crashed when receiving boolean schema types from that version.
**Solution:** Pinned Pydantic to `<2.10` to maintain stability while allowing Gradio to function, illustrating the trade-off between "bleeding edge" features and stability.

---

## 💻 Local Setup

If you want to run this on your own machine (requires NVIDIA GPU recommended):

1. **Clone the repo**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/docchat-portfolio.git](https://github.com/YOUR_USERNAME/docchat-portfolio.git)
   cd docchat-portfolio
   ```

2. **Set up environment**
   ```bash
   conda create -n docchat python=3.12
   pip install -r requirements.txt
   ```


3. **Set API Keys**
   Create a `.env` file in the root directory:
   ```bash
   GOOGLE_API_KEY=your_key_here
   OPENAI_API_KEY=your_key_here
   ```

4. **Run**
   ```bash
   python gradio_app.py
   ```

## ☁️ Deployment (Modal)

To deploy your own instance to the cloud:

1. **Install Modal**
   ```bash
   pip install modal
   modal setup
   ```

2. **Deploy**
   ```bash
   modal deploy modal_app.py
   ```
