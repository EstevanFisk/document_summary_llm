import modal
import os
import hashlib
from typing import List, Dict
import gradio as gr
from fastapi import FastAPI

# 1. Define the Container Image & MOUNTS
image = (
    modal.Image.debian_slim(python_version="3.12")
    # ✅ FONT FIX: Installs standard fonts so Docling doesn't fall back to bad OCR
    .apt_install("libgl1", "libglib2.0-0", "fonts-liberation") 
    .pip_install(
        "pydantic<2.10",      # ✅ CRASH FIX: Prevents "bool is not iterable" error
        "gradio>=5.0",        # Gets the latest Gradio features
        "docling",
        "rapidocr-onnxruntime",
        "pypdfium2",
        "langgraph",
        "langchain",
        "langchain-google-genai",
        "langchain-community",
        "google-generativeai",
        "openai",
        "pandas",
        "python-dotenv",
        "loguru",
        "chromadb",
        "rank_bm25",
        "numpy",
        "pydantic-settings"
    )
    .add_local_dir("agents", remote_path="/root/agents")
    .add_local_dir("config", remote_path="/root/config")
    .add_local_dir("document_processor", remote_path="/root/document_processor")
    .add_local_dir("engine_room", remote_path="/root/engine_room")
    .add_local_dir("examples", remote_path="/root/examples")
    .add_local_dir("providers", remote_path="/root/providers")
    .add_local_dir("utils", remote_path="/root/utils")
)

# 2. Define the Modal App with SECRETS
app = modal.App(
    "docchat-portfolio-project",
    secrets=[modal.Secret.from_dotenv()] 
)

# 3. The Main Function
@app.function(image=image, gpu="T4", timeout=600)
@modal.concurrent(max_inputs=100) # ✅ SCALING FIX: Correct decorator
@modal.asgi_app()
def run_gradio():
    # --- IMPORTS INSIDE THE FUNCTION ---
    import os
    os.environ["RAPIDOCR_LOG_LEVEL"] = "ERROR"
    
    from document_processor.file_handler import DocumentProcessor
    from engine_room.builder import RetrieverBuilder
    from agents.workflow import AgentWorkflow
    from config import constants
    from utils.logging import logger

    # Define Examples
    EXAMPLES = {
        "Google 2024 Environmental Report": {
            "question": "Retrieve the data center PUE efficiency values in Singapore 2nd facility in 2019 and 2022. Also retrieve regional average CFE in Asia pacific in 2023",
            "file_paths": ["examples/google-2024-environmental-report.pdf"]  
        },
        "DeepSeek-R1 Technical Report": {
            "question": "Summarize DeepSeek-R1 model's performance evaluation on all coding tasks against OpenAI o1-mini model",
            "file_paths": ["examples/DeepSeek Technical Report.pdf"]
        }
    }

    # Initialize Modules
    processor = DocumentProcessor()
    retriever_builder = RetrieverBuilder()
    workflow = AgentWorkflow()

    # CSS & JS
    # Added 'font-family' to ensure it looks decent even if the theme font fails
    css = """
    .title { font-size: 1.5em !important; text-align: center !important; color: #FFD700; }
    .subtitle { font-size: 1em !important; text-align: center !important; color: #FFD700; }
    .text { text-align: center; font-family: sans-serif; }
    """

    js = """
    function createGradioAnimation() {
        var container = document.createElement('div');
        container.id = 'gradio-animation';
        container.style.fontSize = '2em';
        container.style.fontWeight = 'bold';
        container.style.textAlign = 'center';
        container.style.marginBottom = '20px';
        container.style.color = '#eba93f';
        var text = 'Welcome to DocChat 🐥!';
        for (var i = 0; i < text.length; i++) {
            (function(i){
                setTimeout(function(){
                    var letter = document.createElement('span');
                    letter.style.opacity = '0';
                    letter.style.transition = 'opacity 0.1s';
                    letter.innerText = text[i];
                    container.appendChild(letter);
                    setTimeout(function() { letter.style.opacity = '0.9'; }, 50);
                }, i * 250);
            })(i);
        }
        var gradioContainer = document.querySelector('.gradio-container');
        gradioContainer.insertBefore(container, gradioContainer.firstChild);
        return 'Animation created';
    }
    """

    # Build the UI
    with gr.Blocks(theme=gr.themes.Citrus(), title="DocChat 🐥", css=css, js=js) as demo:
        gr.Markdown("## DocChat: powered by Docling 🐥 and LangGraph", elem_classes="subtitle")
        gr.Markdown("# How it works ✨:", elem_classes="title")
        gr.Markdown("📤 Upload your document(s), enter your query then hit Submit 📝", elem_classes="text")
        gr.Markdown("Or you can select one of the examples from the drop-down menu, select Load Example then hit Submit 📝", elem_classes="text")
        gr.Markdown("⚠️ **Note:** DocChat only accepts documents in these formats: '.pdf', '.docx', '.txt', '.md'", elem_classes="text")

        session_state = gr.State({"file_hashes": frozenset(), "retriever": None})

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Example 📂")
                example_dropdown = gr.Dropdown(
                    label="Select an Example 🐥",
                    choices=list(EXAMPLES.keys()),
                    value=None,
                )
                load_example_btn = gr.Button("Load Example 🛠️")
                files = gr.Files(label="📄 Upload Documents", file_types=[".pdf", ".docx", ".txt", ".md"])
                question = gr.Textbox(label="❓ Question", lines=3)
                submit_btn = gr.Button("Submit 🚀")
                
            with gr.Column():
                answer_output = gr.Textbox(label="🐥 Answer", interactive=False, lines=15, max_lines=30)
                verification_output = gr.Textbox(label="✅ Verification Report", lines=10)

        # --- Helper Functions ---
        
        def load_example(example_key: str):
            if not example_key or example_key not in EXAMPLES:
                return [], ""
            ex_data = EXAMPLES[example_key]
            loaded_files = []
            for path in ex_data["file_paths"]:
                # Handle paths whether they are absolute (remote) or relative (local)
                full_path = path if path.startswith("/") else f"/root/{path}"
                if os.path.exists(full_path):
                    loaded_files.append(full_path)
                else:
                    print(f"Warning: File not found at {full_path}")
            return loaded_files, ex_data["question"]

        # ✅ RESTORED FUNCTION
        def _get_file_hashes(uploaded_files: List) -> frozenset:
            hashes = set()
            for file in uploaded_files:
                with open(file.name, "rb") as f:
                    hashes.add(hashlib.sha256(f.read()).hexdigest())
            return frozenset(hashes)

        def process_question(question_text: str, uploaded_files: List, state: Dict):
            try:
                if not question_text.strip():
                    raise ValueError("❌ Question cannot be empty")
                if not uploaded_files:
                    raise ValueError("❌ No documents uploaded")

                current_hashes = _get_file_hashes(uploaded_files)
                
                # If documents changed or retriever is missing, rebuild it
                if state["retriever"] is None or current_hashes != state["file_hashes"]:
                    print("Processing new/changed documents...")
                    chunks = processor.process(uploaded_files)

                    if not chunks:
                        raise ValueError("No readable text found.")
                    
                    retriever = retriever_builder.build_hybrid_retriever(chunks)
                    state.update({"file_hashes": current_hashes, "retriever": retriever})
                
                result = workflow.full_pipeline(
                    question=question_text,
                    retriever=state["retriever"],
                    config={"recursion_limit": 10}
                )
                return result["draft_answer"], result["verification_report"], state
                    
            except Exception as e:
                logger.error(f"Processing error: {str(e)}")
                return f"❌ Error: {str(e)}", "", state

        # Event Listeners
        load_example_btn.click(load_example, inputs=[example_dropdown], outputs=[files, question])
        submit_btn.click(process_question, inputs=[question, files, session_state], outputs=[answer_output, verification_output, session_state])

    # Wrap in FastAPI to handle request routing correctly
    web_app = FastAPI()
    return gr.mount_gradio_app(app=web_app, blocks=demo, path="/")