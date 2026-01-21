'''
Seperate utility script to print out agentic workflow of application.  Recommend running terminal from root as a module using the "-m" like: "python -m utils.generate_graph"
'''

from agents import AgentWorkflow 
import os

# 1. Initialize your class
app = AgentWorkflow()

# Determine the directory where this script is located (the 'utils' folder)
utils_folder = os.path.dirname(__file__)

# 2. Get the graphs from the compiled workflow
# Modern Mermaid version
mermaid_png = app.compiled_workflow.get_graph().draw_mermaid_png()

# Legacy/Classic version
# Note: This usually requires 'pygraphviz' or 'graphviz' installed on your system
try:
    classic_png = app.compiled_workflow.get_graph().draw_png()
except Exception as e:
    print(f"Warning: Could not generate classic draw_png. You may need graphviz installed. Error: {e}")
    classic_png = None

# 3. Save to the 'utils' folder
mermaid_path = os.path.join(utils_folder, "workflow_mermaid.png")
with open(mermaid_path, "wb") as f:
    f.write(mermaid_png)

if classic_png:
    classic_path = os.path.join(utils_folder, "workflow_classic.png")
    with open(classic_path, "wb") as f:
        f.write(classic_png)
    print(f"Flow graphs saved in utils: \n - workflow_mermaid.png \n - workflow_classic.png")
else:
    print(f"Only Mermaid graph saved: workflow_mermaid.png")