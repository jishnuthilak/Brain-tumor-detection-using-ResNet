# Brain-tumor-detection-using-ResNet

# Let's load the Jupyter Notebook to inspect its contents
import nbformat

# Define the path to the uploaded file
file_path = '/mnt/data/BRAIN TUMOR MODEL.ipynb'

# Load the notebook
with open(file_path, 'r', encoding='utf-8') as file:
    notebook = nbformat.read(file, as_version=4)

# Extract the content of the notebook
notebook_content = ""
for cell in notebook.cells:
    if cell.cell_type == 'markdown':
        notebook_content += cell.source + "\n\n"

notebook_content[:3000]  # Display the first 3000 characters to get an overview
