import nbformat

nb = nbformat.read("Assignment_05_prob_02.ipynb", as_version=4)

# Remove widget metadata safely
if "widgets" in nb.metadata:
    del nb.metadata["widgets"]

nbformat.write(nb, "Assignment_05_prob_02.ipynb")