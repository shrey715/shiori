"""Execute the notebook from the top up to (not including) a markdown heading, and save the outputs back.

Lets earlier sections run without the later ones, e.g. sections 1-4 only:
    python scripts/run_notebook.py --until "## 5. Failure analysis"
If a cell fails, the outputs up to that cell are still saved and the exit code is 1.
"""
import argparse
import copy
import sys
import time

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

parser = argparse.ArgumentParser()
parser.add_argument("--notebook", default="rag_reranking.ipynb")
parser.add_argument("--until", default=None, help="markdown heading to stop before, e.g. '## 5. Failure analysis'")
args = parser.parse_args()

nb = nbformat.read(args.notebook, 4)
end = len(nb.cells)
if args.until:
    end = next(k for k, c in enumerate(nb.cells) if c.cell_type == "markdown" and c.source.startswith(args.until))
part = copy.deepcopy(nb)
part.cells = part.cells[:end]

t0 = time.time()
status = 0
try:
    NotebookClient(part, kernel_name="rag_assignment", timeout=-1, resources={"metadata": {"path": "."}}).execute()
except CellExecutionError as e:
    status = 1
    print(str(e)[:3000])
nb.cells[:end] = part.cells
nbformat.write(nb, args.notebook)
print(f"ran cells 0-{end - 1} in {time.time() - t0:.0f}s, exit {status}")
sys.exit(status)
