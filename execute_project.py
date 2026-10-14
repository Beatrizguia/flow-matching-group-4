"""Execute notebooks with the invoking Python environment and save outputs.

Usage: python execute_project.py notebook1.ipynb [notebook2.ipynb ...].
Each notebook runs in a fresh kernel with its own parent directory as cwd,
so local imports and data paths do not depend on where this command starts.
The supplied notebook files are updated in place, including partial/error
outputs on failure. This script runs its command-line loop at module level;
use it as an entry point, not as a library import.
"""

import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager, KernelSpec


class ThisPython(KernelSpecManager):
    """Use the active interpreter instead of an unrelated registered Jupyter kernel."""

    def get_kernel_spec(self, kernel_name):
        """Launch ipykernel through sys.executable to match installed dependencies."""
        return KernelSpec(
            argv=[
                sys.executable,
                "-m",
                "ipykernel_launcher",
                "-f",
                "{connection_file}",
            ],
            display_name="Project Python",
            language="python",
        )


for filename in sys.argv[1:]:
    path = Path(filename).resolve()
    notebook = nbformat.read(path, as_version=4)
    manager = KernelManager(kernel_name="python3", kernel_spec_manager=ThisPython())

    def persist(cell, cell_index, **kwargs):
        """Save cell progress so a later failure does not discard earlier outputs.

        NotebookClient supplies the cell and zero-based cell_index. Additional hook
        arguments are accepted for API compatibility; final persistence also occurs
        in finally, followed by kernel shutdown.
        """
        nbformat.write(notebook, path)
        print(f"{filename}: finished cell {cell_index+1}", flush=True)

    client = NotebookClient(
        notebook,
        km=manager,
        timeout=3600,
        resources={"metadata": {"path": str(path.parent)}},
        on_cell_executed=persist,
    )
    try:
        client.execute()
    finally:
        nbformat.write(notebook, path)
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    print("FINISHED", filename, flush=True)
