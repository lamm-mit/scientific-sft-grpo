from __future__ import annotations

import ast
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    failures = []
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        artifact = nbformat.read(path, as_version=4)
        code_cells = [cell for cell in artifact.cells if cell.cell_type == "code"]
        for index, cell in enumerate(code_cells):
            try:
                ast.parse(cell.source, filename=f"{path.name}:code-cell-{index}")
            except SyntaxError as error:
                failures.append(str(error))
        if any(cell.get("outputs") for cell in code_cells):
            failures.append(f"{path.name}: generated notebook contains saved outputs")
        print(f"{path.name}: {len(artifact.cells)} cells, {len(code_cells)} code cells")
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
