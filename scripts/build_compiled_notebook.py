"""Compila el libro completo en un único cuaderno con todas sus salidas.

Concatena, en el orden del índice de ``myst.yml``, los cuadernos ejecutados de
``notebooks/`` y las páginas Markdown de ``docs/``. Las directivas
``literalinclude`` se sustituyen por el contenido del archivo citado y las
figuras MyST por imágenes Markdown, de modo que el resultado se lee en
Jupyter sin depender de MyST.

Uso (desde la raíz del repositorio):
    python scripts/build_compiled_notebook.py
"""

import base64
import re
from pathlib import Path

import nbformat
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "proyecto_heart_disease_mlops.ipynb"

FENCE = re.compile(r"```\{(\w+)\}\s*([^\n]*)\n(.*?)```", re.S)


def md_to_cells(path: Path) -> list:
    text = path.read_text(encoding="utf-8")
    title = None
    if text.startswith("---"):
        _, front, text = text.split("---", 2)
        title = yaml.safe_load(front).get("title")
    base = path.parent

    def repl(m):
        kind, arg, body = m.group(1), m.group(2).strip(), m.group(3)
        opts = dict(re.findall(r"^:(\w+):\s*(.*)$", body, re.M))
        rest = re.sub(r"^:\w+:.*\n?", "", body, flags=re.M).strip()
        if kind == "literalinclude":
            src = (base / arg).resolve()
            code = src.read_text(encoding="utf-8").strip("﻿").rstrip()
            lang = opts.get("language", "")
            caption = opts.get("caption", src.name)
            return f"*{caption}*\n\n```{lang}\n{code}\n```"
        if kind == "figure":
            # La imagen se incrusta como adjunto para que el cuaderno sea autocontenido.
            src = (base / arg).resolve()
            attachments[src.name] = {
                "image/png": base64.b64encode(src.read_bytes()).decode()}
            return f"![{rest}](attachment:{src.name})\n\n*{rest}*"
        return f"```{kind}\n{rest}\n```"

    attachments = {}
    text = FENCE.sub(repl, text).strip()
    if title:
        text = f"# {title}\n\n{text}"
    cell = nbformat.v4.new_markdown_cell(text)
    if attachments:
        cell["attachments"] = attachments
    return [cell]


def main():
    conf = yaml.safe_load((ROOT / "myst.yml").read_text(encoding="utf-8"))
    files = [entry["file"] for entry in conf["project"]["toc"]]

    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "heart-mlops", "language": "python",
                                 "display_name": "Python 3.10 (heart-mlops)"}
    nb.metadata["language_info"] = {"name": "python"}

    for f in files:
        path = ROOT / f
        if path.suffix == ".ipynb":
            src = nbformat.read(path, as_version=4)
            nb.cells.extend(src.cells)
        else:
            nb.cells.extend(md_to_cells(path))
        if f == files[0]:
            nb.cells.append(nbformat.v4.new_markdown_cell(
                "La celda siguiente permite reejecutar el cuaderno desde la "
                "raíz del repositorio; las salidas incluidas provienen de la "
                "ejecución de cada capítulo en `notebooks/`. Reejecutarlo "
                "regenera el modelo y los reportes."))
            nb.cells.append(nbformat.v4.new_code_cell(
                "import os\nimport sys\n\n"
                "os.chdir(\"notebooks\") if os.path.isdir(\"notebooks\") "
                "else None\nsys.path.insert(0, os.getcwd())"))

    for i, cell in enumerate(nb.cells):
        cell["id"] = f"cell-{i:03d}"
    nbformat.validate(nb)
    nbformat.write(nb, OUT)
    print(f"{OUT.name}: {len(nb.cells)} celdas")


if __name__ == "__main__":
    main()
