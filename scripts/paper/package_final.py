"""Package only the files referenced by the current focused manuscript."""

from pathlib import Path
import re
from zipfile import ZipFile

from scripts.paper.prepare_final import REPORT_FIGURES


def package(root):
    source = (root / "main.tex").read_text()
    figures = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^{}]+)\}", source)
    if set(figures) != {f"{name}.pdf" for name in REPORT_FIGURES}:
        raise ValueError("Manuscript must include only the two dense training figures")
    inputs = re.findall(r"\\input\{([^{}]+)\}", source)
    if any("seeds" in name for name in inputs):
        raise ValueError("Seed-comparison table is outside publication scope")
    paths = [Path("main.tex"), *map(Path, inputs), *(Path("figures") / name for name in figures)]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate manuscript asset")
    for path in paths:
        if not (root / path).resolve().is_relative_to(root.resolve()) or not (root / path).is_file():
            raise ValueError(f"Missing or invalid manuscript asset: {path}")
    with ZipFile(root / "final_overleaf.zip", "w") as archive:
        for path in paths:
            archive.write(root / path, str(path))
    return paths


if __name__ == "__main__":
    root = Path("docs/final")
    paths = package(root)
    print(f"Packaged {len(paths)} referenced manuscript assets in {root / 'final_overleaf.zip'}")
