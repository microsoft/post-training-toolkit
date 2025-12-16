import nbformat as nbf
from pathlib import Path
import sys


def main() -> None:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell("""# RLHF Diagnostics Demo

This minimal notebook:
- Shows a few lines of the raw TRL-style log
- Runs the diagnostics engine on the log
- Renders the report and displays a few plots
"""))

    cells.append(nbf.v4.new_code_cell("""from itertools import islice
from pathlib import Path

log_path = Path("../demo_logs/run_001.jsonl").resolve()
print("Log path:", log_path)

print("\\nFirst 5 lines of log:")
with open(log_path, "r", encoding="utf-8") as f:
    for line in islice(f, 5):
        print(line.strip())"""))

    cells.append(nbf.v4.new_code_cell("""from pathlib import Path
import sys
sys.path.append(str(Path("..").resolve()))  # ensure parent on sys.path

from diagnostics import run_diagnostics

reports_dir = Path("../reports").resolve()
out_path = run_diagnostics(log_path, reports_dir, make_plots=True)
print("Report written to:", out_path)"""))

    cells.append(nbf.v4.new_code_cell("""print("\\nReport preview (first 40 lines):")
with open(out_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 40:
            break
        print(line.rstrip())"""))

    cells.append(nbf.v4.new_code_cell("""from IPython.display import Image, display
from pathlib import Path

plots_dir = reports_dir / "plots"
for name in ["reward.png", "kl.png", "drift.png", "slices.png"]:
    p = plots_dir / name
    if p.exists():
        print(name)
        display(Image(filename=str(p)))"""))

    nb["cells"] = cells

    out = Path(__file__).parent / "demo.ipynb"
    nbf.write(nb, out)
    print(f"Wrote notebook: {out}")


if __name__ == "__main__":
    main()


