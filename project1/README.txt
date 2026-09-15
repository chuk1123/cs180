CS180 Project 1

Setup from repository root:
  uv venv .venv --python 3.10
  uv pip install --python .venv/bin/python -r requirements.txt

Place the provided JPG/TIFF glass plates in project1/data, then run:
  .venv/bin/python project1/main.py

Results are written as JPG files in project1/output. The program prints offsets
in NumPy order (dy, dx); the webpage reports the same offsets as (x, y).

Run the optional Emir raw/gradient comparison with:
  .venv/bin/python project1/main.py --emir-experiments

Project webpage:
  https://chuk1123.github.io/cs180/proj1/
