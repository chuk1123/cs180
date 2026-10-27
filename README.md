# CS180 Projects

Course website for CS180: Intro to Computer Vision and Computational Photography.

## Python environment

Use the shared `.venv` in the repository root for class projects (Python 3.10).
Dependencies cover the Project 1 starter code, including Matplotlib for image display.

```sh
source .venv/bin/activate
```

To recreate the environment using uv:

```sh
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python -r requirements.txt
```

Select `.venv/bin/python` as your IDE's Python interpreter. Run `deactivate`
to leave the environment.

## Website structure

GitHub Pages serves the `docs/` directory:

```text
docs/
  index.html          # course-project landing page
  site.css            # shared landing-page styles
  proj0/              # Project 0 page and its assets
  proj1/              # Project 1 report and assets
  proj2/              # Project 2 report and assets
  proj3/              # Project 3 report and assets
  proj4/              # Project 4 report and assets
```

Each future project gets its own `docs/projN/` directory with an `index.html`,
optional project-specific stylesheet, and an `assets/` directory. Add a link to
the new project in `docs/index.html` when it is ready.

## Local project folders

Only `docs/` is committed. Everything else stays on this machine, because the
course asks students not to host project code publicly.

```text
project0/      # camera originals (dolly zoom, etc.)
project1/      # Project 1 code, data, outputs
project2/      # Project 2 code (main.py), inputs in data/, results in output/
project3/      # Project 3 notebooks: partA/ (flow models), partB/ (flow matching)
project4/      # Project 4 code; photos in data/, results in output/ (see project4/README.md)
submissions/   # what was turned in, e.g. submissions/project2/submission.zip
```

