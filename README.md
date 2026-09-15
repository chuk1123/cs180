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
  proj1/              # add when Project 1 begins
  proj2/              # add when Project 2 begins
```

Each future project gets its own `docs/projN/` directory with an `index.html`,
optional project-specific stylesheet, and an `assets/` directory. Add a link to
the new project in `docs/index.html` when it is ready.
