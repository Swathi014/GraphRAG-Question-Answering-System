# Ensures pytest can import the `privgraphai` package regardless of whether
# it's invoked as `pytest tests/` or `python -m pytest tests/`. Without this
# file, plain `pytest tests/` fails with `ModuleNotFoundError: No module
# named 'privgraphai'` because pytest's default import mode only adds the
# `tests/` folder itself to sys.path, not its parent (this project's root,
# where the `privgraphai` package actually lives).
