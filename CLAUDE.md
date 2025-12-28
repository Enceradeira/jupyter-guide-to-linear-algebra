# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational Python package for learning linear algebra concepts. Contains reusable library functions and practical exercise examples built on numpy/scipy.

The project closely follows the content at https://bvanderlei.github.io/jupyter-guide-to-linear-algebra/Bases.html. Solutions to the exercises are partly under https://bvanderlei.github.io/jupyter-guide-to-linear-algebra/Solutions.html

## Development Commands

```bash
# Install dependencies (requires uv package manager)
uv sync

# Run all tests
uv run pytest

# Run a specific test file
uv run pytest tests/test_solve_system.py

# Run a specific test
uv run pytest tests/test_solve_system.py::test_rref_1

# Run an exercise file
uv run python exercises/linear_combinations.py
```

## Architecture

### Source Code (`src/guide_to_linear_algebra/`)

- **laguide.py** - Core linear algebra utilities: row operations (RowSwap, RowScale, RowAdd), matrix operations (DotProduct, Magnitude, Inverse), QR factorization, and graph visualization with networkx
- **solve_system.py** - System solving: BackSubstitution, RowReduction, SolveSystem, RREF, Inverse
- **general_linear_systems_functions.py** - Analysis functions: find_pivots, nr_free_variables, has_no_solution
- **cryptography.py** - Matrix-based encryption using modular arithmetic
- **plt_helper.py** - Matplotlib helper for saving/displaying plots

### Exercises (`exercises/`)

Standalone Python scripts demonstrating linear algebra concepts: linear combinations, vector spaces, LU factorization, interpolation, graph theory applications. Each file is runnable independently.

### Tests (`tests/`)

Pytest-based tests covering solve_system and general_linear_systems_functions modules.

## Dependencies

Python >= 3.12. Runtime dependencies (numpy, scipy, matplotlib, networkx) are in the `dev` dependency group - install with `uv sync`.

## Commit Guidelines
- Use commit messages that span a single line, unless told otherwise.
