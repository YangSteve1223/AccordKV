# Contributing to ACCORD-KV

Thank you for your interest in contributing to ACCORD-KV! This document provides guidelines and instructions for contributing.

## Development Environment Setup

### Prerequisites

- Python 3.9 or higher
- Git
- (Optional) CUDA-capable GPU for GPU experiment scripts

### Installation

```bash
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/<your-username>/AccordKV.git
cd AccordKV

# 3. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# 4. Install in development mode
pip install -e .

# 5. Install test dependencies
pip install pytest pytest-cov

# 6. Verify installation
python -c "from core import ACR, merge; print('Installation OK')"
```

### GPU Setup (Optional)

For GPU experiment scripts in `gpu/`:

```bash
pip install torch transformers vllm flash-attn --index-url https://download.pytorch.org/whl/cu118
```

Model weights are expected at `/root/autodl-tmp/` by default (modify paths in `gpu_model_loader.py` as needed).

## Code Style

### Python Style Guide

We follow **PEP 8** with the following additions:

- Line length: 100 characters (flexible for long imports/docstrings)
- Use type hints for all public function signatures
- Docstrings follow the **NumPy style** for functions and classes

### Docstring Format

```python
def merge_stats(a: AttnStats, b: AttnStats) -> AttnStats:
    """Merge two AttnStats objects.

    Performs numerically stable online-softmax state fusion.
    The merge is associative and commutative.

    Parameters
    ----------
    a : AttnStats
        First attention statistics object.
    b : AttnStats
        Second attention statistics object.

    Returns
    -------
    AttnStats
        Merged attention statistics.
    """
    ...
```

### Import Organization

Imports should be grouped in this order (one blank line between groups):

```python
# Standard library
import sys
from dataclasses import dataclass

# Third-party packages
import torch
import numpy as np

# Local project
from core.attn_stats import AttnStats, EPS
from simulation.exp8_svd_attention_sketch import SVDSketch
```

## Pull Request Workflow

### 1. Fork and Branch

```bash
# Create a feature branch from main
git checkout -b feature/your-feature-name

# Or a bugfix branch
git checkout -b fix/description-of-bug
```

Branch naming conventions:
- `feature/` - new features
- `fix/` - bug fixes
- `docs/` - documentation only
- `refactor/` - code refactoring without behavior change
- `exp/` - experiment scripts (non-production code)

### 2. Development

```bash
# Make your changes
# ...

# Run tests
pytest tests/ -v

# Run simulation tests (no GPU required)
python simulation/exp14_svd_mly_wire.py
python simulation/exp24_cluster_aware.py
```

### 3. Testing Requirements

**All existing tests must pass before opening a PR:**

```bash
# Run full test suite
pytest tests/ -v --tb=short

# Run with coverage
pytest tests/ --cov=. --cov-report=term-missing

# Simulation sanity checks
python simulation/exp14_svd_mly_wire.py
python simulation/exp24_cluster_aware.py
```

**New code should include tests** if:
- Adding new functions in `core/` or `simulation/`
- Changing mathematical logic in `merge.py` or compression algorithms
- Modifying the ACR protocol or contract types

Tests go in `tests/` (create if not present) or alongside modules as `test_*.py`.

### 4. Commit Your Changes

```bash
# Stage changes
git add .

# Commit with a descriptive message
git commit -m "feat(core): add new merge strategy for remote blocks

- Add merge_remote_exact() function for remote block merging
- Update AttnStats to support remote metadata
- Add unit tests for remote merge edge cases

Closes #42"
```

**Commit message format** (follows [Conventional Commits](https://www.conventionalcommits.org/)):

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `exp`, `chore`

### 5. Push and Create Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name

# Open a Pull Request on GitHub
```

**Pull Request checklist:**
- [ ] Tests pass (`pytest tests/`)
- [ ] Simulation scripts run without errors
- [ ] No new `print()` statements or debug code
- [ ] Type hints added for new functions
- [ ] Docstrings added for new public APIs
- [ ] README.md updated if adding new features that affect usage

## Issue Reporting

### Bug Reports

Please report bugs via GitHub Issues. Include:
- Python version, OS
- Steps to reproduce
- Expected vs actual behavior
- Full traceback if applicable
- Output of `pip list` for relevant packages

### Feature Requests

For new features, open a GitHub Issue with:
- Use case description
- Proposed API (if applicable)
- References to related work (papers, other repos)

## Questions?

For questions about contributing, open a GitHub Discussion or reach out via the issue tracker.
