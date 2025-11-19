# Contributing to GroundedDINO-VL

Thank you for your interest in contributing to **GroundedDINO-VL**! This document provides guidelines and best practices for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Code Style](#code-style)
- [License](#license)
- [Getting Help](#getting-help)

## Code of Conduct

This project adheres to a Contributor Code of Conduct. By participating, you are expected to uphold this code. Please be respectful, inclusive, and professional in all interactions with other contributors and maintainers.

**Expected Behavior:**
- Use welcoming and inclusive language
- Be respectful of differing opinions and experiences
- Accept constructive criticism gracefully
- Focus on the code, not the person

**Unacceptable Behavior:**
- Harassment or discrimination of any kind
- Offensive language or comments
- Personal attacks or trolling
- Unwelcome sexual attention

## Getting Started

### Prerequisites

Before you begin, ensure you have:
- **Python**: 3.9 or higher
- **Git**: For version control
- **C++ Compiler**: GCC 7+, Clang 5+, or MSVC 2019+
- **CUDA Toolkit** (optional): 12.6 or 12.8 for GPU support
- **PyTorch**: 2.7+ (installed during development setup)

### Fork and Clone

1. **Fork the repository** on GitHub (click the "Fork" button)

2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/GroundedDINO-VL.git
   cd GroundedDINO-VL
   ```

3. **Add upstream remote** to stay synchronized:
   ```bash
   git remote add upstream https://github.com/ghostcipher1/GroundedDINO-VL.git
   ```

4. **Verify remotes**:
   ```bash
   git remote -v
   # origin    https://github.com/YOUR-USERNAME/GroundedDINO-VL.git (fetch)
   # origin    https://github.com/YOUR-USERNAME/GroundedDINO-VL.git (push)
   # upstream  https://github.com/ghostcipher1/GroundedDINO-VL.git (fetch)
   # upstream  https://github.com/ghostcipher1/GroundedDINO-VL.git (push)
   ```

## Development Setup

### Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Verify Python version
python --version  # Should be 3.9+
```

### Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip setuptools wheel

# Install PyTorch (CPU - recommended for development without GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Or install with CUDA 12.8 support (if you have NVIDIA GPU)
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install runtime dependencies
pip install -r requirements.txt

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Set Up CUDA (Optional - for GPU Development)

If building with CUDA support:

```bash
# Set CUDA environment variables (Linux/macOS)
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# On Windows, set via System Properties or command prompt
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set PATH=%CUDA_HOME%\bin;%PATH%
```

### Verify Installation

```bash
# Test basic imports
python -c "import groundeddino_vl; print(f'Version: {groundeddino_vl.__version__}')"

# Run test suite
pytest tests/test_import.py -v
```

## Making Changes

### Branch Naming Convention

Use descriptive branch names that indicate the type of change:

```
feature/add-onnx-export       - New feature
fix/cuda-extension-build      - Bug fix
docs/update-readme            - Documentation
refactor/simplify-inference   - Code refactoring
test/add-coverage-tests       - Tests
chore/update-dependencies     - Maintenance
```

**Branch Naming Rules:**
- Use lowercase with hyphens
- Be specific and descriptive
- Include issue number if applicable: `fix/cuda-build-#123`

### Commit Messages

Follow the **Conventional Commits** specification:

```
type(scope): brief description (50 chars max)

Optional longer description explaining the change (wrap at 72 chars).
Include any relevant context, motivation, or implications.

Fixes #issue-number
Closes #another-issue
See also #related-issue
```

**Type Categories:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style (formatting, missing semicolons, etc.)
- `refactor`: Code refactoring without changing functionality
- `perf`: Performance improvements
- `test`: Test additions or changes
- `chore`: Maintenance, dependency updates, etc.
- `ci`: CI/CD configuration changes

**Examples:**

Good commit messages:
```
feat(models): add support for custom backbone architectures

Users can now specify alternative backbones via configuration.
This enables greater flexibility in model composition.

Fixes #123
```

```
fix(build): resolve C++17 compiler detection on macOS

The previous detection logic didn't account for Clang's version
reporting format. This fix properly parses Clang versions.

Closes #456
```

### Code Changes

**Do:**
- ✅ Write clear, focused changes
- ✅ One feature/fix per commit
- ✅ Add tests for new functionality
- ✅ Update documentation
- ✅ Keep commits atomic and logical

**Don't:**
- ❌ Mix multiple unrelated changes
- ❌ Include whitespace-only changes in feature PRs
- ❌ Commit generated files or build artifacts
- ❌ Force-push after PR is submitted (unless requested)

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_import.py -v

# Run with coverage report
pytest tests/ --cov=groundeddino_vl --cov=groundingdino --cov-report=html

# Run with detailed output
pytest tests/ -vv --tb=short

# Run tests matching a pattern
pytest tests/ -k "import" -v
```

### Writing Tests

**Test Structure:**
- Place tests in `tests/` directory
- Name test files: `test_*.py`
- Name test functions: `test_*`
- Use descriptive names that explain what is being tested

**Best Practices:**
```python
def test_model_forward_pass():
    """Test that model performs forward pass with valid inputs.

    This ensures the model can process a batch of images without errors.
    """
    from groundeddino_vl.utils import inference

    # Arrange
    model = build_model(config)
    input_tensor = torch.randn((1, 3, 512, 512))

    # Act
    output = model(input_tensor)

    # Assert
    assert output is not None
    assert output.shape[0] == 1  # Batch size
```

**Test Guidelines:**
- Use descriptive assertions with messages: `assert x == y, "Expected x to equal y"`
- Test both success and failure cases
- Avoid hardcoding paths; use fixtures or temporary directories
- Mock external dependencies when appropriate
- Keep tests fast (< 1s per test if possible)

### Code Quality Checks

Before committing, run these checks:

```bash
# Format code with Black
black groundeddino_vl groundingdino tests

# Sort imports with isort
isort groundeddino_vl groundingdino tests

# Check style with flake8
flake8 groundeddino_vl groundingdino tests --max-line-length=100

# Type checking with mypy
mypy groundeddino_vl --ignore-missing-imports

# Security check with bandit
bandit -r groundeddino_vl groundingdino

# Dependency vulnerabilities
safety check --file requirements.txt
```

**Or run all checks at once:**
```bash
# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install

# Manually run all checks
black --check groundeddino_vl groundingdino tests
isort --check-only groundeddino_vl groundingdino tests
flake8 groundeddino_vl groundingdino tests --max-line-length=100
mypy groundeddino_vl --ignore-missing-imports
```

## Submitting Changes

### Before Creating a Pull Request

1. **Sync with upstream**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all tests locally**:
   ```bash
   pytest tests/ -v
   ```

3. **Run code quality checks**:
   ```bash
   black --check groundeddino_vl groundingdino tests
   isort --check-only groundeddino_vl groundingdino tests
   flake8 groundeddino_vl groundingdino tests
   mypy groundeddino_vl --ignore-missing-imports
   ```

4. **Update documentation** if you made changes to:
   - Public APIs
   - Behavior or configuration
   - Installation process

5. **Verify no sensitive information** in your changes

### Creating a Pull Request

1. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create PR on GitHub**:
   - Go to https://github.com/YOUR-USERNAME/GroundedDINO-VL
   - Click "Compare & pull request"
   - Compare `main` branch against your feature branch

3. **Fill out PR description** with:
   - Clear title describing the change
   - Motivation and context
   - Type of change (bugfix, feature, documentation, etc.)
   - How to test the changes
   - Screenshots (if UI-related)
   - Related issue numbers: `Fixes #123`

4. **PR Title Format**:
   ```
   feat: add ONNX export support
   fix: resolve CUDA extension build on Apple Silicon
   docs: update installation guide
   ```

5. **Wait for checks**:
   - GitHub Actions CI/CD must pass
   - Code review from maintainers
   - Automated tests and linting checks

6. **Respond to feedback**:
   - Address all review comments
   - Push new commits (don't force-push unless requested)
   - Re-request review after making changes

### Pull Request Checklist

Before submitting, ensure:

- [ ] Branch is up-to-date with `upstream/main`
- [ ] All tests pass locally: `pytest tests/ -v`
- [ ] Code follows style guidelines (Black, isort, flake8)
- [ ] New features have corresponding tests
- [ ] Documentation is updated
- [ ] Commit messages follow convention
- [ ] No hardcoded paths, credentials, or sensitive data
- [ ] PR description is clear and complete
- [ ] No merge conflicts with main branch
- [ ] Changes are focused and atomic

## Code Style

### Python Code Style

**Tools & Configuration:**
- **Formatter**: Black (line length 100)
- **Import Sorting**: isort (Black-compatible profile)
- **Linter**: flake8
- **Type Hints**: Use for public APIs
- **Python Target**: 3.9+

**Configuration** (in `pyproject.toml`):
```toml
[tool.black]
line-length = 100
target-version = ["py39", "py310", "py311", "py312"]

[tool.isort]
profile = "black"
line_length = 100

[tool.mypy]
python_version = "3.9"
ignore_missing_imports = true
```

### Code Organization

**Module Structure:**
```
groundeddino_vl/
├── models/           # Model implementations
├── utils/            # Utility functions
├── data/             # Data loading and transforms
├── ops/              # CUDA operations
└── api/              # High-level API
```

**Function/Class Design:**
- Keep functions small and focused (< 50 lines ideal)
- Use descriptive names for variables and functions
- Add docstrings to all public functions and classes
- Comment complex logic (the "why", not the "what")
- Avoid deeply nested code (max 3 levels)

### Docstring Format

Use **Google-style docstrings** for consistency:

```python
def load_model(config_path: str, checkpoint_path: str, device: str = "cuda") -> torch.nn.Module:
    """Load a GroundingDINO model from config and checkpoint.

    Args:
        config_path: Path to model configuration file (.py)
        checkpoint_path: Path to model weights (.pth)
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        Loaded model in evaluation mode

    Raises:
        FileNotFoundError: If config or checkpoint not found
        RuntimeError: If model loading fails
        ValueError: If config is invalid

    Example:
        >>> model = load_model("config.py", "weights.pth")
        >>> model.eval()
    """
    ...
```

**Docstring Sections:**
- **Summary**: One-line description
- **Description**: (Optional) Longer explanation
- **Args**: Parameter descriptions with types
- **Returns**: Return value description
- **Raises**: Exceptions that may be raised
- **Example**: (Optional) Usage example

### Type Hints

Use type hints for public APIs:

```python
from typing import List, Optional, Tuple
import torch

def process_images(
    images: List[str],
    model: torch.nn.Module,
    batch_size: int = 32,
) -> Tuple[torch.Tensor, List[str]]:
    """Process images and return detections."""
    ...
```

## Important Guidelines

### License and Attribution

This project is licensed under **Apache License 2.0**.

**Requirements:**
- All contributions must be compatible with Apache 2.0
- Maintain attribution to original GroundingDINO project
- Include proper copyright headers in new files

**Copyright Header:**
```python
"""
GroundedDINO-VL - Vision Language Model

Copyright (c) 2025 GhostCipher. All rights reserved.
Based on GroundingDINO by IDEA-Research.
Licensed under Apache License 2.0.
"""
```

### What NOT to Include

- ❌ **Proprietary code** or incompatible licenses
- ❌ **Large binary files** (> 100MB)
- ❌ **Credentials or API keys** in code or commits
- ❌ **Hardcoded paths** or system-specific configurations
- ❌ **Temporary or build files** (.pyc, __pycache__, dist/, etc.)
- ❌ **Personal information** of contributors

### Code Review Process

1. **Submission**: Create PR with clear description
2. **Automated Checks**: GitHub Actions runs tests and linting
3. **Manual Review**: Maintainers review code for:
   - Correctness and quality
   - Documentation completeness
   - Test coverage
   - Performance implications
4. **Approval**: At least one maintainer approval required
5. **Merge**: Maintainers handle merging (contributors don't self-merge)

**Review Etiquette:**
- Assume good intent
- Ask questions rather than making demands
- Suggest improvements with examples
- Be specific in feedback

## Development Tips

### Building with CUDA

```bash
# Set CUDA environment (Linux/macOS)
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH

# Build with verbose output
python setup.py build_ext --inplace -vv
```

### Debugging

**Common Issues:**

| Issue | Solution |
|-------|----------|
| Import errors | Run `pip install -e .` in editable mode |
| CUDA compilation errors | Check `CUDA_HOME` env var, verify C++17 compiler |
| Test failures | Ensure `PYTHONPATH` includes project root |
| Extension not found | Run `python -m build --no-isolation` |

**Debug Commands:**

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Verify extension is built
python -c "from groundeddino_vl.ops import _C; print(_C)"

# Check Python path
python -c "import sys; print(sys.path)"

# Run test with detailed output
pytest tests/ -vv --tb=long --log-cli-level=DEBUG
```

## Getting Help

### Resources

- **Issues**: Open a GitHub issue for bugs or feature requests
  - Use issue templates provided
  - Include reproduction steps for bugs
  - Link related PRs or issues

- **Discussions**: Use GitHub Discussions for:
  - Questions about usage
  - Architecture discussions
  - Feature ideas

- **Documentation**:
  - [README.md](README.md) - Project overview and quick start
  - [BUILD_GUIDE.md](BUILD_GUIDE.md) - Detailed build instructions
  - [docs/](docs/) - Complete documentation

### Community

- **GitHub**: Primary communication channel
- **Issues**: Bug reports and feature requests
- **Pull Requests**: Discuss changes directly in PRs
- **Discussions**: Use for open-ended questions

## Recognition

Contributors make this project possible! We recognize contributions in:

- **GitHub contributors page** - Automatic
- **Release notes** - For significant contributions
- **Project documentation** - For major features
- **CONTRIBUTORS.md** - (To be added)

Thank you for contributing to GroundedDINO-VL! 🚀

---

**Questions?** Feel free to reach out by opening an issue or discussion on GitHub.

**Want to contribute?** Start with issues marked `good first issue` or `help wanted`.
