# 🦅 Poetry Local Setup

---

## ❖ "Don't solve the setup. Win it."

### 1. Install Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

If it doesn't work, you're not on the field.  
Also, add this to your shell:

- If you are using Zsh:
```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

- If you're using Bash:
```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bash_profile
source ~/.bash_profile
```

Test it:

```bash
poetry --version
```

---

### 2. Configure Project-Local Environment

> "We don't share benches. We fight alone, inside the box."

```bash
poetry config virtualenvs.in-project true
```

If you want to scope it to this repo only:

```bash
poetry config --local virtualenvs.in-project true
```

If needed, erase the existing poetry env:
```bash
poetry env remove python
```

---

### 3. Initialize the Field

```bash
poetry init
# or
poetry init --no-interaction
```

---

### 4. Lock the Formation

```bash
poetry lock
```

> "No second-guessing. Lock it down. Then run."

---

### 5. Install Your Winning Squad

```bash
poetry install
```

---

### 6. Add Dev Dependencies from requirements-dev.txt

```bash
poetry add --group dev $(< requirements-dev.txt)
```

Re-lock after major changes:

```bash
poetry lock
poetry install
```

---

### 7. Confirm the `.venv`

```bash
poetry env info --path
```

→ should be inside your project root, like:

```
./.venv
```

> "I shoot from this turf only."

---

### 8. PyCharm Sync

Use the `.venv/bin/python` as your interpreter.  
No excuses. Set it.

---

## ❖ When Updating Package Versions

### ✅ Step-by-step:

1. **Regenerate the lock file completely:**

```bash
poetry lock --no-cache --no-update --check
# ↑ preview only (optional)

poetry lock --no-cache --no-update --regenerate
```

2. **Install freshly with new lock:**

```bash
poetry install
```

---

## ❖ Run Coverage with Poetry

### ✅ Run like this:

```bash
poetry run coverage erase && poetry run pytest -v --cov=src --cov-report=html tests/
```

This does:

- Erase old coverage data (no carry-overs from past matches)
- Run all tests in `tests/` directory
- Measure coverage for `src/`
- Output a visual report in `htmlcov/`

To open the coverage result:

```bash
open htmlcov/index.html  # macOS
# or
xdg-open htmlcov/index.html  # Linux
```

---

## ❖ Handling Platform-Specific Dependencies (e.g., `jaxlib`)

> Define your battlefield. Target your architecture. Victory starts with clarity.

### ✅ Installing `jaxlib` for x86_64 Only:

Add below to `pyproject.toml`:

```toml
[tool.poetry.group.experimental.dependencies]
jax = { version = "0.4.23", markers = "platform_machine == 'x86_64'" }
```

Reconstruction of the poetry virtual env:

```bash
rm poetry.lock
poetry lock --no-cache --regenerate
poetry install --with experimental
```

> When restricting jaxlib to x86_64 architectures, make your intention explicit. This avoids unnecessary conflicts on Apple Silicon or ARM environments. 

---

## 🔁 Regenerate Poetry Lock (For Poetry 2.1.3)

To regenerate poetry.lock based on your current pyproject.toml:

```bash
rm poetry.lock
poetry lock --no-cache
```

**Note:**
- The `--check` and `--no-update` options are not available in Poetry 2.1.3.
- The `--no-cache` flag forces Poetry to bypass cache and resolve dependencies using fresh metadata.

> In other words — when in doubt, rebuild with full awareness of your environment.

---
