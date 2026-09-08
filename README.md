# fall-precursor

Skeleton-based fall precursor detection. See `implementation_plan.md` for the full design and build order, and `STYLE.md` for coding conventions.

## Setup (Windows, uv)

Install uv if not already installed. Open PowerShell:

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

Close and reopen the terminal so `uv` is on PATH, then confirm:

```powershell
uv --version
```

From the project root, create the environment and install dependencies:

```powershell
uv sync
```

This reads `pyproject.toml`, creates `.venv`, and installs everything, including the `dev` group. To include dev dependencies explicitly:

```powershell
uv sync --extra dev
```

Run any script through uv so it uses the project environment without manual activation:

```powershell
uv run python scripts/01_extract_poses.py
```

Run tests:

```powershell
uv run pytest
```

Add a new dependency later:

```powershell
uv add <package-name>
```

## Project layout

See `implementation_plan.md` for the full structure and module responsibilities. In short: `src/fallprecursor/` holds all logic, `scripts/` holds thin CLI entry points that call into it, `configs/` holds every experiment's settings, `tests/` covers the pure-logic modules (labeling, windowing, splits, lead-time metric).
