# Repository Architecture

## Purpose

This document defines the target repository structure for FloorPlanRetrieval and the migration rules that should govern all future refactoring. The goal is to turn the current mixed root-level project into a clear monorepo with explicit ownership boundaries between the browser app, the HTTP API, shared floor-plan domain logic, offline tools, and non-code artifacts.

## Current Problem

The repository root currently mixes:

- browser runtime files
- Flask server entrypoints
- Python domain modules
- one-off tooling scripts
- generated assets and datasets
- prompts and developer docs
- workflow/config JSON files

That flat layout increases cognitive load, makes ownership unclear, and encourages accidental coupling between unrelated concerns. The issue is not that JavaScript and Python coexist in one repository; the issue is that runtime code, tooling, data, and documentation all share the same namespace.

## Architectural Decision

Keep a single repository, but split it into explicit bounded areas:

```text
/
  apps/
    api/
    web/
  packages/
    floorplan_core/
  tools/
  data/
    fixtures/
    index/
    outputs/
  configs/
    workflows/
  docs/
    architecture/
    dev/
```

## Ownership Model

### apps/web

Owns browser UX only.

Contains:
- Vite entrypoint and frontend package metadata
- HTML shell
- browser JS modules
- public static assets that belong to the web app

Must not contain:
- Python files
- datasets
- generated indexes
- server-only configuration

### apps/api

Owns HTTP only.

Contains:
- Flask app bootstrap
- route handlers
- server-specific configuration and startup scripts

Rules:
- Route handlers validate input, call domain services, and format responses.
- Route handlers must not contain retrieval, fitting, or rendering business logic inline.
- API code imports domain logic from `packages/floorplan_core`.

### packages/floorplan_core

Owns reusable floor-plan domain logic.

Contains:
- retrieval algorithms
- fitting logic
- geometry helpers
- index loading/query services
- render backend integrations
- request/response schema helpers that are not HTTP-framework-specific

Rules:
- No Flask app objects.
- No browser DOM code.
- Keep import surfaces explicit and small.

### tools

Owns offline and developer-only scripts.

Contains:
- index builders
- data exploration scripts
- export/import utilities
- one-shot migration helpers

Rules:
- Tools may depend on `packages/floorplan_core`.
- Tools must not be imported by runtime apps as core dependencies.

### data

Owns non-code artifacts.

Contains:
- fixtures and sample plans
- built indexes
- generated outputs
- reusable local development sample files

Rules:
- No executable source modules.
- Generated artifacts should land under a predictable subfolder.

### configs

Owns environment-agnostic configuration assets.

Contains:
- workflow JSON files
- manifests
- model pipeline config where appropriate

### docs

Owns human-facing documentation.

Contains:
- architecture documents
- development plans
- prompt/reference docs
- migration notes

## Target Mapping From Current Root

### Move to apps/web

- `index.html`
- `src/`
- `manifest.json`
- frontend package metadata

### Move to apps/api

- `apps/api/server.py`
- server-local config files

### Move to packages/floorplan_core

- `retrieval.py`
- `geo_ent_num_retrieval.py`
- `fitting.py`
- `fit_plan.py`
- `fit_rooms.py`
- `render_backend.py`
- `comfy_client.py`
- `request_parsing.py`
- `index_store.py`
- adaptation/retrieval support modules

### Move to tools

- `build_index_ent.py`
- `explore_index.py`
- `export_index.py`
- similar one-off or CLI-only scripts

### Move to data

- `test_data/` -> `data/fixtures/plans/`
- `browser_files/` -> `data/index/browser/`
- `adapted/` -> `data/outputs/adapted/`
- local sample json/png outputs

### Move to configs

- `comfyui-flux-depth-workflow.json`
- `comfyui-sdxl-depth-multi-workflow.json`
- similar workflow or manifest assets

### Move to docs

- `docs/architecture/system-overview.md`
- `docs/dev/`
- prompt/reference markdown files that are not runtime inputs

## Migration Status

The structural migration is complete:

- Frontend is owned by `apps/web`.
- API is owned by `apps/api/server.py`.
- Domain logic is owned by `packages/floorplan_core`.
- Offline tooling is owned by `tools/`.
- Non-code artifacts are owned by `data/` and `configs/`.
- Root now contains repository-level files and top-level directories only.

## Engineering Rules For The Migration

- Prefer additive, reversible changes over mass renames.
- Introduce new canonical paths before deleting old ones.
- Keep runtime behavior stable during each phase.
- Update documentation with every structural move.
- Do not mix generated artifacts with source code.
- Keep frontend and backend dependencies separate even inside one repo.

## Ongoing Maintenance

1. Keep `apps/api/server.py` as the only API entrypoint.
2. Avoid introducing new runtime source files in the repository root.
3. Keep docs synchronized whenever paths or ownership boundaries change.
4. Prefer package-level domain imports over cross-layer shortcuts.
