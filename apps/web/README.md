# Web App Workspace

This directory is the canonical frontend workspace.

Current status:
- Frontend sources are owned by this workspace (`index.html` and `src/`).
- Vite dev/build/preview are run from this workspace.
- Legacy pages now live in `apps/web/legacy/`.

## Commands

From repository root:

```bash
npm --prefix apps/web install
npm --prefix apps/web run dev
```

Build and preview:

```bash
npm --prefix apps/web run build
npm --prefix apps/web run preview
```

## Notes

Flask now serves `/` from `apps/web/index.html` with a compatibility fallback to the repository root during migration.
Legacy URLs `/retrieval_tool.html` and `/index_explorer.html` are preserved via server aliases to files under `apps/web/legacy/`.
