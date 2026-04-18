# Booster Deploy Policy Runtime Recovery Design

## Context
`tasks/local_crawl_dwaq/loco_crawl_dwaq.py` is one of the few surviving upgraded task implementations. It depends on a richer `Policy` base runtime that is currently missing from `booster_deploy/controllers/base_controller.py`.

Observed missing runtime surface from this task:
- `initialize_model_runtime(checkpoint_path)`
- `parse_onnx_outputs(outputs)`
- base ONNX runtime state (`_backend`, `_onnx_session`, `_onnx_input_names`, `_onnx_output_names`)
- structured logging hooks (`log_named_vector`, `log_stats`)
- robust path resolution for checkpoint loading

Without these, `t1_crawl_dwaq` cannot instantiate the policy or run inference.

## Goal
Reconstruct the previously upgraded policy runtime abstractions so that:
- `t1_crawl_dwaq` can run with both `.pt` and `.onnx` checkpoints,
- logging hooks used by DWAQ are available and stable,
- existing tasks (`locomotion`, `beyond_mimic`) keep working without behavior regressions,
- project governance for lint/test commands becomes consistent (`make ruff`, `make lint`, `make test`).

## Non-Goals
- Rebuilding every historical optimization that cannot be inferred from the remaining code.
- Changing task-level control logic beyond compatibility fixes.
- Reworking ROS/MuJoCo runtime architecture.

## Design
### 1) Recovered `Policy` Runtime Layer
Add a reusable model runtime layer inside `Policy`:
- backend detection by checkpoint suffix (`.onnx` vs TorchScript-like files),
- ONNX Runtime session init with configurable providers,
- TorchScript model loading to configured device,
- default ONNX input packaging and output parsing for single-input/single-output models,
- shared runtime attributes for task overrides (DWAQ custom two-input path).

### 2) Recovered Logging Layer
Add two logging APIs required by DWAQ:
- `log_named_vector(name, tensor_or_array)` for observation component instrumentation,
- `log_stats({name: value, ...})` for batched inference I/O summary metrics.

Logging behavior:
- disabled unless `PolicyCfg.stat_log_path` is set,
- auto-create parent directories,
- append CSV rows with `step`, `time_s`, `name`, `kind`, `shape`, `min`, `max`, `mean`, `std`,
- never crash inference if logging fails (best-effort).

### 3) Typed Config Recovery
Extend `PolicyCfg` with typed runtime/logging options used by upgraded tasks:
- `stat_log_path: str | None`
- `stat_log_inference_io: bool`
- `onnx_providers: list[str]`

This removes dynamic attribute mutation and makes task configs explicit.

### 4) Engineering Governance
Add repository-level development entrypoints:
- `Makefile` with `make ruff`, `make lint`, `make format`, `make test`,
- `pyproject.toml` with baseline `ruff` and `pytest` config.

## Compatibility Plan
- Preserve existing task custom loading paths (`locomotion` and `beyond_mimic` directly load TorchScript).
- New base runtime APIs are additive and used by DWAQ tasks.
- Maintain `Policy` constructor contract and `BaseController` lifecycle.

## Test Strategy (TDD)
1. Add unit tests that fail on current code:
- ONNX backend init sets session and io names.
- ONNX input preparation injects batch dim correctly.
- ONNX output parser returns torch tensor on configured device.
- Logging API writes CSV with expected columns.

2. Implement minimal code to pass these tests.
3. Re-run targeted tests, then full test suite and lint.

## Risks and Mitigations
- ONNX provider mismatch across environments: default to CPU provider and keep provider list configurable.
- Logging overhead in control loop: summary stats only; opt-in by config.
- Backward compatibility: avoid changing existing task inference codepaths.
