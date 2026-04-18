# Policy Runtime Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the missing upgraded policy runtime so `t1_crawl_dwaq` supports stable PT/ONNX inference and built-in runtime logging.

**Architecture:** Reconstruct additive runtime helpers in `Policy` and typed config fields in `PolicyCfg`, then validate via unit tests and repository-level lint/test entrypoints.

**Tech Stack:** Python 3.10, PyTorch, ONNX Runtime, unittest, ruff, make.

---

### Task 1: Lock regression tests first (RED)

**Files:**
- Create: `tests/controllers/test_policy_runtime.py`

- [ ] **Step 1: Write failing tests for ONNX runtime init and IO metadata**
- [ ] **Step 2: Write failing tests for ONNX input/output tensor adaptation**
- [ ] **Step 3: Write failing tests for CSV stats logging APIs**
- [ ] **Step 4: Run targeted tests and confirm failure is due to missing runtime methods**

### Task 2: Restore base policy runtime (GREEN)

**Files:**
- Modify: `booster_deploy/controllers/base_controller.py`

- [ ] **Step 1: Add checkpoint path resolution and backend initialization methods**
- [ ] **Step 2: Add ONNX helper methods (`prepare_onnx_inputs`, `parse_onnx_outputs`)**
- [ ] **Step 3: Add logging helpers (`log_named_vector`, `log_stats`) with CSV append**
- [ ] **Step 4: Keep logging best-effort and non-fatal in control loop**

### Task 3: Recover typed runtime config

**Files:**
- Modify: `booster_deploy/controllers/controller_cfg.py`

- [ ] **Step 1: Add explicit policy logging/runtime fields in `PolicyCfg`**
- [ ] **Step 2: Preserve default behavior for existing tasks**

### Task 4: Add engineering governance commands

**Files:**
- Create: `Makefile`
- Create: `pyproject.toml`

- [ ] **Step 1: Add `make ruff`, `make lint`, `make format`, `make test` targets**
- [ ] **Step 2: Add baseline lint/test config for consistent local execution**

### Task 5: Verify and ship

**Files:**
- Modify as needed from previous tasks

- [ ] **Step 1: Run `make ruff` in `booster_deploy` conda env**
- [ ] **Step 2: Run unit tests (unittest discovery in `tests/controllers`)**
- [ ] **Step 3: Create branch, commit changes, push to `Mr-tooth/booster_deploy`**
- [ ] **Step 4: Open PR and include verification evidence**
