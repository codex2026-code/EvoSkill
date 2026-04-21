# Reproducible 3-benchmark experiment templates

本目录提供一套**统一且可复现**的模板，覆盖：
- SEAL-QA
- DabStep
- LiveCodeBench

并严格按当前项目实现逻辑区分：
- **SEAL-QA**：保持现有 CLI 流程 `run_loop_sealqa.py -> run_eval_sealqa.py`
- **DabStep / LiveCodeBench**：采用 `src.api.EvoSkill` 接口做 loop，再走各自 eval CLI

## 目录结构

```text
scripts/repro_templates/
├── run_all.sh
├── common/
│   ├── run_loop_evoskill.py
│   └── summarize_results.py
├── sealqa/
│   ├── run_control.sh
│   └── run_iter.sh
├── dabstep/
│   ├── run_control.sh
│   └── run_iter.sh
└── livecodebench/
    ├── run_control.sh
    └── run_iter.sh
```

## 统一输出命名规范

所有脚本遵循以下路径布局：

```text
experiments/repro_3bench/<RUN_TAG>/<task>/<config>/
├── skills_profile/
├── loop/iteration_log.json          # 仅迭代组有
└── eval/
    ├── results.pkl
    └── summary.json

experiments/repro_3bench/<RUN_TAG>/summary/all_runs.csv
```

- `<task>`: `sealqa | dabstep | livecodebench`
- `<config>`: `control_no_iter | iter_skill`
- `<RUN_TAG>` 默认 UTC 时间戳（可通过环境变量固定，便于复现实验）

## 如何运行

### 1) 一次跑完全部 3x2 配置

```bash
RUN_TAG=demo_001 bash scripts/repro_templates/run_all.sh
```

### 2) 按任务单独跑

```bash
# SEAL-QA
RUN_TAG=demo_001 bash scripts/repro_templates/sealqa/run_control.sh
RUN_TAG=demo_001 bash scripts/repro_templates/sealqa/run_iter.sh

# DabStep
RUN_TAG=demo_001 bash scripts/repro_templates/dabstep/run_control.sh
RUN_TAG=demo_001 bash scripts/repro_templates/dabstep/run_iter.sh

# LiveCodeBench
RUN_TAG=demo_001 bash scripts/repro_templates/livecodebench/run_control.sh
RUN_TAG=demo_001 bash scripts/repro_templates/livecodebench/run_iter.sh
```

## 默认参数与可覆写参数

通用可覆写环境变量：
- `EXP_ROOT`（默认 `experiments/repro_3bench`）
- `RUN_TAG`（默认 UTC 时间戳）
- `SDK`（默认 `claude`）
- `MODEL`（默认 `claude-opus-4-5-20251101`）
- `OPENAI_BASE_URL`（当 `SDK=openai` 时透传到 loop/eval；支持自部署 OpenAI 兼容网关）
- `OPENAI_API_KEY`（当 `SDK=openai` 时透传）
- `MAX_ITERATIONS`、`FRONTIER_SIZE`、`CONCURRENCY`、`FAILURE_SAMPLES`
- `EVAL_MAX_CONCURRENT`

SEAL-QA 统计额外支持：
- `GRADER_MODEL`（默认 `openai/gpt-5-mini`）
- `GRADER_BASE_URL`、`GRADER_API_KEY`（未设置时会回退到 `OPENAI_BASE_URL`/`OPENAI_API_KEY`）

## 统计汇总说明

`common/summarize_results.py` 会读取 eval 生成的 `results.pkl`，并输出：
- `total`, `successful`, `failed`, `scored`, `correct`, `accuracy`

同时可 `--append-csv` 追加到统一汇总表 `all_runs.csv`。

## 复现建议

为保证公平对比：
1. 对照组与迭代组使用相同模型/SDK。
2. 两组都使用 `--no-resume`（模板默认已启用）。
3. 每个配置独立 `skills_profile` 子目录，避免互相污染。
4. 固定 `RUN_TAG` 保存同批次实验的全量产物。
