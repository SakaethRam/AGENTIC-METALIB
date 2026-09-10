# METALIB: AGENTIC DATA FLOW

MetaLib is an evidence-grounded customer-support intelligence system. It learns
from historical support conversations, classifies incoming intent, retrieves
similar resolutions, drafts a brand-consistent response, and decides when the
case must be escalated to a human.

This repository contains the presentation layer, a Render-ready API boundary,
the modular intelligence core, a CLI demonstration, and a reproducible
evaluation harness. It is designed for the Hiver SDE Intern take-home
assignment.

<img width="1832" height="862" alt="LIB-#1" src="https://github.com/user-attachments/assets/69a65282-934a-42d0-8a17-0344855e5a39" />

<img width="1852" height="848" alt="LIB-#2" src="https://github.com/user-attachments/assets/ae52c984-e16d-48e5-bd18-34ce47835c74" />

---

## Why it exists

Support automation should not only generate an answer. It should understand
the evidence, make the routing decision, and make that decision inspectable.
MetaLib keeps those stages explicit so the system can be evaluated, replaced,
and explained.

---

## Quick start

The zero-setup CLI path is intentionally dependency-light:

```bash
git clone <repository>
cd metlib
./run
```

The command creates a local virtual environment, loads the included golden
set, runs a duplicate-charge demonstration through the pipeline, evaluates the
dataset, and writes machine-readable output to `evaluation/results.json`.

The web presentation runs in the included pnpm workspace:

```bash
pnpm install
pnpm --filter @workspace/metalib run dev
```

The API server is a separate process:

```bash
pnpm --filter @workspace/api-server run dev
```

---

## Architecture

```text
Historical Twitter conversations
  → cleaning / thread reconstruction
  → brand selection and intent taxonomy
  → historical resolution index
  → intent classification
  → evidence retrieval
  → grounded response generation
  → escalation decision
  → evaluation
```

The frontend lives at `/home` in the product brief and is served at the root
route in this workspace. It uses the typed API contract for pipeline overview,
demo analysis, and evaluation summary surfaces.

The API contract is source-controlled in `lib/api-spec/openapi.yaml`. Run
`pnpm --filter @workspace/api-spec run codegen` after changing it. The Express
server currently exposes:

* `GET /api/healthz`
* `GET /api/pipeline/overview`
* `POST /api/pipeline/analyze`
* `GET /api/evaluation/summary`

---

## CLI and AI architecture

`metlib_core/pipeline.py` keeps the pipeline stages replaceable:

* `classify_intent` — deterministic demo classifier
* `retrieve_evidence` — transparent historical evidence lookup
* `draft_response` — grounded response generator
* `decide` — risk-aware routing

The deterministic mode is deliberate. A reviewer can reproduce the baseline
without an API key, while the interfaces leave room for a future LLM,
embedding model, classifier, or retrieval strategy.

---

## Dataset and intent taxonomy

The included `data/golden_set.jsonl` is a small, transparent demonstration
dataset with `message`, `intent`, `expected_action`, and `evidence` fields.
Replace it with a 150–250 example manually labelled golden set before making
claims about production quality.

Current demo intents:

* `DUPLICATE_CHARGE`
* `ACCOUNT_ACCESS`
* `GENERAL_SUPPORT`

---

## Evaluation methodology

`metlib_core/evaluation.py` compares predictions against the golden set and
saves:

* intent accuracy and F1
* escalation F1
* evidence-grounding status
* reply-quality and LLM-judge agreement placeholders until human labels exist
* inspectable failure records

`metlib_core/judge.py` defines the structured judge contract across relevance,
grounding, correctness, brand consistency, safety, and escalation
appropriateness. `data/human_validation.jsonl` is the transparent label format;
its `null` values are intentional until a human reviewer supplies ratings.

The harness includes a comparison vocabulary for a trivial baseline, a simple
keyword baseline, and MetaLib. Illustrative values on the landing page are
explicitly marked as illustrative and must not be treated as measured results.

---

## Configuration

Copy `.env.example` to `.env` when configuring a real model:

```text
LLM_API_KEY=
MODEL=deterministic-demo
DATASET_PATH=data/golden_set.jsonl
RESULTS_PATH=evaluation/results.json
METALIB_MODE=demo
```

No secret is required for the included demo mode.

---

## Render deployment

The API is separated from the frontend and binds to the `PORT` environment
variable. A Render web service can use the API server's build/start commands:

```text
Build: pnpm install --frozen-lockfile && pnpm --filter @workspace/api-server run build
Start: pnpm --filter @workspace/api-server run start
```

Set `MODEL`, `DATASET_PATH`, and any provider-specific secret in Render's
environment settings. Keep `.env` files and real API keys out of Git.

---

## Project structure

```text
artifacts/metalib/          React + Vite presentation layer
artifacts/api-server/       Express API boundary
lib/api-spec/               OpenAPI source of truth
lib/api-client-react/       Generated React Query client
lib/api-zod/                Generated request/response validators
metlib_core/                Modular Python intelligence core
cli/                        Zero-setup CLI entrypoint
data/                       Transparent golden-set example
evaluation/                 Generated machine-readable results
```

---

## Limitations and decision log

The included classifier and evidence store are deterministic baselines, not
claims about live customer-support quality. The real system should add a
validated Twitter conversation loader, brand-specific taxonomy review, stronger
retrieval, human labels for response quality, and a model-backed judge. Those
changes belong behind the same pipeline interfaces.

## License

See `LICENSE`.
