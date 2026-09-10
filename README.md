# METALIB AI: AGENTIC DATA FLOW

AI-powered customer support analysis and response drafting, grounded in historical support cases.

MetaLib is a standalone Python CLI that combines intent classification, historical case retrieval, and LLM reasoning to determine whether a customer-support request can be handled automatically or should be escalated to a human.

The system is designed around one core principle:

**Historical evidence is the source of truth for operational support actions.**

>VISIT METALIB : [@MetaLib]()

<img width="1832" height="862" alt="LIB-#1" src="https://github.com/user-attachments/assets/69a65282-934a-42d0-8a17-0344855e5a39" />

<img width="1852" height="848" alt="LIB-#2" src="https://github.com/user-attachments/assets/ae52c984-e16d-48e5-bd18-34ce47835c74" />

---

## END-USER SET-UP GUIDE

### 1. Clone the repository

```bash
git clone https://github.com/SakaethRam/MetaLib-AI.git && cd AGENTIC-METALIB
```

### 2. Install MetaLib

```bash
pip install -r Requirements.txt && pip install -e .
```

### 3. Launch MetaLib

```bash
metalib
```

MetaLib automatically handles dataset preparation and connects to the hosted MetaLib API for LLM-powered analysis.

>**No Groq API key, `.env` configuration, backend setup, or Render deployment is required for end users. If Developer, See the Developer Installation Guide below for detailed local-system setup instructions.**

---

## Overview

MetaLib processes an incoming customer-support message through three complementary layers:

1. **Intent Classification**
   - Predicts the customer's support intent using a TF-IDF + Logistic Regression classifier.
   - Applies deterministic intent rules where appropriate.

2. **Historical Retrieval**
   - Retrieves semantically similar historical customer-support cases.
   - Uses the retrieved resolutions as grounding evidence for the response.

3. **LLM Reasoning**
   - Analyzes the customer request against the retrieved historical evidence.
   - Drafts a support response.
   - Determines whether the request can be safely auto-handled or requires human escalation.

The final system fails closed when sufficient historical evidence is unavailable or when the proposed response is not adequately grounded.

---

## Architecture

```text
                    ┌──────────────────────┐
                    │   MetaLib CLI        │
                    │   Python Application  │
                    └──────────┬───────────┘
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
             ▼                 ▼                 ▼
      Intent Classifier   Historical Search   Request Analysis
             │                 │                 │
             └─────────────────┼─────────────────┘
                               ▼
                    ┌──────────────────────┐
                    │   MetaLib API        │
                    │   Hosted on Render   │
                    └──────────┬───────────┘
                               │
                               ▼
                         Groq LLM API
                               │
                               ▼
                    Grounded Final Decision
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
           AUTO_HANDLE                   ESCALATE
````

The Groq API key and model configuration remain server-side on the hosted API. They are never required by or distributed with the CLI.

---

## Key Capabilities

### Intent Classification

MetaLib classifies incoming support requests into customer-support intents using a locally prepared machine-learning classifier.

The classifier provides:

* Predicted intent
* Classification confidence
* Deterministic rule overrides for high-confidence patterns

### Historical Case Retrieval

MetaLib retrieves relevant historical support cases from the customer-support dataset.

Each retrieved case provides:

* Historical customer message
* Historical resolution
* Similarity score
* Case identifier

This evidence is passed to the reasoning layer instead of relying solely on the LLM's general knowledge.

### Grounded Response Drafting

The LLM is instructed to generate responses using only information supported by the retrieved historical resolutions.

MetaLib is designed to avoid inventing:

* Refund policies
* Operational capabilities
* Processing timelines
* Account actions
* Support procedures
* Other unsupported commitments

### Auto-Handle vs Escalation

Each request receives a final operational decision:

```text
AUTO_HANDLE
```

or

```text
ESCALATE
```

Automatic handling requires sufficient historical evidence and a response that remains within the capabilities supported by that evidence.

Requests involving ambiguity, insufficient evidence, unsupported actions, or human judgment are escalated.

---

## Dataset

MetaLib uses the public Twitter customer-support dataset:

```text
thoughtvector/customer-support-on-twitter
```

The dataset is downloaded automatically through `kagglehub` during initial preparation.

Historical cases are used for both:

* Intent classification
* Retrieval-based grounding

The dataset is not committed to the repository.

---

## Technology Stack

* Python 3.10+
* scikit-learn
* FastAPI
* Groq
* Requests
* KaggleHub
* python-dotenv
* SlowAPI

### Machine Learning

* TF-IDF Vectorization
* Logistic Regression
* Cosine similarity retrieval

### API

* FastAPI
* Uvicorn
* SlowAPI rate limiting
* Server-side Groq credentials

---

# Installation

## Requirements

* Python 3.10 or newer
* Git
* Internet access for initial dataset preparation
* Access to the hosted MetaLib API

Users do **not** need a Groq API key.

---

## Clone the Repository

```bash
git clone https://github.com/SakaethRam/MetaLib-AI.git
cd MetaLib-AI
```

---

## Create a Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r Requirements.txt
```

Install MetaLib as a local CLI package:

```bash
pip install -e .
```

---

# Configuration

MetaLib communicates with the hosted API through:

```text
METALIB_API_URL
```

Create a `.env` file in the project root if you need to override the default API endpoint:

```env
METALIB_API_URL=https://YOUR-RENDER-URL.onrender.com
```

If the CLI has been configured with the production API endpoint as its default, no `.env` file is required for normal usage.

### Server-side configuration

The hosted API uses:

```text
GROQ_API_KEY
GROQ_MODEL
```

These variables are configured exclusively on the server.

They must never be placed in:

* `MetaLib.py`
* `.env` committed to Git
* `.env.example`
* the Docker image
* the public repository

---

# Usage

Once installation is complete:

```bash
metalib
```

MetaLib starts the CLI and initializes the local dataset and retrieval/classification cache when required.

---

## First Run

On the first execution, MetaLib:

```text
Download historical dataset
        ↓
Prepare training and retrieval data
        ↓
Train classifier
        ↓
Build retrieval index
        ↓
Persist prepared artifacts locally
        ↓
Connect to MetaLib API
        ↓
Process support requests
```

Prepared data is cached in:

```text
.metalib_cache/
```

The current cache artifact is:

```text
.metalib_cache/prepared_v4.pkl
```

Subsequent executions reuse the cache instead of repeating the full preparation process.

---

# Processing Pipeline

For every customer-support request, MetaLib follows this general flow:

```text
Customer Message
       │
       ▼
Intent Classification
       │
       ▼
Intent Validation / Rule Override
       │
       ▼
Historical Case Retrieval
       │
       ▼
Evidence Evaluation
       │
       ├── No usable evidence
       │          │
       │          ▼
       │      ESCALATE
       │
       ▼
LLM Reasoning
       │
       ▼
Response + Decision Validation
       │
       ├── Grounded and supported
       │          │
       │          ▼
       │      AUTO_HANDLE
       │
       └── Unsupported / ambiguous
                  │
                  ▼
              ESCALATE
```

---

# Grounding Strategy

MetaLib does not treat an LLM-generated response as automatically trustworthy.

The reasoning layer receives historical evidence and is explicitly instructed to:

* use historical resolutions as operational evidence
* avoid unsupported claims
* avoid inventing capabilities
* avoid promising unsupported outcomes
* escalate when evidence is insufficient

The final validation layer provides an additional safety boundary.

For example, if historical cases only demonstrate that a duplicate charge can be investigated, MetaLib should not independently promise:

```text
"We will refund the duplicate charge immediately."
```

unless the retrieved evidence explicitly supports that operational action.

Instead, the request should be escalated when the requested action exceeds what the historical evidence establishes.

---

# Example

### Customer Request

```text
I was charged twice for the same order.
Please refund the extra charge immediately and let me know
exactly when the money will be back in my account.
```

### Classification

```text
Intent: DUPLICATE_CHARGE
```

### Historical Evidence

```text
Historical cases support:
- Investigating the duplicate charge
- Requesting account information
- Contacting the customer

Historical cases do not establish:
- Immediate refund processing
- A guaranteed refund
- A specific refund timeline
```

### Final Decision

```text
Decision: ESCALATE
```

### Grounded Response

```text
We’re sorry you were charged twice. Please DM us your account
email so we can look into this.
```

The response deliberately avoids promising an unsupported refund or timeline.

---

# API

MetaLib uses a lightweight FastAPI service for LLM inference.

### Health Check

```http
GET /
```

Returns:

```json
{
  "name": "MetaLib API",
  "status": "online"
}
```

### Health Endpoint

```http
GET /health
```

Returns:

```json
{
  "status": "healthy"
}
```

### Chat Endpoint

```http
POST /chat
```

Request:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "System instructions"
    },
    {
      "role": "user",
      "content": "Customer support request"
    }
  ]
}
```

Response:

```json
{
  "response": "Generated response"
}
```

---

# Security

MetaLib is designed so that the public CLI does not contain the LLM provider credentials.

The architecture is:

```text
End User
   │
   │ HTTPS
   ▼
MetaLib CLI
   │
   │ HTTPS
   ▼
Hosted MetaLib API
   │
   │ Server-side credentials
   ▼
Groq API
```

The following are intentionally excluded from version control:

```text
.env
raw/
.metalib_cache/
__pycache__/
```

Never commit production credentials.

---

# Rate Limiting

The hosted API applies request rate limiting to the `/chat` endpoint.

The current limit is:

```text
20 requests / minute / client
```

This helps prevent accidental or abusive request spikes against the hosted LLM service.

---

# Repository Structure

```text
MetaLib-AI/
│
├── MetaLib.py
├── API.py
├── Requirements.txt
├── pyproject.toml
├── Dockerfile
├── .dockerignore
├── .gitignore
├── .env.example
├── LICENSE
└── README.md
```

Runtime-generated directories are intentionally excluded from the repository:

```text
raw/
.metalib_cache/
```

---

# CLI Packaging

MetaLib is exposed as the following command:

```bash
metalib
```

The command is configured through `pyproject.toml`:

```toml
[project.scripts]
metalib = "MetaLib:main"
```

After:

```bash
pip install -e .
```

the command becomes available in the active Python environment.

---

# Hosted Deployment

The API is designed to run as a containerized FastAPI service.

The included `Dockerfile` starts:

```bash
uvicorn API:app --host 0.0.0.0 --port ${PORT:-10000}
```

The hosted deployment requires the following server-side environment variables:

```text
GROQ_API_KEY
GROQ_MODEL
```

The client-facing API URL is configured separately through:

```text
METALIB_API_URL
```

---

# Design Principles

## Evidence Before Automation

A request should not be automatically handled simply because an LLM can produce a plausible answer.

Historical evidence must support the operational response.

## Fail Closed

When MetaLib cannot establish sufficient evidence for safe handling, it escalates rather than inventing an answer.

## Separation of Concerns

The system separates:

```text
Classification
Retrieval
Reasoning
Validation
```

This makes the individual components independently testable and reduces dependence on a single model decision.

## Server-Side Credentials

LLM provider credentials remain outside the distributed CLI.

Users interact with the hosted API rather than receiving the underlying provider credentials.

---

# License

MetaLib AI is distributed under the terms defined in [`LICENSE`](LICENSE).
