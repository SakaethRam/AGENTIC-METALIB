"""
MetaLib: AI Customer Support Agent

Core capabilities:
    1. Intent classification
    2. Historically grounded reply drafting
    3. Auto-handle vs human escalation

Additional evaluation support:
    - TF-IDF + Logistic Regression baseline
    - Historical case retrieval
    - LLM-powered reasoning and response generation

Dataset:
    thoughtvector/customer-support-on-twitter

Runtime:
    - Kaggle dataset is downloaded automatically with kagglehub
    - MetaLib connects to the hosted Render API
    - The Groq API key and model remain server-side on Render

IMPORTANT:
    This file is intentionally standalone.

PERFORMANCE:
    - First run performs expensive dataset preparation.
    - Prepared classifier + retrieval index are persisted locally.
    - Second and subsequent runs load the prepared cache.
    - The dataset does NOT need to be reprocessed every launch.

CACHE:
    .metalib_cache/prepared_v2.pkl

If the dataset, intent rules, or preprocessing configuration changes,
increment CACHE_VERSION below to force a rebuild.
"""

from __future__ import annotations

import csv
import json
import os
import pickle
import re
import shutil
import sys
import time

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import kagglehub
import requests

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# ============================================================================
# CLI STARTUP ANIMATION
# ============================================================================


def metalib_startup_animation() -> None:
    """
    Display the MetaLib startup animation before initializing the agent.
    """

    import shutil
    import sys
    import time

    # Clear terminal.
    os.system(
        "cls" if os.name == "nt" else "clear"
    )

    logo = [
        "╔══════════════════════════════════════════════════════════════════════╗",
        "║                                                                      ║",
        "║   ███╗   ███╗███████╗████████╗ █████╗ ██╗     ██╗██████╗             ║",
        "║   ████╗ ████║██╔════╝╚══██╔══╝██╔══██╗██║     ██║██╔══██╗            ║",
        "║   ██╔████╔██║█████╗     ██║   ███████║██║     ██║██████╔╝            ║",
        "║   ██║╚██╔╝██║██╔══╝     ██║   ██╔══██║██║     ██║██╔══██╗            ║",
        "║   ██║ ╚═╝ ██║███████╗   ██║   ██║  ██║███████╗██║██████╔╝            ║",
        "║   ╚═╝     ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝╚═════╝             ║",
        "║                                                                      ║",
        "║                    AI CUSTOMER SUPPORT AGENT                         ║",
        "║                                                                      ║",
        "╚══════════════════════════════════════════════════════════════════════╝",
    ]

    # ----------------------------------------------------------------------
    # Draw the logo line-by-line.
    # ----------------------------------------------------------------------

    terminal_width = shutil.get_terminal_size(
        fallback=(80, 24)
    ).columns

    print()

    for line in logo:

        # Keep the artwork centered on wider terminals.
        if terminal_width > 78:
            padding = (terminal_width - 78) // 2
        else:
            padding = 0

        sys.stdout.write(
            " " * padding
            + line
            + "\n"
        )

        sys.stdout.flush()

        time.sleep(0.045)

    print()

    # ----------------------------------------------------------------------
    # Startup message.
    # ----------------------------------------------------------------------

    startup_messages = [
        "Initializing MetaLib",
        "Loading historical support intelligence",
        "Loading classifier",
        "Loading retrieval index",
        "Connecting to MetaLib API",
    ]

    for message in startup_messages:

        sys.stdout.write(
            f"{message}..."
        )

        sys.stdout.flush()

        time.sleep(0.45)

        print(
            " OK"
        )

    print()

    time.sleep(0.25)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Standalone raw dataset directory.
RAW_DIR = Path("raw")

# Persistent processed-data cache.
#
# The first run creates this directory and stores:
#   - historical cases
#   - fitted intent classifier
#   - fitted historical retrieval vectorizer
#   - historical retrieval matrix
#
# Subsequent runs load these objects directly.
CACHE_DIR = Path(".metalib_cache")

# Increment this whenever the preprocessing/model configuration changes
# and you want MetaLib to rebuild its cache.
CACHE_VERSION = "v4"

CACHE_FILE = (
    CACHE_DIR
    / f"prepared_{CACHE_VERSION}.pkl"
)

# Runtime configuration is supplied by the environment.
# The script never embeds a provider key or a model identifier.
METALIB_API_URL_ENV = "METALIB_API_URL"
DEFAULT_METALIB_API_URL = "https://agentic-metalib.onrender.com"

# Kaggle dataset.
KAGGLE_DATASET = (
    "thoughtvector/customer-support-on-twitter"
)

SUPPORTED_EXTENSIONS = {
    ".csv",
    ".json",
    ".jsonl",
    ".tsv",
}

# Retrieval configuration.
TOP_K_EVIDENCE = 5
MIN_RETRIEVAL_SIMILARITY = 0.12
RETRIEVAL_INTENT_BONUS = 0.03
RETRIEVAL_RESOLUTION_BONUS = 0.02

# Classification configuration.
MIN_CLASSIFIER_CONFIDENCE = 0.55
RULE_OVERRIDE_MIN_CONFIDENCE = 0.90


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class HistoricalCase:
    case_id: str
    brand: str
    customer_message: str
    resolution: str
    intent: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentResult:
    customer_message: str

    # Feature 1
    intent: str
    intent_confidence: float
    classifier_intent: str
    classifier_confidence: float
    intent_source: str

    # Feature 2
    evidence: list[dict]
    draft_reply: str

    # Feature 3
    decision: str
    escalation_reason: str

    # LLM metadata
    llm_used: bool
    llm_model: str

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================================
# KAGGLE DATASET
# ============================================================================


def download_dataset() -> Path:
    """
    Download the Kaggle dataset directly from this standalone script.

    kagglehub handles its own download caching.

    The dataset files are copied into ./raw so the remainder of MetaLib
    works against a predictable local directory.
    """

    print(
        "\nDownloading Customer Support on Twitter dataset..."
    )

    downloaded_path = Path(
        kagglehub.dataset_download(
            KAGGLE_DATASET
        )
    )

    print(
        "Kaggle dataset downloaded to:"
    )

    print(
        f"  {downloaded_path}"
    )

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    copied_any = False

    for source in downloaded_path.rglob("*"):

        if not source.is_file():
            continue

        if (
            source.suffix.lower()
            not in SUPPORTED_EXTENSIONS
        ):
            continue

        relative = source.relative_to(
            downloaded_path
        )

        destination = (
            RAW_DIR / relative
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Avoid rewriting an identical local file.
        if destination.exists():

            try:

                if (
                    destination.stat().st_size
                    == source.stat().st_size
                ):

                    copied_any = True

                    print(
                        f"  already exists: "
                        f"{destination}"
                    )

                    continue

            except OSError:
                pass

        destination.write_bytes(
            source.read_bytes()
        )

        copied_any = True

        print(
            f"  copied: "
            f"{source.name} -> {destination}"
        )

    if not copied_any:

        raise RuntimeError(
            "Kaggle download completed, but no supported "
            "CSV/JSON/JSONL/TSV files were found."
        )

    return RAW_DIR


# ============================================================================
# DATASET DISCOVERY
# ============================================================================


def discover_dataset_files() -> list[Path]:
    """
    Find supported dataset files under ./raw.

    The Kaggle dataset is downloaded only when no supported local
    dataset file exists.
    """

    files = []

    if RAW_DIR.exists():

        files = [
            path
            for path in RAW_DIR.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            )
        ]

    if not files:

        print(
            "\nNo local dataset found."
        )

        download_dataset()

        files = [
            path
            for path in RAW_DIR.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            )
        ]

    if not files:

        raise FileNotFoundError(
            "No dataset files found under ./raw"
        )

    return files


# ============================================================================
# GENERIC DATA LOADING
# ============================================================================


def normalize_key(
    key: str,
) -> str:

    return re.sub(
        r"[^a-z0-9]",
        "",
        str(key).lower(),
    )


def find_field(
    record: dict,
    candidates: list[str],
) -> Optional[str]:
    """
    Locate a field while tolerating naming differences.

    Examples:
        tweet_text
        tweet text
        TweetText
        text
    """

    normalized = {
        normalize_key(key): key
        for key in record.keys()
    }

    for candidate in candidates:

        key = normalized.get(
            normalize_key(candidate)
        )

        if key:
            return key

    return None


def load_records(
    path: Path,
) -> list[dict]:
    """
    Load CSV, TSV, JSON or JSONL.
    """

    suffix = path.suffix.lower()

    if suffix in {
        ".csv",
        ".tsv",
    }:

        delimiter = (
            "\t"
            if suffix == ".tsv"
            else ","
        )

        with path.open(
            "r",
            encoding="utf-8",
            errors="replace",
            newline="",
        ) as file:

            return list(
                csv.DictReader(
                    file,
                    delimiter=delimiter,
                )
            )

    if suffix == ".jsonl":

        records = []

        with path.open(
            "r",
            encoding="utf-8",
            errors="replace",
        ) as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                try:

                    records.append(
                        json.loads(line)
                    )

                except json.JSONDecodeError:

                    continue

        return records

    if suffix == ".json":

        with path.open(
            "r",
            encoding="utf-8",
            errors="replace",
        ) as file:

            data = json.load(file)

        if isinstance(
            data,
            list,
        ):

            return data

        if isinstance(
            data,
            dict,
        ):

            for key in (
                "data",
                "records",
                "tweets",
                "conversations",
            ):

                value = data.get(
                    key
                )

                if isinstance(
                    value,
                    list,
                ):

                    return value

        raise ValueError(
            f"Unsupported JSON structure: {path}"
        )

    raise ValueError(
        f"Unsupported dataset file: {path}"
    )


# ============================================================================
# TEXT / BRAND EXTRACTION
# ============================================================================


def extract_text(
    record: dict,
) -> Optional[str]:

    field = find_field(
        record,
        [
            "text",
            "tweet",
            "tweet_text",
            "message",
            "customer_message",
            "content",
            "body",
        ],
    )

    if not field:
        return None

    value = record.get(
        field
    )

    if value is None:
        return None

    text = str(
        value
    ).strip()

    return (
        text
        if text
        else None
    )


def extract_brand(
    record: dict,
) -> Optional[str]:

    field = find_field(
        record,
        [
            "brand",
            "company",
            "company_name",
            "airline",
            "brand_name",
            "username",
            "user_name",
            "screen_name",
        ],
    )

    if not field:
        return None

    value = record.get(
        field
    )

    if value is None:
        return None

    value = str(
        value
    ).strip()

    return (
        value
        if value
        else None
    )


# ============================================================================
# BOOTSTRAP INTENT TAXONOMY
# ============================================================================


# Specific problem intents are evaluated before generic action/payment terms.
# Regexes are deliberately explicit to avoid broad substring matches such as
# treating every occurrence of "charged" as PAYMENT.
INTENT_PATTERNS = {
    "DUPLICATE_CHARGE": [
        r"\bcharged\s+(?:twice|two(?:\s+times)?|2(?:\s+times)?)\b",
        r"\bdouble[-\s]?charged\b",
        r"\bduplicate\s+(?:charge|charges|payment|payments)\b",
        r"\b(?:two|2)\s+(?:charges|payments)\b",
        r"\bcharged\s+more\s+than\s+once\b",
        r"\bcharged\s+multiple\s+times\b",
        r"\bbilled\s+(?:twice|two(?:\s+times)?|2(?:\s+times)?)\b",
    ],
    "PAYMENT": [
        r"\bpayment\s+(?:failed|declined|rejected|reversed|pending|didn['’]?t\s+go\s+through|not\s+go(?:ing)?\s+through)\b",
        r"\b(?:card|credit\s+card|debit\s+card)\s+(?:declined|rejected|failed)\b",
        r"\b(?:unable|cannot|can['’]?t)\s+(?:make|complete)\s+(?:a\s+)?payment\b",
        r"\b(?:payment|transaction)\s+(?:error|issue|problem)\b",
    ],
    "ACCOUNT_ACCESS": [
        r"\bcan['’]?t\s+(?:log|sign)\s*in\b",
        r"\bcannot\s+(?:log|sign)\s*in\b",
        r"\bunable\s+to\s+(?:log|sign)\s*in\b",
        r"\b(?:login|log\s*in|sign\s*in)\s+(?:problem|issue|error)\b",
        r"\b(?:forgot|reset|change)\s+(?:my\s+)?password\b",
        r"\b(?:locked\s+out|account\s+access)\b",
    ],
    "CANCELLATION": [
        r"\b(?:cancel|cancellation)\b",
        r"\bunsubscribe\b",
        r"\b(?:stop|terminate)\s+(?:my|the)\s+(?:order|subscription|service)\b",
    ],
    "DELIVERY": [
        r"\b(?:where\s+is|when\s+will)\s+(?:my\s+)?(?:order|package|parcel|shipment)\b",
        r"\b(?:order|package|parcel|shipment)\s+(?:hasn['’]?t|has\s+not|never)\s+arrived\b",
        r"\b(?:late|delayed|missing)\s+(?:delivery|shipment|package|parcel|order)\b",
        r"\b(?:delivery|shipment|shipping)\s+(?:is\s+)?(?:late|delayed|missing)\b",
        r"\b(?:delivery|shipment|shipping)\b",
    ],
    "REFUND_REQUEST": [
        r"\brefund\b",
        r"\b(?:money|cash)\s+back\b",
        r"\bgive\s+(?:me\s+)?my\s+money\s+back\b",
        r"\breimburse(?:ment)?\b",
    ],
    "PAYMENT_GENERIC": [
        r"\b(?:payment|pay|billing|invoice|card|credit\s+card|debit\s+card)\b",
    ],
}

# PAYMENT_GENERIC is mapped to PAYMENT at the boundary so the taxonomy remains
# stable for the classifier and historical cases.

def infer_intent(text: str) -> str:
    """Infer a high-precision bootstrap intent from explicit issue patterns."""
    normalized = re.sub(r"\s+", " ", text.lower()).strip()

    for intent, patterns in INTENT_PATTERNS.items():
        if any(re.search(pattern, normalized) for pattern in patterns):
            return "PAYMENT" if intent == "PAYMENT_GENERIC" else intent

    return "GENERAL_SUPPORT"


def resolve_intent(
    message: str,
    classifier_intent: str,
    classifier_confidence: float,
) -> tuple[str, float, str]:
    """Resolve the classifier result using high-precision issue rules.

    The ML confidence is preserved rather than artificially recalibrated.
    A deterministic rule is only used as an override when it identifies a
    specific customer problem that is more precise than a generic action label.
    """
    rule_intent = infer_intent(message)

    high_precision_intents = {
        "DUPLICATE_CHARGE",
        "PAYMENT",
        "ACCOUNT_ACCESS",
        "CANCELLATION",
        "DELIVERY",
    }

    if rule_intent in high_precision_intents:
        return (
            rule_intent,
            classifier_confidence,
            "RULE_OVERRIDE",
        )

    return (
        classifier_intent,
        classifier_confidence,
        "CLASSIFIER",
    )


# ============================================================================
# DATASET PREPARATION
# ============================================================================


def build_cases(records, file_prefix):
    tweet_map = {
        str(record.get("tweet_id")): record
        for record in records
        if isinstance(record, dict) and record.get("tweet_id")
    }

    cases = []

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue

        text = extract_text(record)
        if not text:
            continue

        # TWCS: inbound=True = customer message
        inbound = str(record.get("inbound", "")).lower() == "true"
        if not inbound:
            continue

        brand = extract_brand(record) or "UNKNOWN"

        # response_tweet_id points to the support agent's reply
        response_id = str(record.get("response_tweet_id", "")).strip()
        resolution = ""

        if response_id:
            response_record = tweet_map.get(response_id)
            if response_record:
                resolution = extract_text(response_record)

        intent = infer_intent(text)

        cases.append(
            HistoricalCase(
                case_id=f"{file_prefix}_{index}",
                brand=brand,
                customer_message=text,
                resolution=resolution,
                intent=intent,
            )
        )

    return cases


# ============================================================================
# TF-IDF INTENT CLASSIFIER
# ============================================================================


class IntentClassifier:

    """
    Simple ML classifier.

    This remains useful because the Hiver assignment explicitly requires
    comparison against simple baselines.
    """

    def __init__(self) -> None:

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=50_000,
            sublinear_tf=True,
        )

        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        )

        self.is_fitted = False

    def fit(
        self,
        messages: list[str],
        labels: list[str],
    ) -> None:

        if len(
            set(labels)
        ) < 2:

            raise ValueError(
                "Intent classifier requires at least "
                "two distinct intent classes."
            )

        X = self.vectorizer.fit_transform(
            messages
        )

        self.model.fit(
            X,
            labels,
        )

        self.is_fitted = True

    def predict(
        self,
        message: str,
    ) -> tuple[str, float]:

        if not self.is_fitted:

            raise RuntimeError(
                "Intent classifier has not been fitted."
            )

        X = self.vectorizer.transform(
            [message]
        )

        probabilities = (
            self.model.predict_proba(
                X
            )[0]
        )

        index = probabilities.argmax()

        intent = (
            self.model.classes_[index]
        )

        confidence = float(
            probabilities[index]
        )

        return (
            intent,
            confidence,
        )


# ============================================================================
# HISTORICAL RETRIEVER
# ============================================================================


class HistoricalRetriever:

    """
    Retrieve historically similar customer-support cases.

    Retrieval happens before the LLM call.

    The LLM therefore does not have to invent the resolution. It receives
    actual historical evidence from the dataset.
    """

    def __init__(
        self,
        cases: list[HistoricalCase],
    ) -> None:

        usable_cases = [
            case
            for case in cases
            if case.customer_message.strip()
        ]

        self.cases = usable_cases

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_features=100_000,
            sublinear_tf=True,
        )

        self.matrix = (
            self.vectorizer.fit_transform(
                [
                    case.customer_message
                    for case in self.cases
                ]
            )
        )

    def search(
        self,
        message: str,
        intent: Optional[str] = None,
        top_k: int = TOP_K_EVIDENCE,
    ) -> list[dict]:
        """Retrieve by semantic similarity, using intent only as a soft signal."""
        query = self.vectorizer.transform([message])
        similarities = cosine_similarity(query, self.matrix)[0]

        ranked_candidates = []

        for index, similarity_value in enumerate(similarities):
            similarity = float(similarity_value)

            if similarity < MIN_RETRIEVAL_SIMILARITY:
                continue

            case = self.cases[index]
            intent_bonus = (
                RETRIEVAL_INTENT_BONUS
                if intent and case.intent == intent
                else 0.0
            )
            resolution_bonus = (
                RETRIEVAL_RESOLUTION_BONUS
                if case.resolution.strip()
                else 0.0
            )

            ranking_score = (
                similarity
                + intent_bonus
                + resolution_bonus
            )

            ranked_candidates.append(
                (
                    ranking_score,
                    similarity,
                    case,
                )
            )

        ranked_candidates.sort(
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )

        results = []

        for ranking_score, similarity, case in ranked_candidates[:top_k]:
            results.append(
                {
                    "case_id": case.case_id,
                    "customer_message": case.customer_message,
                    "resolution": case.resolution,
                    "intent": case.intent,
                    "similarity": round(similarity, 4),
                    "ranking_score": round(ranking_score, 4),
                }
            )

        return results


# ============================================================================
# GROQ LLM
# ============================================================================


class MetaLibSupportLLM:

    """
    Render-backed LLM layer.

    The CLI never receives or stores the Groq API key.

    The request flow is:

        MetaLib CLI
            |
            v
        Render API
            |
            v
        Groq
            |
            v
        LLM response
    """

    def __init__(self) -> None:

        self.api_url = os.getenv(
            METALIB_API_URL_ENV,
            DEFAULT_METALIB_API_URL,
        ).rstrip("/")

        if not self.api_url:
            raise RuntimeError(
                "METALIB_API_URL is not configured."
            )

        # The actual Groq model is intentionally not exposed to
        # the client. Render controls the model server-side.
        self.model = "Render API"

    def analyze(
        self,
        message: str,
        candidate_intent: str,
        candidate_confidence: float,
        evidence: list[dict],
    ) -> dict:

        evidence_text = []

        for item in evidence:

            evidence_text.append(
                f"""
CASE ID: {item["case_id"]}
HISTORICAL CUSTOMER:
{item["customer_message"]}

HISTORICAL INTENT:
{item["intent"]}

HISTORICAL RESOLUTION:
{item["resolution"] or "[No recorded resolution]"}

SIMILARITY:
{item["similarity"]}
""".strip()
            )

        evidence_block = (
            "\n\n".join(
                evidence_text
            )
            if evidence_text
            else "[No historical evidence found]"
        )

        system_prompt = """
You are MetaLib, an AI customer-support agent.

Your job is to analyze an incoming customer-support message using
historical support cases supplied by the system.

You must perform THREE tasks:

1. CLASSIFY THE CUSTOMER'S INTENT
2. DRAFT A REPLY GROUNDED IN HISTORICAL RESOLUTIONS
3. DECIDE AUTO-HANDLE VS ESCALATE TO A HUMAN

CRITICAL GROUNDING RULE:

The draft reply must contain only actions, required information,
policies, timelines, refunds, or operational capabilities explicitly
supported by the historical resolutions.

Do not invent or infer operational capabilities.

If the historical evidence does not clearly support a specific
instruction or requested outcome, omit it rather than infer it.

If no relevant historical resolution supports the request, ESCALATE.

INTENT RULES:

The supplied candidate intent comes from a statistical classifier.
Treat it as a candidate, not as ground truth.

Identify the underlying customer problem rather than merely the
requested action.

Examples:

- "I was charged twice and want a refund" -> DUPLICATE_CHARGE
- "I cancelled my order and want my money back" -> REFUND_REQUEST
- "My payment was declined" -> PAYMENT
- "Where is my order? It is two days late" -> DELIVERY

ESCALATION:

AUTO_HANDLE only when:
- the intent is sufficiently clear,
- historical evidence is relevant,
- the evidence contains a usable resolution,
- and the requested action is sufficiently supported by history.

ESCALATE when:
- intent is ambiguous,
- evidence is weak,
- there is no usable historical resolution,
- the request requires an unsupported action,
- or the customer appears to require human judgment.

RETURN VALID JSON ONLY.

Schema:

{
  "intent": "STRING",
  "intent_confidence": 0.0,
  "draft_reply": "STRING",
  "decision": "AUTO_HANDLE",
  "escalation_reason": "STRING",
  "evidence_used": ["case_id"]
}

decision must be exactly one of:
- AUTO_HANDLE
- ESCALATE

intent_confidence must be between 0 and 1.
""".strip()

        user_prompt = f"""
CUSTOMER MESSAGE:

{message}

STATISTICAL CLASSIFIER RESULT:

Intent: {candidate_intent}
Confidence: {candidate_confidence:.4f}

HISTORICAL SUPPORT EVIDENCE:

{evidence_block}

Now independently assess the message using the evidence.

Return JSON only.
""".strip()

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ]
        }

        try:

            response = requests.post(
                f"{self.api_url}/chat",
                json=payload,
                timeout=90,
            )

        except requests.RequestException as exc:

            raise RuntimeError(
                f"Unable to reach MetaLib API: {exc}"
            ) from exc

        if response.status_code == 429:

            raise RuntimeError(
                "MetaLib API rate limit reached. "
                "Please try again later."
            )

        if not response.ok:

            try:
                detail = response.json().get(
                    "detail",
                    "Unknown API error.",
                )
            except ValueError:
                detail = response.text or "Unknown API error."

            raise RuntimeError(
                f"MetaLib API returned HTTP "
                f"{response.status_code}: {detail}"
            )

        try:

            data = response.json()

        except ValueError as exc:

            raise RuntimeError(
                "MetaLib API returned invalid JSON."
            ) from exc

        result = data.get(
            "response"
        )

        if not result:

            raise RuntimeError(
                "MetaLib API returned an empty LLM response."
            )

        return self._parse_json(
            result
        )

    @staticmethod
    def _parse_json(
        raw: str,
    ) -> dict:

        cleaned = raw.strip()

        if cleaned.startswith(
            "```"
        ):

            cleaned = re.sub(
                r"^```(?:json)?\s*",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )

            cleaned = re.sub(
                r"\s*```$",
                "",
                cleaned,
            )

        try:

            parsed = json.loads(
                cleaned
            )

        except json.JSONDecodeError:

            match = re.search(
                r"\{.*\}",
                cleaned,
                flags=re.DOTALL,
            )

            if not match:

                raise RuntimeError(
                    "LLM did not return valid JSON."
                )

            try:

                parsed = json.loads(
                    match.group(0)
                )

            except json.JSONDecodeError as exc:

                raise RuntimeError(
                    "Unable to parse LLM JSON response."
                ) from exc

        if not isinstance(
            parsed,
            dict,
        ):

            raise RuntimeError(
                "LLM JSON response was not an object."
            )

        return parsed

# ============================================================================
# LLM OUTPUT VALIDATION
# ============================================================================


def validate_llm_result(
    result: dict,
    evidence: list[dict],
    fallback_intent: str,
    fallback_confidence: float,
) -> dict:

    intent = str(
        result.get(
            "intent",
            fallback_intent,
        )
    ).strip()

    if not intent:

        intent = fallback_intent

    try:

        confidence = float(
            result.get(
                "intent_confidence",
                fallback_confidence,
            )
        )

    except (
        TypeError,
        ValueError,
    ):

        confidence = (
            fallback_confidence
        )

    confidence = max(
        0.0,
        min(
            1.0,
            confidence,
        ),
    )

    reply = str(
        result.get(
            "draft_reply",
            "",
        )
    ).strip()

    decision = str(
        result.get(
            "decision",
            "ESCALATE",
        )
    ).strip().upper()

    if decision not in {
        "AUTO_HANDLE",
        "ESCALATE",
    }:

        decision = "ESCALATE"

    reason = str(
        result.get(
            "escalation_reason",
            "",
        )
    ).strip()

    evidence_ids = result.get(
        "evidence_used",
        [],
    )

    if not isinstance(
        evidence_ids,
        list,
    ):

        evidence_ids = []

    valid_ids = {
        item["case_id"]
        for item in evidence
    }

    evidence_ids = [
        str(case_id)
        for case_id in evidence_ids
        if str(case_id)
        in valid_ids
    ]

    # ----------------------------------------------------------------------
    # Safety gate:
    #
    # The LLM cannot AUTO_HANDLE if there is no usable historical
    # resolution evidence.
    # ----------------------------------------------------------------------

    usable_evidence = [
        item
        for item in evidence
        if item.get("resolution")
    ]

    if not usable_evidence:

        decision = "ESCALATE"

        reason = (
            "No usable historical resolution evidence "
            "was available for this request."
        )

    if not reply:

        reply = (
            "Thanks for reaching out. We need to review "
            "your request before providing a resolution."
        )

        decision = "ESCALATE"

        reason = (
            reason
            or "The system could not produce a grounded reply."
        )

    if (
        decision == "ESCALATE"
        and not reason
    ):

        reason = (
            "The available evidence does not provide "
            "sufficient confidence for automated handling."
        )

    return {
        "intent": intent,
        "intent_confidence": confidence,
        "draft_reply": reply,
        "decision": decision,
        "escalation_reason": reason,
        "evidence_used": evidence_ids,
    }


# ============================================================================
# PERSISTENT PREPROCESSING CACHE
# ============================================================================


def save_prepared_cache(
    cases: list[HistoricalCase],
    classifier: IntentClassifier,
    retriever: HistoricalRetriever,
) -> None:
    """
    Persist all expensive preprocessing artifacts.

    Saved artifacts include:
        - historical cases
        - fitted intent classifier
        - fitted intent TF-IDF vectorizer
        - fitted historical retrieval vectorizer
        - historical retrieval sparse matrix

    This means subsequent launches do not need to rebuild the ML pipeline.
    """

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "cache_version": CACHE_VERSION,
        "cases": cases,
        "classifier": classifier,
        "retriever": retriever,
    }

    # Write to a temporary file first.
    #
    # If the process is interrupted during pickle.dump(), the previous
    # valid cache remains untouched.
    temporary_file = CACHE_FILE.with_suffix(
        ".tmp"
    )

    print(
        "\nSaving prepared MetaLib cache..."
    )

    with temporary_file.open(
        "wb"
    ) as file:

        pickle.dump(
            payload,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # Atomic replacement.
    temporary_file.replace(
        CACHE_FILE
    )

    print(
        "Prepared MetaLib cache saved to:"
    )

    print(
        f"  {CACHE_FILE}"
    )


def load_prepared_cache() -> Optional[dict]:
    """
    Load previously prepared preprocessing artifacts.

    Returns None if:
        - no cache exists
        - cache version is outdated
        - cache is incomplete
        - cache cannot be deserialized
    """

    if not CACHE_FILE.exists():

        return None

    try:

        print(
            "\nLoading prepared MetaLib cache..."
        )

        with CACHE_FILE.open(
            "rb"
        ) as file:

            payload = pickle.load(
                file
            )

        if not isinstance(
            payload,
            dict,
        ):

            print(
                "Cache format is invalid. "
                "Rebuilding preprocessing cache."
            )

            return None

        if (
            payload.get(
                "cache_version"
            )
            != CACHE_VERSION
        ):

            print(
                "Cache version mismatch. "
                "Rebuilding preprocessing cache."
            )

            return None

        cases = payload.get(
            "cases"
        )

        classifier = payload.get(
            "classifier"
        )

        retriever = payload.get(
            "retriever"
        )

        if (
            not cases
            or classifier is None
            or retriever is None
        ):

            print(
                "Cache is incomplete. "
                "Rebuilding preprocessing cache."
            )

            return None

        # Basic validation of the restored objects.
        if not isinstance(
            classifier,
            IntentClassifier,
        ):

            print(
                "Cached classifier is invalid. "
                "Rebuilding preprocessing cache."
            )

            return None

        if not isinstance(
            retriever,
            HistoricalRetriever,
        ):

            print(
                "Cached retriever is invalid. "
                "Rebuilding preprocessing cache."
            )

            return None

        print(
            f"Loaded {len(cases):,} historical cases from cache."
        )

        return {
            "cases": cases,
            "classifier": classifier,
            "retriever": retriever,
        }

    except Exception as exc:

        print(
            f"Unable to load cache: {exc}"
        )

        print(
            "Rebuilding preprocessing cache..."
        )

        return None


# ============================================================================
# META LIB AGENT
# ============================================================================


class MetaLibAgent:

    """
    End-to-end MetaLib support agent.

    Pipeline:

        Customer message
              |
              v
        TF-IDF classifier
              |
              v
        Historical retrieval
              |
              v
        Groq LLM
          /    |    \
         /     |     \
    intent   reply   escalation
    """

    def __init__(
        self,
        cases: list[HistoricalCase],
        classifier: Optional[
            IntentClassifier
        ] = None,
        retriever: Optional[
            HistoricalRetriever
        ] = None,
    ) -> None:

        if len(cases) < 2:

            raise ValueError(
                "MetaLib requires at least two historical cases."
            )

        self.cases = cases

        # ------------------------------------------------------------------
        # Reuse cached classifier when available.
        # ------------------------------------------------------------------

        if classifier is not None:

            print(
                "\nLoading cached statistical intent classifier..."
            )

            self.classifier = classifier

        else:

            print(
                "\nTraining statistical intent classifier..."
            )

            self.classifier = (
                IntentClassifier()
            )

            self.classifier.fit(
                [
                    case.customer_message
                    for case in cases
                ],
                [
                    case.intent
                    for case in cases
                ],
            )

        # ------------------------------------------------------------------
        # Reuse cached historical retrieval index when available.
        # ------------------------------------------------------------------

        if retriever is not None:

            print(
                "Loading cached historical evidence index..."
            )

            self.retriever = retriever

        else:

            print(
                "Building historical evidence index..."
            )

            self.retriever = (
                HistoricalRetriever(
                    cases
                )
            )

        # ------------------------------------------------------------------
        # Groq connection is lightweight.
        #
        # It is intentionally created on every launch because it is not
        # the expensive part of the pipeline.
        # ------------------------------------------------------------------

        print(
            "Connecting to MetaLib API..."
        )

        self.llm = MetaLibSupportLLM()

    def analyze(
        self,
        message: str,
    ) -> AgentResult:

        message = message.strip()

        if not message:

            raise ValueError(
                "Customer message cannot be empty."
            )

        # ------------------------------------------------------------------
        # STEP 1: STATISTICAL INTENT CLASSIFICATION
        # ------------------------------------------------------------------

        (
            classifier_intent,
            classifier_confidence,
        ) = self.classifier.predict(
            message
        )

        # ------------------------------------------------------------------
        # STEP 2: PRECISION INTENT RESOLUTION
        # ------------------------------------------------------------------

        (
            resolved_intent,
            resolved_confidence,
            intent_source,
        ) = resolve_intent(
            message=message,
            classifier_intent=classifier_intent,
            classifier_confidence=classifier_confidence,
        )

        # ------------------------------------------------------------------
        # STEP 3: HISTORICAL RETRIEVAL
        # ------------------------------------------------------------------

        evidence = self.retriever.search(
            message=message,
            intent=resolved_intent,
            top_k=TOP_K_EVIDENCE,
        )

        usable_evidence = [
            item
            for item in evidence
            if item.get("resolution", "").strip()
        ]

        if not usable_evidence:
            return AgentResult(
                customer_message=message,
                intent=resolved_intent,
                intent_confidence=round(resolved_confidence, 4),
                classifier_intent=classifier_intent,
                classifier_confidence=round(classifier_confidence, 4),
                intent_source=intent_source,
                evidence=evidence,
                draft_reply=(
                    "Thanks for reaching out. "
                    "We need to review your request before providing a resolution."
                ),
                decision="ESCALATE",
                escalation_reason=(
                    "No usable historical resolution evidence "
                    "was available for this request."
                ),
                llm_used=False,
                llm_model=self.llm.model,
            )

        # ------------------------------------------------------------------
        # STEP 3: LLM REASONING
        # ------------------------------------------------------------------

        try:

            llm_result = self.llm.analyze(
                message=message,
                candidate_intent=resolved_intent,
                candidate_confidence=resolved_confidence,
                evidence=evidence,
            )

            validated = (
                validate_llm_result(
                    result=llm_result,
                    evidence=evidence,
                    fallback_intent=resolved_intent,
                    fallback_confidence=resolved_confidence,
                )
            )

            return AgentResult(
                customer_message=message,

                intent=validated["intent"],

                intent_confidence=round(
                    validated["intent_confidence"],
                    4,
                ),

                classifier_intent=classifier_intent,
                classifier_confidence=round(
                    classifier_confidence,
                    4,
                ),
                intent_source=intent_source,

                evidence=evidence,

                draft_reply=validated[
                    "draft_reply"
                ],

                decision=validated[
                    "decision"
                ],

                escalation_reason=validated[
                    "escalation_reason"
                ],

                llm_used=True,

                llm_model=self.llm.model,
            )

        except Exception:

            # Fail closed.
            #
            # If the LLM is unavailable, MetaLib must not fabricate
            # a confident automated answer.

            usable_evidence = [
                item
                for item in evidence
                if item.get(
                    "resolution"
                )
            ]

            if usable_evidence:

                fallback_reply = (
                    "Thanks for reaching out. We found "
                    "a potentially relevant historical case, "
                    "but this request requires review before "
                    "we provide a final resolution."
                )

            else:

                fallback_reply = (
                    "Thanks for reaching out. We need to "
                    "review your request before providing "
                    "a resolution."
                )

            return AgentResult(
                customer_message=message,

                intent=resolved_intent,

                intent_confidence=round(
                    resolved_confidence,
                    4,
                ),

                classifier_intent=classifier_intent,
                classifier_confidence=round(
                    classifier_confidence,
                    4,
                ),
                intent_source=intent_source,

                evidence=evidence,

                draft_reply=fallback_reply,

                decision="ESCALATE",

                escalation_reason=(
                    "LLM processing was unavailable. "
                    "The request was safely escalated."
                ),

                llm_used=False,

                llm_model=self.llm.model,
            )


# ============================================================================
# DATASET LOADING
# ============================================================================


def load_dataset() -> list[HistoricalCase]:

    files = discover_dataset_files()

    print(
        "\nDataset files discovered:"
    )

    for file in files:

        print(
            f"  {file}"
        )

    all_cases = []

    for file in files:

        print(
            f"\nReading {file.name}..."
        )

        try:

            records = load_records(
                file
            )

        except Exception as exc:

            print(
                f"  Skipping file: {exc}"
            )

            continue

        cases = build_cases(
            records,
            file_prefix=file.stem,
        )

        print(
            f"  {len(records):,} records"
            f" -> {len(cases):,} usable cases"
        )

        all_cases.extend(
            cases
        )

    if not all_cases:

        raise RuntimeError(
            "No usable customer-support records were found."
        )

    return all_cases


# ============================================================================
# DATASET SUMMARY
# ============================================================================


def print_dataset_summary(
    cases: list[HistoricalCase],
) -> None:

    print(
        "\n" + "=" * 78
    )

    print(
        "DATASET SUMMARY"
    )

    print(
        "=" * 78
    )

    print(
        f"Total usable cases: "
        f"{len(cases):,}"
    )

    brands = {}

    for case in cases:

        brands[case.brand] = (
            brands.get(
                case.brand,
                0,
            )
            + 1
        )

    print(
        f"Distinct brands/accounts: "
        f"{len(brands):,}"
    )

    print(
        "\nTop brands/accounts:"
    )

    for brand, count in sorted(
        brands.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:10]:

        print(
            f"  {brand}: {count:,}"
        )

    intents = {}

    for case in cases:

        intents[case.intent] = (
            intents.get(
                case.intent,
                0,
            )
            + 1
        )

    print(
        "\nBootstrap intent distribution:"
    )

    for intent, count in sorted(
        intents.items(),
        key=lambda item: item[1],
        reverse=True,
    ):

        print(
            f"  {intent}: {count:,}"
        )


# ============================================================================
# CLI OUTPUT
# ============================================================================


def print_result(
    result: AgentResult,
) -> None:

    print(
        "\n" + "=" * 78
    )

    print(
        "METALIB SUPPORT AGENT"
    )

    print(
        "=" * 78
    )

    print(
        "\nCUSTOMER MESSAGE"
    )

    print(
        result.customer_message
    )

    print(
        "\n1. INTENT CLASSIFICATION"
    )

    print(
        f"Intent:       "
        f"{result.intent}"
    )

    print(
        f"Confidence:   "
        f"{result.intent_confidence:.2%}"
    )

    print(
        f"LLM used:     "
        f"{'YES' if result.llm_used else 'NO'}"
    )

    print(
        f"Classifier:   "
        f"{result.classifier_intent} "
        f"({result.classifier_confidence:.2%})"
    )

    print(
        f"Intent source: {result.intent_source}"
    )

    print(
        "\n2. HISTORICAL EVIDENCE"
    )

    if not result.evidence:

        print(
            "No historical evidence found."
        )

    for index, item in enumerate(
        result.evidence,
        start=1,
    ):

        print(
            f"\nEvidence #{index}"
        )

        print(
            f"Case ID:     "
            f"{item['case_id']}"
        )

        print(
            f"Similarity:  "
            f"{item['similarity']:.4f}"
        )

        print(
            f"Intent:      "
            f"{item['intent']}"
        )

        print(
            f"Customer:    "
            f"{item['customer_message']}"
        )

        print(
            f"Resolution:  "
            f"{item['resolution'] or '[none]'}"
        )

    print(
        "\n3. GROUNDED DRAFT REPLY"
    )

    print(
        result.draft_reply
    )

    print(
        "\n4. ESCALATION DECISION"
    )

    print(
        f"Decision:     "
        f"{result.decision}"
    )

    print(
        f"Reason:       "
        f"{result.escalation_reason}"
    )

    print(
        "\nLLM MODEL"
    )

    print(
        result.llm_model
    )

    print(
        "\n" + "=" * 78
    )


# ============================================================================
# INTERACTIVE CLI
# ============================================================================


def run_cli(
    agent: MetaLibAgent,
) -> None:

    print(
        "\n" + "=" * 78
    )

    print(
        "METALIB"
    )

    print(
        "AI Customer Support Agent"
    )

    print(
        "=" * 78
    )

    print(
        "\nMetaLib is ready."
    )

    print(
        "Enter a customer-support message."
    )

    print(
        "Type 'exit' to stop."
    )

    while True:

        try:

            message = input(
                "\nCustomer > "
            ).strip()

        except KeyboardInterrupt:

            print()

            break

        if message.lower() in {
            "exit",
            "quit",
        }:

            break

        if not message:

            continue

        try:

            result = agent.analyze(
                message
            )

            print_result(
                result
            )

        except Exception as exc:

            print(
                f"\nError: {exc}"
            )


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:

    metalib_startup_animation()

    # ======================================================================
    # FAST PATH: LOAD EXISTING PREPROCESSING CACHE
    # ======================================================================

    cached = load_prepared_cache()

    if cached is not None:

        print(
            "\nUsing previously prepared MetaLib artifacts."
        )

        cases = cached[
            "cases"
        ]

        print_dataset_summary(
            cases
        )

        print(
            "\nBuilding MetaLib from cached artifacts..."
        )

        agent = MetaLibAgent(
            cases=cases,
            classifier=cached[
                "classifier"
            ],
            retriever=cached[
                "retriever"
            ],
        )

    else:

        # ==================================================================
        # FIRST-RUN PATH
        # ==================================================================

        print(
            "\nNo prepared cache found."
        )

        print(
            "Running first-time dataset preparation..."
        )

        cases = load_dataset()

        print_dataset_summary(
            cases
        )

        print(
            "\nBuilding MetaLib..."
        )

        agent = MetaLibAgent(
            cases=cases
        )

        save_prepared_cache(
            cases=cases,
            classifier=agent.classifier,
            retriever=agent.retriever,
        )

    print(
        "\nMetaLib agent ready."
    )

    run_cli(
        agent
    )

# ============================================================================
# ENTRY POINT
# ============================================================================


if __name__ == "__main__":

    main()
