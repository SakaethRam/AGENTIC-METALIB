import os

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY_ENV = "GROQ_API_KEY"
GROQ_MODEL_ENV = "GROQ_MODEL"

groq_api_key = os.getenv(GROQ_API_KEY_ENV)
groq_model = os.getenv(GROQ_MODEL_ENV)

if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY is not configured")

if not groq_model:
    raise RuntimeError("GROQ_MODEL is not configured")

client = Groq(api_key=groq_api_key)

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="MetaLib API",
    version="1.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler,
)


class ChatRequest(BaseModel):
    messages: list[dict[str, str]] = Field(
        ...,
        min_length=1,
        max_length=20,
    )


class ChatResponse(BaseModel):
    response: str


@app.get("/")
def root():
    return {
        "name": "MetaLib API",
        "status": "online",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("20/minute")
async def chat(request: Request, body: ChatRequest):
    try:
        # Basic request-size protection.
        total_chars = sum(
            len(message.get("content", ""))
            for message in body.messages
        )

        if total_chars > 30_000:
            raise HTTPException(
                status_code=413,
                detail="Request is too large.",
            )

        # Only allow expected message roles.
        allowed_roles = {"system", "user", "assistant"}

        for message in body.messages:
            if message.get("role") not in allowed_roles:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid message role.",
                )

            if not isinstance(message.get("content"), str):
                raise HTTPException(
                    status_code=400,
                    detail="Message content must be text.",
                )

        completion = client.chat.completions.create(
            model=groq_model,
            messages=body.messages,
            temperature=0.2,
        )

        response = completion.choices[0].message.content

        if not response:
            raise HTTPException(
                status_code=502,
                detail="LLM returned an empty response.",
            )

        return ChatResponse(response=response)

    except HTTPException:
        raise

    except Exception as exc:
        print(f"Groq request failed: {type(exc).__name__}: {exc}")

        raise HTTPException(
            status_code=502,
            detail="Unable to generate a response.",
        )