import asyncio
import logging
import os
import time
from dotenv import dotenv_values
import anthropic
from typing import List

logger = logging.getLogger(__name__)

# User-facing fallback messages returned on API errors.
# Callers can check LLMService.is_failure(text) to detect failure placeholders.
FAILURE_PREFIX = 'I apologize, I could not generate a response at this time. '
_QUOTA_MESSAGE = FAILURE_PREFIX + "I've hit the rate limit for this — please try again in a moment."
_GENERIC_FAILURE_MESSAGE = FAILURE_PREFIX + 'Please try again in a moment.'

# Optional live-mounted .env file path in container environments.
_ENV_FILE = '/project.env'


def _read_anthropic_api_key() -> str:
    if os.path.exists(_ENV_FILE):
        key = dotenv_values(_ENV_FILE).get('ANTHROPIC_API_KEY')
        if key:
            return key
    from core.config import settings
    return settings.ANTHROPIC_API_KEY


# Retry parameters for transient network and rate-limit errors
_MAX_ATTEMPTS = 2
_RETRY_SLEEP_CAP = 5.0


def _is_transient(err: Exception) -> bool:
    if isinstance(err, (anthropic.RateLimitError, anthropic.APIConnectionError,
                         anthropic.APITimeoutError, anthropic.InternalServerError)):
        return True
    if isinstance(err, anthropic.APIStatusError):
        return err.status_code >= 500
    return False


class LLMService:
    _MODEL = 'claude-sonnet-5'
    _MAX_TOKENS = 2048

    _clients: dict = {}       # api_key -> anthropic.AsyncAnthropic
    _client_key = None

    @classmethod
    def get_client(cls) -> anthropic.AsyncAnthropic:
        api_key = _read_anthropic_api_key()
        if not api_key:
            raise ValueError('ANTHROPIC_API_KEY is not set in the .env file')

        if cls._client_key != api_key:   # key changed → drop cached client
            cls._clients = {}
            cls._client_key = api_key

        if api_key not in cls._clients:
            cls._clients[api_key] = anthropic.AsyncAnthropic(
                api_key=api_key,
                timeout=25,
                max_retries=0,
            )
        return cls._clients[api_key]

    @classmethod
    async def chat(
        cls,
        system_prompt: str,
        user_message: str,
        history: List[dict] | None = None,
        fast: bool = False,
    ) -> str:
        """
        Send a chat completion request to Claude with automatic transient error retries.

        history: chronological list of {"role": "human" | "ai", "content": str}
        """
        messages = []
        if history:
            for turn in history:
                role = "user" if turn.get("role", "human") == "human" else "assistant"
                messages.append({"role": role, "content": turn.get("content", "")})
        messages.append({"role": "user", "content": user_message})

        last_err: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                client = cls.get_client()
                t0 = time.monotonic()
                response = await client.messages.create(
                    model=cls._MODEL,
                    max_tokens=cls._MAX_TOKENS,
                    system=system_prompt,
                    messages=messages,
                    thinking={"type": "disabled"},
                )
                logger.info('[LLMService] %s call ok in %.1fs (attempt %d/%d, %d chars)',
                            cls._MODEL, time.monotonic() - t0, attempt, _MAX_ATTEMPTS,
                            len(system_prompt) + len(user_message))
                return "".join(block.text for block in response.content if block.type == "text")
            except Exception as e:
                last_err = e
                transient = _is_transient(e)
                logger.warning('[LLMService] attempt %d/%d failed (%s): %s',
                               attempt, _MAX_ATTEMPTS,
                               'transient — will retry' if transient and attempt < _MAX_ATTEMPTS
                               else 'giving up', str(e)[:180])
                if not transient or attempt == _MAX_ATTEMPTS:
                    break
                retry_after = None
                if isinstance(e, anthropic.RateLimitError):
                    try:
                        retry_after = float(e.response.headers.get('retry-after', ''))
                    except (ValueError, AttributeError):
                        retry_after = None
                sleep_s = min(retry_after if retry_after else attempt * 2.5, _RETRY_SLEEP_CAP)
                await asyncio.sleep(sleep_s)

        logger.error('LLM call failed after %d attempt(s): %s', _MAX_ATTEMPTS, last_err, exc_info=True)
        if isinstance(last_err, anthropic.RateLimitError):
            return _QUOTA_MESSAGE
        return _GENERIC_FAILURE_MESSAGE

    @staticmethod
    def is_failure(text: str) -> bool:
        """True if `text` is chat()'s failure placeholder, not real model output."""
        return text.startswith(FAILURE_PREFIX)
