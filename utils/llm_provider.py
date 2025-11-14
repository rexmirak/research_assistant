"""Provider-agnostic LLM client for Ollama and Gemini."""

import logging
from typing import Any, Dict, Optional

from config import Config

logger = logging.getLogger(__name__)

# Import rate limiter
try:
    from utils.rate_limiter import get_rate_limiter
except ImportError:
    get_rate_limiter = None

# Import Ollama and Gemini clients
try:
    import ollama
except ImportError:
    ollama = None

try:
    from utils.gemini_client import gemini_generate, gemini_generate_json
except ImportError:
    gemini_generate = None
    gemini_generate_json = None


def llm_generate(
    prompt: str, *, model: Optional[str] = None, options: Optional[dict] = None
) -> Dict[str, Any]:
    """
    Provider-agnostic LLM call. Uses provider/model from config.
    Returns a dict with at least a 'response' key (str).
    Logs request and response for debugging.
    """
    cfg = Config()
    provider = getattr(cfg, "llm_provider", "ollama")
    options = options or {}
    
    # Apply rate limiting for Gemini
    if provider == "gemini" and get_rate_limiter:
        rate_limiter = get_rate_limiter(cfg)
        rate_limiter.wait_if_needed()
    
    logger.info(
        f"[LLM REQUEST] Provider: {provider}, Model: {model}, Prompt: {prompt[:500]}{'... [truncated]' if len(prompt) > 500 else ''}"
    )
    if provider == "gemini":
        if not gemini_generate:
            raise ImportError("Gemini client not available.")
        api_key = getattr(cfg, "gemini", None) and getattr(cfg.gemini, "api_key", None)
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in config.")
        gemini_model = model or getattr(cfg.gemini, "model", "gemini-2.0-flash-exp")
        temperature = options.get("temperature", getattr(cfg.gemini, "temperature", 0.1))
        schema = options.get("schema")
        try:
            if schema and gemini_generate_json:
                response = gemini_generate_json(
                    prompt,
                    schema=schema,
                    api_key=api_key,
                    model=gemini_model,
                    temperature=temperature,
                )
                logger.info(
                    f"[LLM RESPONSE] Provider: gemini, JSON Response: {str(response)[:500]}{'... [truncated]' if len(str(response)) > 500 else ''}"
                )
                return {"response": response}
            else:
                response_text = gemini_generate(
                    prompt, api_key=api_key, model=gemini_model, temperature=temperature
                )
                logger.info(
                    f"[LLM RESPONSE] Provider: gemini, Response: {str(response_text)[:500]}{'... [truncated]' if len(str(response_text)) > 500 else ''}"
                )
                return {"response": response_text}
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            raise
    else:
        if not ollama:
            raise ImportError("Ollama client not available.")
        if not model:
            model = getattr(cfg.ollama, "summarize_model", "deepseek-r1:8b")
        # Remove schema from options for Ollama (it doesn't support it)
        ollama_options = {k: v for k, v in options.items() if k != "schema"}
        try:
            result = ollama.generate(model=model, prompt=prompt, options=ollama_options)
            logger.info(
                f"[LLM RESPONSE] Provider: ollama, Response: {str(result.get('response', ''))[:500]}{'... [truncated]' if len(str(result.get('response', ''))) > 500 else ''}"
            )
            return {"response": result["response"]}
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            raise
