import json
import time
import requests
from typing import Any, Dict, List, Optional


class LLMClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        self.api_key = (api_key or "").strip()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.extra_headers = extra_headers or {}

    def _has_real_key(self) -> bool:
        if not self.api_key:
            return False
        return True

    def _chat_completions(self, messages: List[Dict[str, Any]], temperature: float = 0.2, timeout_s: int = 60) -> str:
        """
        OpenRouter(OpenAI-compatible) chat/completions call via requests (with retry on network errors).
        Docs: https://openrouter.ai/docs/api-reference/chat-completion
        """
        if not self._has_real_key():
            raise ValueError(
                "OPENROUTER_API_KEY 未设置。请在环境变量中设置 OPENROUTER_API_KEY 后再运行。"
            )

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {self.api_key}",
            **self.extra_headers,
        }

        # Retry logic for network/SSL errors
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=timeout_s)
                resp.raise_for_status()  # Raise HTTPError for bad status codes
                
                # Parse response
                try:
                    obj = resp.json()
                    return obj["choices"][0]["message"]["content"]
                except Exception:
                    # If parsing fails, return raw body for debugging
                    return resp.text
                    
            except requests.exceptions.HTTPError as e:
                # HTTP errors (4xx, 5xx) should not retry
                err_body = e.response.text if e.response else str(e)
                raise RuntimeError(f"LLM HTTPError {e.response.status_code if e.response else ''}: {err_body}") from e
            except (requests.exceptions.SSLError, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                # Network/SSL errors: retry with exponential backoff
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 1s, 2s, 4s
                    print(f"⚠️  网络错误 (尝试 {attempt + 1}/{max_retries})，{wait_time}秒后重试: {type(e).__name__}")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"LLM 调用失败（已重试 {max_retries} 次）: {e}") from e
            except Exception as e:
                raise RuntimeError(f"LLM 调用失败: {e}") from e

    def infer_requirements(self, prompt):
        """
        Sends the prompt to the LLM and returns the response.
        """
        print(f"--- Sending Prompt to {self.model} ---")
        print(prompt)
        print("---------------------------------------")

        if not self._has_real_key():
            # Mock response for reproduction without actual API key
            return """
1. Requirement: Add a clear visual cue or tooltip to the widget.
   - Target UI Element: Widget 'X' on Page 'Y'
   - Rationale: The user clicked repeatedly, suggesting they expected a response or the button state was unclear.

2. Requirement: Optimize the response time or add a loading spinner.
   - Target UI Element: System Feedback Mechanism
   - Rationale: The user's repetitive clicking indicates frustration with lack of immediate feedback.
""".strip()

        return self._chat_completions(
            messages=[
                {"role": "system", "content": "You are a Requirement Analyst."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

    def infer_intent(self, prompt: str) -> str:
        """
        Intent inference interface.
        If OPENROUTER_API_KEY is set, this will call OpenRouter and return raw text.
        Otherwise it returns a deterministic mock JSON to support running without API credits.
        """
        print(f"--- Sending Intent Prompt to {self.model} ---")
        print(prompt)
        print("--------------------------------------------")

        if self._has_real_key():
            return self._chat_completions(
                messages=[
                    {"role": "system", "content": "You output strictly valid JSON as requested. No extra text."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )

        # Very lightweight mock: choose an intent label based on keywords in prompt
        lowered = (prompt or "").lower()
        intent = "Other"
        if "log in" in lowered or "login" in lowered:
            intent = "Login"
        elif "upload" in lowered or "download" in lowered:
            intent = "Upload/Download"
        elif "repetitive" in lowered or "clicked" in lowered:
            intent = "Waiting/NoFeedback"
        elif "hesitation" in lowered or "long duration" in lowered:
            intent = "Hesitation/Uncertainty"
        elif "navigate" in lowered:
            intent = "Navigate"

        # Extract first evidence idx pattern if present
        ev_idx = "0"
        for token in (prompt or "").split():
            if token.startswith("idx="):
                ev_idx = token.replace("idx=", "").strip()
                break

        return json.dumps(
            {
                "intent": intent,
                "confidence": 0.55,
                "evidence": [{"event_idx": ev_idx, "why": "Mocked evidence reference from STM line."}],
                "notes": "mock_response_no_api_call",
            },
            ensure_ascii=False,
        )
