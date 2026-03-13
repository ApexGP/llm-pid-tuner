#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llm/client.py - LLM 接口封装（支持 OpenAI / Anthropic SDK 及 HTTP 回退）
"""

import json
import math
import re
import traceback
from typing import Any, Dict, List, Optional

from core.config import CONFIG
from llm.prompts import SYSTEM_PROMPT


class LLMTuner:
    def __init__(
        self, api_key: str, base_url: str, model: str, provider: str = "openai"
    ):
        self.api_key         = api_key
        self.base_url        = (base_url or "").rstrip("/")
        self.model           = model
        self.provider_choice = self._normalize_provider_choice(provider)
        self.provider        = self._resolve_transport()
        self.timeout         = CONFIG.get("LLM_REQUEST_TIMEOUT", 60)
        self.debug_output    = CONFIG.get("LLM_DEBUG_OUTPUT", False)
        self.use_sdk         = False
        self.client          = None

        try:
            if self.provider == "openai":
                import openai

                self.client = openai.OpenAI(api_key=api_key, base_url=self.base_url)
            elif self.provider == "anthropic":
                import anthropic

                self.client = anthropic.Anthropic(
                    api_key=api_key, base_url=self.base_url
                )
        except ImportError:
            # SDK 未安装：回退到 requests
            self.requests = self._import_requests()
        except Exception:
            # 其他初始化错误：调试模式下打印堆栈，然后回退
            if self.debug_output:
                traceback.print_exc()
            self.requests = self._import_requests()
        else:
            self.use_sdk  = True

    @staticmethod
    def _normalize_provider_choice(provider: Optional[str]) -> str:
        provider_choice = str(provider or "").strip().lower()
        provider_choice = provider_choice.replace("-", "_").replace(" ", "_")
        return provider_choice or "openai"

    def _resolve_transport(self) -> str:
        if self.provider_choice in (
            "openai",
            "openai_compat",
            "openai_compatible",
            "openai_claude",
            "claude_openai",
            "claude_relay",
        ):
            return "openai"
        if self.provider_choice in ("anthropic", "anthropic_native", "claude_native"):
            return "anthropic"

        base_url_lower = self.base_url.lower()
        if self.provider_choice == "auto" and "api.anthropic.com" in base_url_lower:
            return "anthropic"

        return "openai"

    def _import_requests(self):
        import requests

        return requests

    def _ensure_requests(self) -> None:
        if not hasattr(self, "requests") or self.requests is None:
            self.requests = self._import_requests()

    def _request_via_http(self, user_prompt: str) -> str:
        self._ensure_requests()

        if self.provider == "anthropic":
            headers = {
                "x-api-key"        : self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type"     : "application/json",
            }
            payload = {
                "model"      : self.model,
                "system"     : SYSTEM_PROMPT,
                "messages"   : [{"role": "user", "content": user_prompt}],
                "temperature": 0.3,
                "max_tokens" : 1000,
            }
            resp = self.requests.post(
                f"{self.base_url}/messages",
                headers = headers,
                json    = payload,
                timeout = self.timeout,
            )
            resp.raise_for_status()
            response_json  = resp.json()
            content_blocks = response_json.get("content", [])
            return "\n".join(
                block.get("text", "")
                for block in content_blocks
                if isinstance(block, dict)
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type" : "application/json",
        }
        payload = {
            "model"   : self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
        }
        resp = self.requests.post(
            f"{self.base_url}/chat/completions",
            headers = headers,
            json    = payload,
            timeout = self.timeout,
        )
        resp.raise_for_status()
        response_json = resp.json()
        return response_json["choices"][0]["message"]["content"]

    def _extract_json_candidates(self, text: str) -> List[str]:
        candidates: List[str] = []
        stripped = text.strip()

        if stripped:
            candidates.append(stripped)

        fenced_matches = re.findall(
            r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE
        )
        candidates.extend(fenced_matches)

        for start in range(len(text)):
            if text[start] != "{":
                continue
            depth = 0
            for end in range(start, len(text)):
                char = text[end]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[start : end + 1])
                        break

        return candidates

    def _sanitize_result(self, data: Dict[str, Any]) -> Dict[str, Any]:
        sanitized = dict(data)

        for key in ("p", "i", "d"):
            value = sanitized.get(key)
            try:
                numeric = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                sanitized.pop(key, None)
                continue

            if not math.isfinite(numeric) or numeric < 0:
                sanitized.pop(key, None)
            else:
                sanitized[key] = numeric

        if "status" in sanitized:
            status = str(sanitized["status"]).strip().upper()
            sanitized["status"] = "DONE" if status == "DONE" else "TUNING"

        if not sanitized.get("analysis_summary"):
            sanitized["analysis_summary"] = str(
                sanitized.get("analysis") or "未提供分析摘要"
            )

        if not sanitized.get("thought_process"):
            sanitized["thought_process"] = str(
                sanitized.get("analysis_summary") or "模型未提供详细推理"
            )

        if not sanitized.get("tuning_action"):
            sanitized["tuning_action"] = "ADJUST_PID"

        return sanitized

    def _parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        for candidate in self._extract_json_candidates(text):
            try:
                return self._sanitize_result(json.loads(candidate))
            except Exception:
                pass
        return None

    def analyze(self, prompt_data: str, history_text: str) -> Optional[Dict[str, Any]]:
        user_prompt = f"""
{history_text}

{prompt_data}

请基于以上历史和当前数据，分析 PID 参数表现并给出优化建议。
务必使用 JSON 格式返回，包含 thought_process 字段。
"""
        try:
            content: str = ""
            if self.use_sdk:
                try:
                    if self.provider == "openai":
                        resp = self.client.chat.completions.create(  # type: ignore[union-attr]
                            model    = self.model,
                            messages = [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": user_prompt},
                            ],
                            temperature = 0.3,
                        )
                        content = resp.choices[0].message.content or ""
                    elif self.provider == "anthropic":
                        resp = self.client.messages.create(  # type: ignore[union-attr]
                            model       = self.model,
                            system      = SYSTEM_PROMPT,
                            messages    = [{"role": "user", "content": user_prompt}],
                            temperature = 0.3,
                            max_tokens  = 1000,
                        )
                        content = resp.content[0].text  # type: ignore[union-attr]
                except Exception as sdk_error:
                    print(f"[WARN] SDK 调用失败，尝试 HTTP 回退: {sdk_error}")
                    content = self._request_via_http(user_prompt)
            else:
                content = self._request_via_http(user_prompt)

            if self.debug_output:
                print(f"\n[LLM 原始响应预览]\n{content[:500]}...\n")

            parsed = self._parse_json(content)
            if parsed:
                return parsed

            print("[WARN] LLM 响应未能解析为 JSON，已忽略本轮建议。")
            return None

        except Exception as e:
            print(f"[ERROR] LLM 调用失败: {e}")
            return None
