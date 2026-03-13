#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llm - LLM 接口封装与提示词管理
"""

from .client  import LLMTuner
from .prompts import SYSTEM_PROMPT

__all__ = ["LLMTuner", "SYSTEM_PROMPT"]
