# -*- coding: utf-8 -*-
"""
Gemini Provider - Google Generative AI integration.
"""

import os
from typing import Any, AsyncGenerator, Dict, Optional
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from .base_provider import BaseLLMProvider
from ..types import TutorResponse
from ..exceptions import LLMAPIError, LLMAuthenticationError, LLMRateLimitError

class GeminiProvider(BaseLLMProvider):
    """
    Provider for Google's Gemini models via google-generativeai library.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        if not self.api_key:
            raise LLMAuthenticationError("Google API Key is required for Gemini provider")
            
        genai.configure(api_key=self.api_key)
        
        # Generation config
        self.generation_config = {
            "temperature": config.temperature or 0.7,
            "top_p": config.top_p or 0.95,
            "top_k": config.top_k or 40,
            "max_output_tokens": config.max_tokens or 8192,
        }
        
        # Safety settings - Block only high probability of unsafe content
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }

    async def complete(self, prompt: str, **kwargs) -> TutorResponse:
        """
        Complete a prompt using Gemini.
        """
        async def _call_api():
            model_name = kwargs.get("model") or self.config.model or "gemini-1.5-flash"
            
            # Handle system prompt if present
            system_prompt = kwargs.get("system_prompt")
            if system_prompt:
                # Gemini 1.5 supports system instructions, but strict enforcement varies.
                # Only 1.5 Pro/Flash latest versions support system_instruction arg officially in some SDK versions.
                # For safety, we can prepend it or use the argument if we are sure of the version.
                # Let's prepend for valid broad compatibility or use system_instruction if model supports it.
                model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings,
                    system_instruction=system_prompt
                )
            else:
                model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings
                )
                
            response = await model.generate_content_async(prompt)
            
            # Check finish reason
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                raise LLMAPIError(f"Blocked: {response.prompt_feedback.block_reason}")
                
            return TutorResponse(
                content=response.text,
                original_response=response
            )

        return await self.execute_with_retry(_call_api)

    async def stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """
        Stream completion.
        """
        # Note: 'execute_with_retry' is designed for atomic calls, not streaming generators directly.
        # We usually retry connection, then stream.
        
        model_name = kwargs.get("model") or self.config.model or "gemini-1.5-flash"
        system_prompt = kwargs.get("system_prompt")
        
        try:
            if system_prompt:
                model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings,
                    system_instruction=system_prompt
                )
            else:
                model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings
                )
                
            # Gemini streaming
            response = await model.generate_content_async(prompt, stream=True)
            
            async for chunk in response:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            # Map exception using parent logic if needed or raise
            raise self._map_exception(e)
