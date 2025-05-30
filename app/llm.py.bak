import asyncio
import base64
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from io import BytesIO
from typing import Any, Dict, List, Optional, Union, Generator

import tiktoken
from llama_cpp import Llama
from pydantic import BaseModel

from app.config import LLMSettings, config
from app.exceptions import TokenLimitExceeded
from app.logger import logger
from app.schema import (
    ROLE_VALUES,
    TOOL_CHOICE_TYPE,
    TOOL_CHOICE_VALUES,
    Message,
    ToolChoice,
)

# Define models that support vision capabilities
MULTIMODAL_MODELS = ["qwen-vl-7b"]

# Global model cache to persist models between requests
MODEL_CACHE = {}

class TokenCounter:
    # Token constants
    BASE_MESSAGE_TOKENS = 4
    FORMAT_TOKENS = 2
    LOW_DETAIL_IMAGE_TOKENS = 85
    HIGH_DETAIL_TILE_TOKENS = 170

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def update(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens = self.prompt_tokens + self.completion_tokens

    def reset(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

class ChatCompletionMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Dict[str, Any]] = None

class LLM:
    """
    LLM class for handling interactions with local GGUF models using llama-cpp-python.
    """
    # Define the context window sizes as class variables
    TEXT_MODEL_CONTEXT_SIZE = 16384  # Increased from 8192
    VISION_MODEL_CONTEXT_SIZE = 8192  # Increased from 4096
    
    # Define maximum allowed output tokens to prevent token limit errors
    MAX_ALLOWED_OUTPUT_TOKENS = 8192
    
    # Thread pool for model loading and inference
    _executor = ThreadPoolExecutor(max_workers=2)

    def __init__(self, settings: Optional[LLMSettings] = None, config_name: str = None):
        """
        Initialize the LLM with settings.
        Args:
            settings: LLM configuration settings
            config_name: Optional name for config lookup
        """
        # Ensure settings is an LLMSettings instance
        if settings is None:
            settings = config.llm
        # Double-check that settings is an LLMSettings instance
        if not isinstance(settings, LLMSettings):
            raise TypeError(f"Expected LLMSettings instance, got {type(settings)}")
        
        self.model = settings.model
        self.model_path = settings.model_path
        self.max_tokens = settings.max_tokens
        self.temperature = settings.temperature
        self.token_counter = TokenCounter()
        
        # Vision model settings
        self.vision_settings = settings.vision
        
        # Cache keys for model instances
        self._text_model_key = f"{self.model_path}_{self.TEXT_MODEL_CONTEXT_SIZE}"
        self._vision_model_key = None
        if self.vision_settings:
            self._vision_model_key = f"{self.vision_settings.model_path}_{self.VISION_MODEL_CONTEXT_SIZE}"
        
        logger.info(f"Initialized LLM with model: {self.model}, path: {self.model_path}")
        
        # Preload models in background if not already loaded
        if self._text_model_key not in MODEL_CACHE:
            asyncio.create_task(self._preload_text_model())
        
        if self._vision_model_key and self._vision_model_key not in MODEL_CACHE:
            asyncio.create_task(self._preload_vision_model())

    async def _preload_text_model(self):
        """Preload text model in background"""
        try:
            logger.info(f"Preloading text model from {self.model_path}")
            await asyncio.get_event_loop().run_in_executor(
                self._executor,
                self._load_text_model
            )
            logger.info(f"Text model preloaded successfully")
        except Exception as e:
            logger.error(f"Error preloading text model: {e}")

    async def _preload_vision_model(self):
        """Preload vision model in background"""
        if not self.vision_settings:
            return
            
        try:
            logger.info(f"Preloading vision model from {self.vision_settings.model_path}")
            await asyncio.get_event_loop().run_in_executor(
                self._executor,
                self._load_vision_model
            )
            logger.info(f"Vision model preloaded successfully")
        except Exception as e:
            logger.error(f"Error preloading vision model: {e}")

    def _load_text_model(self):
        """Load text model with memory mapping for faster loading"""
        if self._text_model_key not in MODEL_CACHE:
            logger.info(f"Loading text model from {self.model_path}")
            start_time = time.time()
            
            model = Llama(
                model_path=self.model_path,
                n_ctx=self.TEXT_MODEL_CONTEXT_SIZE,
                n_gpu_layers=-1,  # Use all GPU layers
                n_threads=os.cpu_count(),  # Use all available CPU threads
                use_mmap=True,  # Use memory mapping for faster loading
                use_mlock=True,  # Lock memory to prevent swapping
            )
            
            MODEL_CACHE[self._text_model_key] = model
            load_time = time.time() - start_time
            logger.info(f"Text model loaded in {load_time:.2f} seconds")
        
        return MODEL_CACHE[self._text_model_key]

    def _load_vision_model(self):
        """Load vision model with memory mapping for faster loading"""
        if not self.vision_settings:
            return None
            
        if self._vision_model_key not in MODEL_CACHE:
            logger.info(f"Loading vision model from {self.vision_settings.model_path}")
            start_time = time.time()
            
            model = Llama(
                model_path=self.vision_settings.model_path,
                n_ctx=self.VISION_MODEL_CONTEXT_SIZE,
                n_gpu_layers=-1,  # Use all GPU layers
                n_threads=os.cpu_count(),  # Use all available CPU threads
                use_mmap=True,  # Use memory mapping for faster loading
                use_mlock=True,  # Lock memory to prevent swapping
            )
            
            MODEL_CACHE[self._vision_model_key] = model
            load_time = time.time() - start_time
            logger.info(f"Vision model loaded in {load_time:.2f} seconds")
        
        return MODEL_CACHE[self._vision_model_key]

    @property
    def text_model(self):
        """Get cached text model or load if not available"""
        if self._text_model_key in MODEL_CACHE:
            return MODEL_CACHE[self._text_model_key]
        return self._load_text_model()

    @property
    def vision_model(self):
        """Get cached vision model or load if not available"""
        if not self.vision_settings:
            return None
            
        if self._vision_model_key in MODEL_CACHE:
            return MODEL_CACHE[self._vision_model_key]
        return self._load_vision_model()

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.
        This is an estimate and may not match the exact tokenization of the model.
        """
        # Simple approximation: 1 token ≈ 4 characters for English text
        return len(text) // 4 + 1

    def count_message_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """
        Count the number of tokens in a list of messages.
        This is an estimate and may not match the exact tokenization of the model.
        """
        total_tokens = 0
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                total_tokens += self.count_tokens(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            total_tokens += self.count_tokens(item.get("text", ""))
                        elif item.get("type") == "image_url":
                            # Add token estimate for images
                            total_tokens += self.TokenCounter.LOW_DETAIL_IMAGE_TOKENS
        return total_tokens

    def update_token_count(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Update the token counter with the latest usage."""
        self.token_counter.update(prompt_tokens, completion_tokens)

    def check_token_limit(self, input_tokens: int, has_images: bool = False) -> bool:
        """Check if the input tokens exceed the model's context window."""
        # Adjust context size based on whether we're using vision model
        context_size = (
            self.VISION_MODEL_CONTEXT_SIZE if has_images else self.TEXT_MODEL_CONTEXT_SIZE
        )
        # Reserve space for completion tokens
        available_tokens = context_size - self.max_tokens
        return input_tokens <= available_tokens

    def get_limit_error_message(self, input_tokens: int, has_images: bool = False) -> str:
        """Generate an error message for token limit exceeded."""
        context_size = (
            self.VISION_MODEL_CONTEXT_SIZE if has_images else self.TEXT_MODEL_CONTEXT_SIZE
        )
        available_tokens = context_size - self.max_tokens
        return (
            f"Input tokens ({input_tokens}) exceed available context window "
            f"({available_tokens} tokens). Please reduce your input."
        )

    def format_messages(
        self, messages: List[Union[Message, Dict[str, Any]]], supports_images: bool = False
    ) -> List[Dict[str, Any]]:
        """Format messages for the model."""
        formatted_messages = []
        for message in messages:
            if isinstance(message, Message):
                message_dict = message.model_dump()
            else:
                message_dict = message.copy()
            
            # Ensure role is valid
            if "role" not in message_dict or message_dict["role"] not in ROLE_VALUES:
                message_dict["role"] = "user"
            
            formatted_messages.append(message_dict)
        
        return formatted_messages

    def _format_prompt_for_llama(self, messages: List[Dict[str, Any]]) -> str:
        """Format messages into a prompt string for Llama models."""
        prompt = ""
        for message in messages:
            role = message["role"]
            content = message.get("content", "")
            
            if role == "system":
                prompt += f"<|system|>\n{content}\n"
            elif role == "user":
                prompt += f"<|user|>\n{content}\n"
            elif role == "assistant":
                prompt += f"<|assistant|>\n{content}\n"
            else:
                # Default to user for unknown roles
                prompt += f"<|user|>\n{content}\n"
        
        # Add the final assistant prefix to prompt the model to respond
        prompt += "<|assistant|>\n"
        return prompt

    def _format_vision_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Format messages for vision models, handling image content."""
        prompt = ""
        for message in messages:
            role = message["role"]
            content = message.get("content", "")
            
            if role == "system":
                prompt += f"<|system|>\n{content}\n"
            elif role == "user":
                if isinstance(content, str):
                    prompt += f"<|user|>\n{content}\n"
                elif isinstance(content, list):
                    prompt += "<|user|>\n"
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                prompt += f"{item.get('text', '')}\n"
                            elif item.get("type") == "image_url":
                                # For now, just indicate an image was here
                                # Vision handling will need to be model-specific
                                prompt += "[IMAGE]\n"
            elif role == "assistant":
                prompt += f"<|assistant|>\n{content}\n"
            else:
                # Default to user for unknown roles
                prompt += f"<|user|>\n{content}\n"
        
        # Add the final assistant prefix to prompt the model to respond
        prompt += "<|assistant|>\n"
        return prompt

    async def ask(
        self,
        messages: List[Union[Message, Dict[str, Any]]],
        system_msgs: Optional[List[Union[Message, Dict[str, Any]]]] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        timeout: int = 120,
        **kwargs,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Send a request to the model and get a response.
        
        Args:
            messages: List of messages to send to the model
            system_msgs: Optional system messages to prepend
            temperature: Temperature for sampling (0.0 to 1.0)
            stream: Whether to stream the response
            timeout: Timeout in seconds for the request
            **kwargs: Additional arguments to pass to the model
            
        Returns:
            The model's response as a string or a generator of strings if streaming
            
        Raises:
            TokenLimitExceeded: If the input exceeds the model's token limit
            Exception: For other unexpected errors
        """
        try:
            # Check if the model supports images
            supports_images = self.model in MULTIMODAL_MODELS
            
            # Format messages
            if system_msgs:
                system_msgs = self.format_messages(system_msgs, supports_images)
                messages = system_msgs + self.format_messages(messages, supports_images)
            else:
                messages = self.format_messages(messages, supports_images)
            
            # Check for images in messages
            has_images = False
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            has_images = True
                            break
            
            # Calculate input token count
            input_tokens = self.count_message_tokens(messages)
            
            # Check if token limits are exceeded
            if not self.check_token_limit(input_tokens, has_images):
                error_message = self.get_limit_error_message(input_tokens, has_images)
                # Raise a special exception that won't be retried
                raise TokenLimitExceeded(error_message)
            
            # Use vision model if content has images and vision model is available
            if has_images and supports_images and self.vision_model:
                logger.info("Using vision model for image content")
                prompt = self._format_vision_prompt(messages)
                model = self.vision_model
            else:
                prompt = self._format_prompt_for_llama(messages)
                model = self.text_model
            
            # Set temperature
            temp = temperature if temperature is not None else self.temperature
            
            # Apply safe max tokens limit
            safe_max_tokens = min(self.max_tokens, self.MAX_ALLOWED_OUTPUT_TOKENS)
            
            if stream:
                # Create streaming response
                async def response_stream():
                    try:
                        # Create a generator that yields completion chunks
                        completion_generator = model.create_completion(
                            prompt=prompt,
                            max_tokens=safe_max_tokens,  # Use safe max tokens
                            temperature=temp,
                            stop=["<|user|>", "<|system|>"],
                            stream=True,
                            **kwargs
                        )
                        
                        total_text = ""
                        for chunk in completion_generator:
                            chunk_text = chunk.get("choices", [{}])[0].get("text", "")
                            total_text += chunk_text
                            yield chunk_text
                        
                        # Update token counts after streaming completes
                        prompt_tokens = self.count_tokens(prompt)
                        completion_tokens = self.count_tokens(total_text)
                        self.update_token_count(prompt_tokens, completion_tokens)
                    except Exception as e:
                        logger.error(f"Error in streaming response: {e}")
                        raise
                
                return response_stream()
            else:
                # Create a task for model completion with timeout and run in thread pool
                try:
                    # Run model inference in thread pool to avoid blocking the event loop
                    completion = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            self._executor,
                            lambda: model.create_completion(
                                prompt=prompt,
                                max_tokens=safe_max_tokens,
                                temperature=temp,
                                stop=["<|user|>", "<|system|>"],
                                **kwargs
                            )
                        ),
                        timeout=timeout
                    )
                    
                    # Extract completion text
                    completion_text = completion.get("choices", [{}])[0].get("text", "").strip()
                    
                    # Estimate token counts
                    prompt_tokens = self.count_tokens(prompt)
                    completion_tokens = self.count_tokens(completion_text)
                    
                    # Update token counter
                    self.update_token_count(prompt_tokens, completion_tokens)
                    
                    return completion_text
                except asyncio.TimeoutError:
                    logger.error(f"Model completion timed out after {timeout} seconds")
                    # Return partial result if available
                    return f"[Response incomplete due to timeout after {timeout} seconds]"
                except Exception as e:
                    logger.error(f"Error in model completion: {e}")
                    raise
        except TokenLimitExceeded:
            # Re-raise token limit errors without logging
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask: {e}")
            raise

    @classmethod
    def cleanup_models(cls):
        """Clean up model resources when shutting down"""
        for key, model in MODEL_CACHE.items():
            try:
                # Free up model resources if possible
                if hasattr(model, 'reset'):
                    model.reset()
                logger.info(f"Cleaned up model: {key}")
            except Exception as e:
                logger.error(f"Error cleaning up model {key}: {e}")
        
        # Clear the cache
        MODEL_CACHE.clear()
        
        # Shutdown thread pool
        cls._executor.shutdown(wait=True)
        logger.info("Model resources cleaned up")
