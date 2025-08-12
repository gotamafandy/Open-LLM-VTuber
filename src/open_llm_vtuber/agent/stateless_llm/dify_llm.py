"""Description: This file contains the implementation of the `DifyLLM` class.
This class is responsible for handling asynchronous interaction with Dify API
for language generation.
"""

import json
import aiohttp
from typing import AsyncIterator, List, Dict, Any
from loguru import logger

from .stateless_llm_interface import StatelessLLMInterface


class DifyLLM(StatelessLLMInterface):
    """Dify LLM implementation for chat completion."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.dify.ai/v1",
        user: str = "user",
        inputs: Dict[str, Any] = None,
        query: str = None,
        response_mode: str = "streaming",
    ):
        """
        Initializes an instance of the `DifyLLM` class.

        Args:
            api_key: The API key for Dify.
            base_url: The base URL for the Dify API. Defaults to "https://api.dify.ai/v1".
            user: The user identifier. Defaults to "user".
            inputs: Additional inputs for the conversation. Defaults to None.
            query: The query text. Defaults to None.
            response_mode: The response mode. Defaults to "streaming".
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.conversation_id = None  # Will be set dynamically from API response
        self.user = user
        self.inputs = inputs or {}
        self.query = query
        self.response_mode = response_mode
        self.support_tools = False  # Dify doesn't support tools in the same way

        logger.info(
            f"Initialized DifyLLM with base_url: {self.base_url}, response_mode: {self.response_mode}"
        )

    def set_conversation_id(self, conversation_id: str) -> None:
        """
        Set the conversation ID for this Dify instance.
        This should be called when a conversation_id is received from the API.

        Args:
            conversation_id: The conversation ID from Dify API
        """
        self.conversation_id = conversation_id
        logger.debug(f"Set Dify conversation_id: {conversation_id}")

    def get_conversation_id(self) -> str | None:
        """
        Get the current conversation ID.

        Returns:
            The current conversation ID or None if not set
        """
        return self.conversation_id

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        system: str = None,
        tools: List[Dict[str, Any]] = None,
    ) -> AsyncIterator[str]:
        """
        Generates a chat completion using the Dify API asynchronously.

        Args:
            messages: The list of messages to send to the API.
            system: System prompt to use for this completion (not used in Dify).
            tools: List of tools to use for this completion (not supported in Dify).

        Yields:
            str: The content of each chunk from the API response.

        Raises:
            Exception: For API-related errors.
        """
        if tools:
            logger.warning(
                "Dify doesn't support tools in the same way as OpenAI. Tools will be ignored."
            )

        try:
            # Extract the last user message as the query
            if not messages:
                yield "Error: No messages provided"
                return

            # Find the last user message
            user_messages = [msg for msg in messages if msg.get("role") == "user"]
            if not user_messages:
                yield "Error: No user messages found"
                return

            query = user_messages[-1].get("content", "")
            if not query:
                yield "Error: Empty user message"
                return

            # Prepare the request payload
            payload = {
                "inputs": self.inputs,
                "query": query,
                "response_mode": self.response_mode,
                "user": self.user,
            }

            if self.conversation_id:
                payload["conversation_id"] = self.conversation_id

            # Set up headers
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            # Make the API request
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/chat-messages"

                async with session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(
                            f"Dify API error: {response.status} - {error_text}"
                        )
                        yield f"Error: Dify API returned status {response.status}"
                        return

                    if self.response_mode == "streaming":
                        # Handle streaming response
                        async for line in response.content:
                            line = line.decode("utf-8").strip()
                            if line.startswith("data: "):
                                data = line[6:]  # Remove 'data: ' prefix
                                if data == "[DONE]":
                                    break
                                try:
                                    json_data = json.loads(data)
                                    # Extract conversation_id if present (usually in first chunk)
                                    if (
                                        "conversation_id" in json_data
                                        and not self.conversation_id
                                    ):
                                        self.set_conversation_id(
                                            json_data["conversation_id"]
                                        )
                                        logger.info(
                                            f"Received conversation_id from Dify: {json_data['conversation_id']}"
                                        )

                                    if "answer" in json_data:
                                        yield json_data["answer"]
                                except json.JSONDecodeError:
                                    logger.warning(f"Failed to parse JSON: {data}")
                                    continue
                    else:
                        # Handle non-streaming response
                        data = await response.json()

                        # Extract conversation_id if present
                        if "conversation_id" in data and not self.conversation_id:
                            self.set_conversation_id(data["conversation_id"])
                            logger.info(
                                f"Received conversation_id from Dify: {data['conversation_id']}"
                            )

                        if "answer" in data:
                            yield data["answer"]
                        else:
                            logger.warning(f"Unexpected response format: {data}")
                            yield "Error: Unexpected response format from Dify API"

        except aiohttp.ClientError as e:
            logger.error(f"Connection error with Dify API: {e}")
            yield f"Error: Failed to connect to Dify API - {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error with Dify API: {e}")
            yield f"Error: Unexpected error occurred - {str(e)}"
