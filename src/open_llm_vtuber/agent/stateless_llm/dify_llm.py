"""Description: This file contains the implementation of the `DifyLLM` class.
This class is responsible for handling asynchronous interaction with Dify API
for language generation.
"""

import json
import aiohttp
from typing import AsyncIterator, List, Dict, Any
from loguru import logger

from .stateless_llm_interface import StatelessLLMInterface


# Global cache for Dify conversation IDs, keyed by user
# This ensures each user maintains their own conversation state
_dify_conversation_cache: Dict[str, str] = {}

# Global cache for Dify task IDs, keyed by user
# This ensures each user maintains their own task state
_dify_task_id_cache: Dict[str, str] = {}


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
        self.user = user
        self.inputs = inputs or {}
        self.query = query
        self.response_mode = response_mode
        self.support_tools = False  # Dify doesn't support tools in the same way

        # Load existing conversation_id from cache if available
        self.conversation_id = _dify_conversation_cache.get(self.user)
        self.task_id = None

        logger.info(
            f"Initialized DifyLLM with base_url: {self.base_url}, response_mode: {self.response_mode}, user: {self.user}"
        )

    def set_task_id(self, task_id: str) -> None:
        """
        Set the task ID for this Dify instance and store it in the global cache.
        This should be called when a task_id is received from the API.

        Args:
            task_id: The task ID from Dify API
        """
        self.task_id = task_id
        _dify_task_id_cache[self.user] = task_id

    def get_task_id(self) -> str | None:
        """
        Returns:
            The current task ID or None if not set
        """
        return _dify_task_id_cache.get(self.user)

    def set_conversation_id(self, conversation_id: str) -> None:
        """
        Set the conversation ID for this Dify instance and store it in the global cache.
        This should be called when a conversation_id is received from the API.

        Args:
            conversation_id: The conversation ID from Dify API
        """
        self.conversation_id = conversation_id
        # Store in global cache keyed by user
        _dify_conversation_cache[self.user] = conversation_id
        logger.debug(
            f"Set Dify conversation_id for user '{self.user}': {conversation_id}"
        )

    def get_conversation_id(self) -> str | None:
        """
        Get the current conversation ID from cache.

        Returns:
            The current conversation ID or None if not set
        """
        return _dify_conversation_cache.get(self.user)

    def get_cached_conversation_id(self) -> str | None:
        """
        Get the conversation ID from the global cache for this user.

        Returns:
            The cached conversation ID or None if not found
        """
        return _dify_conversation_cache.get(self.user)

    def clear_conversation_id(self) -> None:
        """
        Clear the conversation ID for this user from cache.
        Useful for starting a new conversation.
        """
        if self.user in _dify_conversation_cache:
            del _dify_conversation_cache[self.user]
            self.conversation_id = None
            logger.debug(f"Cleared Dify conversation_id for user '{self.user}'")

    async def cancel_task(self, task_id: str) -> None:
        """
        Cancel a running chat message task by its ID.

        Args:
            task_id: Task ID to stop
        """
        try:
            stop_url = f"{self.base_url}/chat-messages/{task_id}/stop"
            
            logger.info(f"Stopping Dify task: {task_id} for user: {self.user}")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    stop_url,
                    json={"user": self.user},
                    headers={"Authorization": f"Bearer {self.api_key}"}
                ) as response:
                    if response.status == 200:
                        logger.info(
                            f"Dify task stopped successfully: {task_id} for user: {self.user}, status: {response.status}"
                        )
                    else:
                        logger.warning(
                            f"Dify task stop request returned non-200 status: {task_id} for user: {self.user}, status: {response.status}"
                        )
                        
        except Exception as error:
            logger.error(
                f"Error stopping Dify task: {task_id} for user: {self.user}, error: {str(error)}"
            )

    @classmethod
    def get_cache_info(cls) -> Dict[str, str]:
        """
        Get information about all cached conversation IDs.
        Useful for debugging and monitoring.

        Returns:
            Dictionary mapping user to conversation_id
        """
        return _dify_conversation_cache.copy()

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

            # Extract and process the content from the last user message
            last_message_content = user_messages[-1].get("content", "")
            if not last_message_content:
                yield "Error: Empty user message"
                return

            # Handle VTuber message format: [{'type': 'text', 'text': 'halo'}]
            if isinstance(last_message_content, list):
                # Extract all text content from the array
                text_parts = []
                for item in last_message_content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text = item.get("text", "").strip()
                        if text:
                            text_parts.append(text)

                if not text_parts:
                    yield "Error: No text content found in message"
                    return

                # Combine all text parts with ", " as separator
                query = ", ".join(text_parts)
                logger.debug(
                    f"Combined VTuber message parts: {text_parts} -> '{query}'"
                )
            else:
                # Handle regular string content
                query = str(last_message_content)

            # Log the processed query for debugging
            logger.debug(f"Original message content: {last_message_content}")
            logger.debug(f"Processed query for Dify: '{query}'")

            task_id = self.get_task_id()

            # Cancel previously running task if it exists
            if task_id:
                await self.cancel_task(task_id)

            # Prepare the request payload
            payload = {
                "inputs": self.inputs,
                "query": query,
                "response_mode": self.response_mode,
                "user": self.user,
            }

            # Get conversation_id from cache
            conversation_id = self.get_conversation_id()
            if conversation_id:
                payload["conversation_id"] = conversation_id

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
                                    
                                    if "task_id" in json_data:
                                        self.set_task_id(json_data["task_id"])
                                        logger.info(
                                            f"Received task_id from Dify: {json_data['task_id']}"
                                        )
                                        
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
