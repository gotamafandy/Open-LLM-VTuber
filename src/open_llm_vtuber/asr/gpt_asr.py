import os
from typing import Callable
import numpy as np
from loguru import logger
import openai
from .asr_interface import ASRInterface
import soundfile as sf
import uuid
import asyncio

CACHE_DIR = "cache"


class VoiceRecognition(ASRInterface):
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "whisper-1",
        language: str | None = None,
        response_format: str = "text",
        temperature: float = 0.0,
        prompt: str | None = None,
        callback: Callable = logger.info,
    ):
        """Initialize OpenAI Speech-to-Text service.

        Args:
            api_key: OpenAI API key. If None, will try to get from OPENAI_API_KEY env var.
            model: The model to use for transcription. Defaults to "whisper-1".
            language: The language of the input audio. If None, will auto-detect.
            response_format: The format of the transcript output. Defaults to "text".
            temperature: The sampling temperature. Defaults to 0.0 for deterministic results.
            prompt: An optional text to guide the model's style or continue a previous audio segment.
            callback: Callback function for logging. Defaults to logger.info.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Please set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.model = model
        self.language = language
        self.response_format = response_format
        self.temperature = temperature
        self.prompt = prompt
        self.callback = callback

        try:
            self.client = openai.AsyncOpenAI(api_key=self.api_key)
            logger.info(f"Initialized OpenAI Speech-to-Text with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

    async def async_transcribe_np(self, audio: np.ndarray) -> str:
        """
        Asynchronously transcribe audio data using OpenAI Speech-to-Text API.

        Args:
            audio (np.ndarray): Audio data as numpy array

        Returns:
            str: Transcribed text

        Raises:
            Exception: If transcription fails
        """
        temp_file = os.path.join(CACHE_DIR, f"{uuid.uuid4()}.wav")

        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            # Save audio as WAV file (OpenAI expects audio files)
            sf.write(temp_file, audio, 16000, "PCM_16")

            # Prepare the transcription request
            transcription_params = {
                "model": self.model,
                "response_format": self.response_format,
                "temperature": self.temperature,
            }

            if self.language:
                transcription_params["language"] = self.language

            if self.prompt:
                transcription_params["prompt"] = self.prompt

            # Read the audio file and send to OpenAI
            with open(temp_file, "rb") as audio_file:
                audio_data = audio_file.read()

                response = await self.client.audio.transcriptions.create(
                    file=("audio.wav", audio_data, "audio/wav"), **transcription_params
                )

            if self.response_format == "text":
                return response
            else:
                # Handle other response formats if needed
                logger.warning(f"Unexpected response format: {self.response_format}")
                return str(response)

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
        finally:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logger.debug(f"Failed to remove temporary file {temp_file}: {e}")

    def transcribe_np(self, audio: np.ndarray) -> str:
        """
        Synchronously transcribe audio data using OpenAI Speech-to-Text API.

        Args:
            audio (np.ndarray): Audio data as numpy array

        Returns:
            str: Transcribed text

        Raises:
            Exception: If transcription fails
        """
        try:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Run async method synchronously
            return loop.run_until_complete(self.async_transcribe_np(audio))
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise


if __name__ == "__main__":
    service = VoiceRecognition()
