"""Voice interaction module for Parmanus integration."""

from .stt import SpeechToText
from .tts import TextToSpeech


__all__ = ["TextToSpeech", "SpeechToText"]
