"""Text-to-speech module based on Parmanus's implementation."""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Optional

from app.logger import logger


class TextToSpeech:
    """Converts text to speech using various TTS engines."""

    def __init__(self, enabled: bool = False, engine: str = "pyttsx3"):
        """Initialize the TTS system.

        Args:
            enabled: Whether TTS is enabled.
            engine: TTS engine to use ('pyttsx3', 'gTTS', 'espeak').
        """
        self.enabled = enabled
        self.engine = engine
        self.tts_engine = None

        if self.enabled:
            self._initialize_engine()

    def _initialize_engine(self):
        """Initialize the TTS engine."""
        try:
            if self.engine == "pyttsx3":
                self._initialize_pyttsx3()
            elif self.engine == "gtts":
                self._initialize_gtts()
            elif self.engine == "espeak":
                self._initialize_espeak()
            else:
                logger.warning(
                    f"Unknown TTS engine: {self.engine}, falling back to pyttsx3"
                )
                self._initialize_pyttsx3()
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine {self.engine}: {e}")
            self.enabled = False

    def _initialize_pyttsx3(self):
        """Initialize pyttsx3 TTS engine."""
        try:
            import pyttsx3

            self.tts_engine = pyttsx3.init()

            # Configure voice properties
            voices = self.tts_engine.getProperty("voices")
            if voices:
                # Try to find a female voice, fallback to first available
                female_voice = next(
                    (v for v in voices if "female" in v.name.lower()), voices[0]
                )
                self.tts_engine.setProperty("voice", female_voice.id)

            # Set speech rate and volume
            self.tts_engine.setProperty("rate", 150)  # Speed of speech
            self.tts_engine.setProperty("volume", 0.8)  # Volume level

            logger.info("pyttsx3 TTS engine initialized")
        except ImportError:
            logger.error("pyttsx3 not available. Install with: pip install pyttsx3")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize pyttsx3: {e}")
            raise

    def _initialize_gtts(self):
        """Initialize Google Text-to-Speech engine."""
        try:
            from gtts import gTTS

            self.gtts_class = gTTS
            logger.info("gTTS engine initialized")
        except ImportError:
            logger.error("gTTS not available. Install with: pip install gtts")
            raise

    def _initialize_espeak(self):
        """Initialize espeak TTS engine."""
        try:
            import subprocess

            # Check if espeak is available
            result = subprocess.run(
                ["espeak", "--version"], capture_output=True, text=True
            )
            if result.returncode == 0:
                logger.info("espeak TTS engine initialized")
            else:
                raise Exception("espeak not found")
        except Exception as e:
            logger.error(
                "espeak not available. Install with: sudo apt-get install espeak"
            )
            raise

    def speak(self, text: str, language: str = "en") -> bool:
        """Convert text to speech and play it.

        Args:
            text: Text to convert to speech.
            language: Language code for speech.

        Returns:
            True if speech was successful, False otherwise.
        """
        if not self.enabled or not text.strip():
            return False

        try:
            if self.engine == "pyttsx3":
                return self._speak_pyttsx3(text)
            elif self.engine == "gtts":
                return self._speak_gtts(text, language)
            elif self.engine == "espeak":
                return self._speak_espeak(text, language)
            else:
                logger.error(f"Unknown TTS engine: {self.engine}")
                return False
        except Exception as e:
            logger.error(f"TTS failed: {e}")
            return False

    def _speak_pyttsx3(self, text: str) -> bool:
        """Speak using pyttsx3 engine.

        Args:
            text: Text to speak.

        Returns:
            True if successful.
        """
        try:
            if self.tts_engine:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                return True
            return False
        except Exception as e:
            logger.error(f"pyttsx3 speech failed: {e}")
            return False

    def _speak_gtts(self, text: str, language: str = "en") -> bool:
        """Speak using Google TTS engine.

        Args:
            text: Text to speak.
            language: Language code.

        Returns:
            True if successful.
        """
        try:
            # Create TTS object
            tts = self.gtts_class(text=text, lang=language, slow=False)

            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tts.save(tmp_file.name)

                # Play the audio file
                self._play_audio_file(tmp_file.name)

                # Clean up
                os.unlink(tmp_file.name)

            return True
        except Exception as e:
            logger.error(f"gTTS speech failed: {e}")
            return False

    def _speak_espeak(self, text: str, language: str = "en") -> bool:
        """Speak using espeak engine.

        Args:
            text: Text to speak.
            language: Language code.

        Returns:
            True if successful.
        """
        try:
            import subprocess

            # Run espeak command
            cmd = ["espeak", "-v", language, "-s", "150", text]
            result = subprocess.run(cmd, capture_output=True)

            return result.returncode == 0
        except Exception as e:
            logger.error(f"espeak speech failed: {e}")
            return False

    def _play_audio_file(self, file_path: str):
        """Play an audio file using available audio players.

        Args:
            file_path: Path to the audio file.
        """
        try:
            import platform
            import subprocess

            system = platform.system().lower()

            if system == "linux":
                # Try different audio players on Linux
                players = ["mpg123", "mpv", "vlc", "aplay"]
                for player in players:
                    try:
                        subprocess.run(
                            [player, file_path], check=True, capture_output=True
                        )
                        return
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        continue

                logger.warning("No audio player found on Linux")

            elif system == "darwin":  # macOS
                subprocess.run(["afplay", file_path], check=True)

            elif system == "windows":
                import winsound

                winsound.PlaySound(file_path, winsound.SND_FILENAME)

        except Exception as e:
            logger.error(f"Failed to play audio file: {e}")

    async def speak_async(self, text: str, language: str = "en") -> bool:
        """Asynchronously convert text to speech.

        Args:
            text: Text to convert to speech.
            language: Language code for speech.

        Returns:
            True if speech was successful, False otherwise.
        """
        if not self.enabled:
            return False

        # Run TTS in a separate thread to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.speak, text, language)

    def set_voice_properties(
        self, rate: Optional[int] = None, volume: Optional[float] = None
    ):
        """Set voice properties for pyttsx3 engine.

        Args:
            rate: Speech rate (words per minute).
            volume: Volume level (0.0 to 1.0).
        """
        if self.engine == "pyttsx3" and self.tts_engine:
            try:
                if rate is not None:
                    self.tts_engine.setProperty("rate", rate)
                if volume is not None:
                    self.tts_engine.setProperty("volume", volume)
            except Exception as e:
                logger.error(f"Failed to set voice properties: {e}")

    def get_available_voices(self) -> list:
        """Get list of available voices.

        Returns:
            List of available voice information.
        """
        if self.engine == "pyttsx3" and self.tts_engine:
            try:
                voices = self.tts_engine.getProperty("voices")
                return (
                    [
                        {"id": v.id, "name": v.name, "languages": v.languages}
                        for v in voices
                    ]
                    if voices
                    else []
                )
            except Exception as e:
                logger.error(f"Failed to get voices: {e}")
                return []
        return []

    def set_voice(self, voice_id: str):
        """Set the voice to use.

        Args:
            voice_id: ID of the voice to use.
        """
        if self.engine == "pyttsx3" and self.tts_engine:
            try:
                self.tts_engine.setProperty("voice", voice_id)
            except Exception as e:
                logger.error(f"Failed to set voice: {e}")

    def stop(self):
        """Stop current speech."""
        if self.engine == "pyttsx3" and self.tts_engine:
            try:
                self.tts_engine.stop()
            except Exception as e:
                logger.error(f"Failed to stop speech: {e}")

    def __del__(self):
        """Cleanup TTS engine."""
        if hasattr(self, "tts_engine") and self.tts_engine:
            try:
                self.tts_engine.stop()
            except:
                pass
