"""Speech-to-text module based on Parmanus's implementation."""

import asyncio
import threading
import time
from queue import Queue
from typing import Callable, Optional

from app.logger import logger


class SpeechToText:
    """Converts speech to text using various STT engines."""

    def __init__(
        self,
        enabled: bool = False,
        trigger_word: Optional[str] = None,
        engine: str = "speech_recognition",
    ):
        """Initialize the STT system.

        Args:
            enabled: Whether STT is enabled.
            trigger_word: Wake word to activate listening.
            engine: STT engine to use ('speech_recognition', 'whisper').
        """
        self.enabled = enabled
        self.trigger_word = trigger_word.lower() if trigger_word else None
        self.engine = engine
        self.recognizer = None
        self.microphone = None
        self.is_listening = False
        self.listen_thread = None
        self.audio_queue = Queue()

        if self.enabled:
            self._initialize_engine()

    def _initialize_engine(self):
        """Initialize the STT engine."""
        try:
            if self.engine == "speech_recognition":
                self._initialize_speech_recognition()
            elif self.engine == "whisper":
                self._initialize_whisper()
            else:
                logger.warning(
                    f"Unknown STT engine: {self.engine}, falling back to speech_recognition"
                )
                self._initialize_speech_recognition()
        except Exception as e:
            logger.error(f"Failed to initialize STT engine {self.engine}: {e}")
            self.enabled = False

    def _initialize_speech_recognition(self):
        """Initialize speech_recognition STT engine."""
        try:
            import speech_recognition as sr

            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()

            # Adjust for ambient noise
            with self.microphone as source:
                logger.info("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)

            logger.info("speech_recognition STT engine initialized")
        except ImportError:
            logger.error(
                "speech_recognition not available. Install with: pip install SpeechRecognition"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize speech_recognition: {e}")
            raise

    def _initialize_whisper(self):
        """Initialize Whisper STT engine."""
        try:
            import whisper

            self.whisper_model = whisper.load_model("base")
            logger.info("Whisper STT engine initialized")
        except ImportError:
            logger.error(
                "whisper not available. Install with: pip install openai-whisper"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize whisper: {e}")
            raise

    def listen(
        self, timeout: float = 5.0, phrase_timeout: float = 1.0
    ) -> Optional[str]:
        """Listen for speech and convert to text.

        Args:
            timeout: Maximum time to wait for speech.
            phrase_timeout: Time to wait for phrase completion.

        Returns:
            Recognized text or None if no speech detected.
        """
        if not self.enabled:
            return None

        try:
            if self.engine == "speech_recognition":
                return self._listen_speech_recognition(timeout, phrase_timeout)
            elif self.engine == "whisper":
                return self._listen_whisper(timeout)
            else:
                logger.error(f"Unknown STT engine: {self.engine}")
                return None
        except Exception as e:
            logger.error(f"STT failed: {e}")
            return None

    def _listen_speech_recognition(
        self, timeout: float, phrase_timeout: float
    ) -> Optional[str]:
        """Listen using speech_recognition engine.

        Args:
            timeout: Maximum time to wait for speech.
            phrase_timeout: Time to wait for phrase completion.

        Returns:
            Recognized text or None.
        """
        try:
            with self.microphone as source:
                logger.info("Listening for speech...")
                audio = self.recognizer.listen(
                    source, timeout=timeout, phrase_time_limit=phrase_timeout
                )

            logger.info("Processing speech...")

            # Try different recognition services
            try:
                # Google Speech Recognition (free)
                text = self.recognizer.recognize_google(audio)
                logger.info(f"Recognized: {text}")
                return text
            except Exception:
                try:
                    # Sphinx (offline)
                    text = self.recognizer.recognize_sphinx(audio)
                    logger.info(f"Recognized (Sphinx): {text}")
                    return text
                except Exception as e:
                    logger.error(f"All recognition services failed: {e}")
                    return None

        except Exception as e:
            logger.error(f"Speech recognition failed: {e}")
            return None

    def _listen_whisper(self, timeout: float) -> Optional[str]:
        """Listen using Whisper engine.

        Args:
            timeout: Maximum time to wait for speech.

        Returns:
            Recognized text or None.
        """
        try:
            import tempfile

            import sounddevice as sd
            import soundfile as sf

            # Record audio
            logger.info("Recording audio...")
            sample_rate = 16000
            duration = min(timeout, 10)  # Max 10 seconds
            audio_data = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
            )
            sd.wait()

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                sf.write(tmp_file.name, audio_data, sample_rate)

                # Transcribe with Whisper
                result = self.whisper_model.transcribe(tmp_file.name)
                text = result["text"].strip()

                logger.info(f"Recognized (Whisper): {text}")
                return text if text else None

        except Exception as e:
            logger.error(f"Whisper recognition failed: {e}")
            return None

    async def listen_async(
        self, timeout: float = 5.0, phrase_timeout: float = 1.0
    ) -> Optional[str]:
        """Asynchronously listen for speech.

        Args:
            timeout: Maximum time to wait for speech.
            phrase_timeout: Time to wait for phrase completion.

        Returns:
            Recognized text or None.
        """
        if not self.enabled:
            return None

        # Run STT in a separate thread to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.listen, timeout, phrase_timeout)

    def start_continuous_listening(self, callback: Callable[[str], None]):
        """Start continuous listening in background thread.

        Args:
            callback: Function to call with recognized text.
        """
        if not self.enabled or self.is_listening:
            return

        self.is_listening = True
        self.listen_thread = threading.Thread(
            target=self._continuous_listen_worker, args=(callback,), daemon=True
        )
        self.listen_thread.start()
        logger.info("Started continuous listening")

    def stop_continuous_listening(self):
        """Stop continuous listening."""
        if self.is_listening:
            self.is_listening = False
            if self.listen_thread:
                self.listen_thread.join(timeout=2)
            logger.info("Stopped continuous listening")

    def _continuous_listen_worker(self, callback: Callable[[str], None]):
        """Worker function for continuous listening.

        Args:
            callback: Function to call with recognized text.
        """
        while self.is_listening:
            try:
                text = self.listen(timeout=1.0, phrase_timeout=1.0)
                if text:
                    # Check for trigger word if configured
                    if self.trigger_word:
                        if self.trigger_word in text.lower():
                            # Remove trigger word and process the rest
                            processed_text = (
                                text.lower().replace(self.trigger_word, "").strip()
                            )
                            if processed_text:
                                callback(processed_text)
                    else:
                        callback(text)

            except Exception as e:
                logger.error(f"Continuous listening error: {e}")
                time.sleep(1)  # Brief pause before retrying

    def set_trigger_word(self, trigger_word: str):
        """Set the trigger word for wake-up.

        Args:
            trigger_word: Word that activates processing.
        """
        self.trigger_word = trigger_word.lower() if trigger_word else None
        logger.info(f"Trigger word set to: {self.trigger_word}")

    def calibrate_microphone(self, duration: float = 2.0):
        """Calibrate microphone for ambient noise.

        Args:
            duration: Calibration duration in seconds.
        """
        if not self.enabled or self.engine != "speech_recognition":
            return

        try:
            with self.microphone as source:
                logger.info(f"Calibrating microphone for {duration} seconds...")
                self.recognizer.adjust_for_ambient_noise(source, duration=duration)
            logger.info("Microphone calibration complete")
        except Exception as e:
            logger.error(f"Microphone calibration failed: {e}")

    def get_microphone_info(self) -> dict:
        """Get information about available microphones.

        Returns:
            Dictionary with microphone information.
        """
        try:
            import speech_recognition as sr

            mic_list = sr.Microphone.list_microphone_names()
            return {
                "available_microphones": mic_list,
                "default_microphone": (
                    self.microphone.device_index if self.microphone else None
                ),
            }
        except Exception as e:
            logger.error(f"Failed to get microphone info: {e}")
            return {"available_microphones": [], "default_microphone": None}

    def set_microphone(self, device_index: Optional[int] = None):
        """Set the microphone device to use.

        Args:
            device_index: Index of the microphone device.
        """
        if self.engine == "speech_recognition":
            try:
                import speech_recognition as sr

                self.microphone = sr.Microphone(device_index=device_index)
                logger.info(f"Microphone set to device index: {device_index}")
            except Exception as e:
                logger.error(f"Failed to set microphone: {e}")

    def test_microphone(self) -> bool:
        """Test if microphone is working.

        Returns:
            True if microphone test is successful.
        """
        try:
            logger.info("Testing microphone... Please say something.")
            text = self.listen(timeout=3.0)
            if text:
                logger.info(f"Microphone test successful. Heard: {text}")
                return True
            else:
                logger.warning("Microphone test failed - no speech detected")
                return False
        except Exception as e:
            logger.error(f"Microphone test failed: {e}")
            return False

    def __del__(self):
        """Cleanup STT engine."""
        if hasattr(self, "is_listening") and self.is_listening:
            self.stop_continuous_listening()
