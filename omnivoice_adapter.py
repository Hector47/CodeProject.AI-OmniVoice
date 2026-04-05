"""
OmniVoice Text-to-Speech adapter for CodeProject.AI-Server.

Exposes OmniVoice (https://github.com/k2-fsa/OmniVoice) as a CodeProject.AI
module that can synthesise speech via three modes:

  * Voice Cloning   – supply ``text`` + ``ref_audio`` (+ optional ``ref_text``)
  * Voice Design    – supply ``text`` + ``instruct`` (e.g. "female, low pitch")
  * Auto Voice      – supply ``text`` only; the model picks a voice automatically
"""

import base64
import io
import os
import sys
import time

from codeproject_ai_sdk import JSON, LogMethod, ModuleRunner, RequestData


class OmniVoiceTTS_Adapter(ModuleRunner):

    # --------------------------------------------------------------------- #
    # Lifecycle                                                               #
    # --------------------------------------------------------------------- #

    def initialise(self) -> None:
        self.model = None
        self._model_loaded = False

        # Read configuration from environment / modulesettings.json
        self._model_name  = os.getenv("MODEL_NAME",  "k2-fsa/OmniVoice")
        self._num_steps   = int(os.getenv("NUM_STEPS", "32"))
        self._default_speed = float(os.getenv("SPEED", "1.0"))
        self._use_cuda    = os.getenv("USE_CUDA", "True").lower() == "true"

        # Honour GPU availability reported by the server
        if self._use_cuda:
            self._use_cuda = self.system_info.hasTorchCuda

        self.can_use_GPU      = self._use_cuda
        self.inference_device = "GPU" if self._use_cuda else "CPU"

        self.log(LogMethod.Info | LogMethod.Server, {
            "filename": __file__,
            "method": sys._getframe().f_code.co_name,
            "loglevel": "information",
            "message": f"OmniVoice TTS initialising (model={self._model_name}, "
                       f"device={self.inference_device}, steps={self._num_steps})"
        })

        try:
            self._load_model()
        except Exception as ex:
            self.report_error(ex, __file__,
                              f"Failed to load OmniVoice model: {ex}")

    def cleanup(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            self._model_loaded = False

    # --------------------------------------------------------------------- #
    # Request handling                                                        #
    # --------------------------------------------------------------------- #

    def process(self, data: RequestData) -> JSON:
        if data.command == "speak":
            return self._synthesize(data)

        return {"success": False, "error": f"Unknown command: {data.command}"}

    # --------------------------------------------------------------------- #
    # Status / self-test                                                      #
    # --------------------------------------------------------------------- #

    def status(self) -> JSON:
        status = super().status()
        status["modelLoaded"] = self._model_loaded
        status["modelName"]   = self._model_name
        return status

    def selftest(self) -> JSON:
        request_data = RequestData()
        request_data.command = "speak"
        request_data.add_value("text", "Hello, this is an OmniVoice self-test.")

        result = self.process(request_data)
        return {
            "success": result.get("success", False),
            "message": "OmniVoice TTS self-test completed successfully."
                       if result.get("success") else
                       f"Self-test failed: {result.get('error', 'unknown error')}"
        }

    # --------------------------------------------------------------------- #
    # Private helpers                                                         #
    # --------------------------------------------------------------------- #

    def _load_model(self) -> None:
        """Load the OmniVoice model into memory."""
        import torch
        from omnivoice import OmniVoice

        device_map = "cuda:0" if self._use_cuda else "cpu"
        dtype      = torch.float16 if self._use_cuda else torch.float32

        self.log(LogMethod.Info | LogMethod.Server, {
            "filename": __file__,
            "method": "_load_model",
            "loglevel": "information",
            "message": f"Loading OmniVoice model from '{self._model_name}' onto {device_map} …"
        })

        self.model = OmniVoice.from_pretrained(
            self._model_name,
            device_map=device_map,
            dtype=dtype
        )
        self._model_loaded = True

        self.log(LogMethod.Info | LogMethod.Server, {
            "filename": __file__,
            "method": "_load_model",
            "loglevel": "information",
            "message": "OmniVoice model loaded successfully."
        })

    def _synthesize(self, data: RequestData) -> JSON:
        """
        Run TTS inference and return the audio as base64-encoded WAV data.

        Mode selection (evaluated in order):
        1. Voice Cloning  – ``ref_audio`` file is present in the request
        2. Voice Design   – ``instruct`` form value is present
        3. Auto Voice     – neither of the above
        """
        if not self._model_loaded:
            return {"success": False,
                    "error": "OmniVoice model is not yet loaded. Please try again shortly."}

        # ------------------------------------------------------------------ #
        # Parse inputs                                                        #
        # ------------------------------------------------------------------ #
        text = data.get_value("text")
        if not text:
            return {"success": False, "error": "'text' parameter is required."}

        import torchaudio

        instruct  = data.get_value("instruct",  None)
        ref_text  = data.get_value("ref_text",  None)
        speed     = data.get_float("speed", self._default_speed)
        speed     = max(0.5, min(2.0, speed))       # clamp to allowed range

        # Detect whether a reference audio file was uploaded
        ref_audio_bytes: bytes | None = None
        if data.files and len(data.files) > 0:
            ref_audio_bytes = data.get_file_bytes(0)

        # ------------------------------------------------------------------ #
        # Determine synthesis mode                                            #
        # ------------------------------------------------------------------ #
        if ref_audio_bytes:
            mode = "voice_cloning"
        elif instruct:
            mode = "voice_design"
        else:
            mode = "auto_voice"

        self.log(LogMethod.Info | LogMethod.Server, {
            "filename": __file__,
            "method": "_synthesize",
            "loglevel": "information",
            "message": f"Synthesizing speech – mode={mode}, len(text)={len(text)}"
        })

        # ------------------------------------------------------------------ #
        # Inference                                                           #
        # ------------------------------------------------------------------ #
        start_time = time.perf_counter()

        try:
            generate_kwargs: dict = {
                "text":     text,
                "num_step": self._num_steps,
                "speed":    speed,
            }

            if mode == "voice_cloning":
                # Write reference audio to a temporary file because OmniVoice
                # expects a file path, not raw bytes.
                import tempfile
                suffix = ".wav"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(ref_audio_bytes)
                    tmp_path = tmp.name

                try:
                    generate_kwargs["ref_audio"] = tmp_path
                    if ref_text:
                        generate_kwargs["ref_text"] = ref_text
                    audio_tensors = self.model.generate(**generate_kwargs)
                finally:
                    os.unlink(tmp_path)

            elif mode == "voice_design":
                generate_kwargs["instruct"] = instruct
                audio_tensors = self.model.generate(**generate_kwargs)

            else:  # auto_voice
                audio_tensors = self.model.generate(**generate_kwargs)

        except Exception as ex:
            self.report_error(ex, __file__, f"OmniVoice inference error: {ex}")
            return {"success": False, "error": f"Speech synthesis failed: {str(ex)}"}

        inference_ms = int((time.perf_counter() - start_time) * 1000)

        # ------------------------------------------------------------------ #
        # Encode output audio as base64 WAV                                  #
        # ------------------------------------------------------------------ #
        try:
            # audio_tensors is a list of Tensors with shape (1, T) at 24 kHz
            audio_tensor = audio_tensors[0]  # first (and usually only) result

            buf = io.BytesIO()
            torchaudio.save(buf, audio_tensor, sample_rate=24000, format="wav")
            buf.seek(0)
            audio_b64 = base64.b64encode(buf.read()).decode("ascii")
        except Exception as ex:
            self.report_error(ex, __file__, f"Audio encoding error: {ex}")
            return {"success": False, "error": f"Failed to encode audio output: {str(ex)}"}

        process_ms = int((time.perf_counter() - start_time) * 1000)

        return {
            "success":     True,
            "audio":       audio_b64,
            "mode":        mode,
            "inferenceMs": inference_ms,
            "processMs":   process_ms,
            "message":     f"Speech synthesized successfully ({mode})."
        }


if __name__ == "__main__":
    OmniVoiceTTS_Adapter().start_loop()
