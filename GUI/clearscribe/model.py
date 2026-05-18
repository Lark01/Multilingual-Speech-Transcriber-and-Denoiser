"""
TranscriberModel — loads openai/whisper-large-v3-turbo with a LoRA adapter
and exposes a single transcribe(audio_chunk, sr) method.
"""
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel


class TranscriberModel:
    def __init__(self, adapter_path: str, device: str = "cpu", language: str = None):
        self.device   = torch.device(device)
        self.language = language

        base_name = "openai/whisper-large-v3-turbo"
        print(f"[model] loading base model {base_name} …")
        self.processor = WhisperProcessor.from_pretrained(base_name)
        base_model     = WhisperForConditionalGeneration.from_pretrained(
            base_name, torch_dtype=torch.float32
        )

        print(f"[model] applying LoRA adapter from {adapter_path} …")
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"[model] ready on {device}")

    # ------------------------------------------------------------------
    def transcribe(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> str:
        """
        audio_chunk : float32 numpy array, shape (samples,), normalised [-1, 1]
        sample_rate : must be 16 000 Hz for Whisper
        returns     : transcribed text string (empty string on silence)
        """
        if len(audio_chunk) == 0:
            return ""

        # Whisper's feature extractor expects 16 kHz float32 mono
        inputs = self.processor(
            audio_chunk,
            sampling_rate=sample_rate,
            return_tensors="pt"
        )
        input_features = inputs.input_features.to(self.device)

        # Build forced-decoder kwargs
        gen_kwargs = dict(
            input_features=input_features,
            max_new_tokens=128,
        )
        if self.language:
            forced_ids = self.processor.get_decoder_prompt_ids(
                language=self.language, task="transcribe"
            )
            gen_kwargs["forced_decoder_ids"] = forced_ids

        with torch.no_grad():
            predicted_ids = self.model.generate(**gen_kwargs)

        text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return text[0].strip() if text else ""
