import torchaudio as ta
import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
elif torch.xpu.is_available():
    device = "xpu"
else:
    device = "cpu"

# Load the Nano model (same class as Turbo, selected with nano=True).
# Also runs on CPU: device="cpu"
model = ChatterboxTurboTTS.from_pretrained(device=device, nano=True)

# Generate with Paralinguistic Tags
text = "Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. It's the SkyNet T-800 series and it's got basically everything. Including AI integration with ChatGPT and all that jazz. Would you like me to get some prices for you?"

# Generate audio (requires a reference clip for voice cloning)
# wav = model.generate(text, audio_prompt_path="your_10s_ref_clip.wav")
wav = model.generate(text)
ta.save("test-nano.wav", wav, model.sr)
