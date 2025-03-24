import torch
import torchaudio
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

# Example Script for Inference

# Load model
model_name = "mispeech/r1-aqa"
processor = AutoProcessor.from_pretrained(model_name)
model = Qwen2AudioForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

# Load example audio
wav_path = "data/MMAU/test-mini-audios/3fe64f3d-282c-4bc8-a753-68f8f6c35652.wav"  # from MMAU dataset
waveform, sampling_rate = torchaudio.load(wav_path)
if sampling_rate != 16000:
    waveform = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)(waveform)
audios = [waveform[0].numpy()]

# Make prompt text
question = "Based on the given audio, identify the source of the speaking voice."
options = ["Man", "Woman", "Child", "Robot"]
prompt = f"{question} Please choose the answer from the following options: {str(options)}. Output the final answer in <answer> </answer>."
message = [
    {"role": "user", "content": [
        {"type": "audio", "audio_url": wav_path},
        {"type": "text", "text": prompt}
    ]}
]
texts = processor.apply_chat_template(message, add_generation_prompt=True, tokenize=False)

# Process
inputs = processor(text=texts, audios=audios, sampling_rate=16000, return_tensors="pt", padding=True).to(model.device)
generated_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids = generated_ids[:, inputs.input_ids.size(1):]
response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

print(response)