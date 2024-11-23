from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_audio_with_pydub(file):
    audio = AudioSegment.from_file(file)
    audio = audio.set_frame_rate(16000).set_channels(1)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    return torch.tensor(samples), 16000

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    try:
        waveform, sample_rate = load_audio_with_pydub(file.file)
        input_values = processor(waveform, sampling_rate=sample_rate, return_tensors="pt").input_values
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        return {"text": transcription}
    except Exception as e:
        return {"error": str(e)}
