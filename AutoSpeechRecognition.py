from datasets import load_dataset
from transformers import pipeline
import sounddevice as sd
import soundfile as sf
import io
import numpy as np
import librosa

# By default this will transcribe audio of up to 30 seconds in duration (model will truncate everything beyond 30s)
asr = pipeline("automatic-speech-recognition",
               model="distil-whisper/distil-medium.en",
               trust_remote_code=True)


def speech_recognition(play_sound: bool):
    # Load the audio dataset via streaming
    dataset = load_dataset("librispeech_asr",
                           split="train.clean.100",
                           streaming=True,
                           trust_remote_code=True)
    #  Get the first item from the dataset
    example = next(iter(dataset))
    print(example)
    # To get the first five items from the dataset, you could iter 5 time or just use take(5)
    first_five = list(dataset.take(5))
    # Print the third item from the dataset
    print(first_five[2])
    if play_sound:
        sd.play(example["audio"]["array"], example["audio"]["sampling_rate"])
        sd.wait()

    print(asr.feature_extractor.sampling_rate)
    print(example['audio']['sampling_rate'])
    result = asr(example["audio"]["array"])
    print(f"Transcribed text:{result['text']}")
    print(f'Source text: {example["text"]}')


def longer_stereo_file_transcribe(play_sound: bool):
    audio, sampling_rate = sf.read('data/narration_example.wav')

    print(f'Sampling rate: {sampling_rate}')
    print(f"asr sampling rate: {asr.feature_extractor.sampling_rate}")
    print(f"audio shape: {audio.shape}")

    # Convert stereo to mono
    audio_transposed = np.transpose(audio)
    print(f"audio mono shape: {audio_transposed.shape}")
    audio_mono = librosa.to_mono(audio_transposed)

    if play_sound:
        sd.play(audio_mono, sampling_rate)
        sd.wait()

    # Match sampling rate of sound with the dataset of the model
    audio_new_sample_rate = librosa.resample(audio_mono, orig_sr=sampling_rate,
                                             target_sr=asr.feature_extractor.sampling_rate)

    # Process the audio in chunks of 30 seconds
    result = asr(audio_new_sample_rate,
                 chunk_length_s=30,
                 batch_size=4,
                 return_timestamps=True)
    print(result["chunks"])


def run_app():
    play_sound = False
    speech_recognition(play_sound)
    longer_stereo_file_transcribe(play_sound)

run_app()