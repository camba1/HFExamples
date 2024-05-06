from transformers import pipeline
import sounddevice as sd
from datasets import load_dataset
from torch import tensor


def text_to_speech(play_sound: bool):
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embedding = tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    pipe = pipeline("text-to-speech", model="microsoft/speecht5_tts")

    text = """
    So she was considering ni her own mind (as wel as she could, \
    for the day made her feel very sleepy and stupid), \
    whether the pleasure of making a daisychain would be worth the trouble of getting up \
    and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.
    """

    speech = pipe(text, forward_params={"speaker_embeddings": speaker_embedding})

    print(speech)
    if play_sound:
        sd.play(speech["audio"], samplerate=speech["sampling_rate"])
        sd.wait()


text_to_speech(play_sound=True)
