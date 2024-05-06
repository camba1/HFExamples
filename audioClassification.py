from transformers import pipeline
from datasets import load_dataset, Audio
# import soundfile
# import librosa
from datasets import load_dataset



def audio_classification():
    # We use a clap feature extraction model for zero shot classification
    zero_shot_classifier = pipeline("zero-shot-audio-classification", model="laion/clap-htsat-unfused")
    dataset = load_dataset("ashraq/esc50", split="train")
    audio_sample = dataset[0]
    print(audio_sample)
    print(f"Default sampling rate: {zero_shot_classifier.feature_extractor.sampling_rate}")
    print(f"First sample rate: {audio_sample['audio']['sampling_rate']}")

    #  update sampling rate on dataset to match feature extractor
    dataset = dataset.cast_column("audio",  Audio(sampling_rate=zero_shot_classifier.feature_extractor.sampling_rate))

    # Clap takes an audio input and a set of labels and then computes the similarity of them
    # It the does a
    candidate_labels = ["Sound of a dog",
                        "Sound of vacuum cleaner"]
    result = zero_shot_classifier(audio_sample["audio"]["array"], candidate_labels=candidate_labels)
    print(result)



audio_classification()

