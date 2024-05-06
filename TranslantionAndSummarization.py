from transformers import pipeline
import torch
import gc

def translate():
    translator = pipeline(task="translation", model="facebook/nllb-200-distilled-600M", torch_dtype=torch.bfloat16)
    text = "Mum and I climbed in a fort, Monster was scared I thought."
    translated_text = translator(text, src_lang="eng_Latn", tgt_lang="spa_Latn")
    print(translated_text)
    gc.collect()
    del translator
    gc.collect()


def summarization():
    summarizer = pipeline(task="summarization", model="facebook/bart-large-cnn", torch_dtype=torch.bfloat16)
    text = """
    There"s a MONSTER!!!
    Mum and I went out one day... I"m sure my monster followed us there.
    She thought I didn"t notice, but I was quite aware.
    Mum and I soared on the swings, Monster hid behind other things.
    Mum and I whizzed down the slide, Monster looked for somewhere to hide.
    Mum and I climbed in a fort, Monster was scared I thought.
    But she was always by my side, swings, forts, or slides.
    Some big bullies came out to play... Monster wanted to come out, but I said stay!
    Mum said "Yay Ignore bullies, walk away, or just say "Hey." "
    We went for a stroll along by the sea, Monster was scarce as scarce can be.
    Mum and I got yummy ice cream sundaes, Monster was afraid of these type of fun days.
    We romped, played, laughed, had fun in the sun, while Monster stayed back and watched everyone.
    My monster was very very afraid it was clear, not like home where she made chaos everywhere.
    PSST!! 
    But now it"s time to go home, lucky she wasn"t seen Imagine the commotion a monster would have been?
    33Back at home it was time for tea, just my Monster, my Monster and me.
    Would you like some tea too? 
    There are monsters everywhere in our imagination. The trick is to fnd their friendly side, while working with them on the bad habits. Do you have monsters in your house?
    """
    summary = summarizer(text, min_length=10, max_length=150)
    print(summary)
    gc.collect()
    del summarizer
    gc.collect()

translate()
summarization()
