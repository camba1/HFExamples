import os
import gradio as gr
from transformers import pipeline

demo = gr.Blocks()


def transcribe(filepath):
    # By default this will transcribe audio of up to 30 seconds in duration (model will truncate everything beyond 30s)
    asr = pipeline("automatic-speech-recognition",
                   model="distil-whisper/distil-medium.en",
                   trust_remote_code=True)
    if filepath is None:
        gr.Warning("No audio file found, please retry!")
        return
    output = asr(filepath)
    return output["text"]


mic_transcribe = gr.Interface(fn=transcribe, inputs=gr.Audio(sources="microphone", type="filepath"),
                              outputs=gr.Textbox(label="Transcript", lines=3), title="Gradio Speech Recognition",
                              allow_flagging="never")
file_transcribe = gr.Interface(fn=transcribe, inputs=gr.Audio(sources="upload", type="filepath"),
                              outputs=gr.Textbox(label="Transcript", lines=3), title="Gradio Speech Recognition",
                              allow_flagging="never")

with demo:
    gr.TabbedInterface([mic_transcribe, file_transcribe], ["Use your Voice", "Upload a file"])

demo.launch()
