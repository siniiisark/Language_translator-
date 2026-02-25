import gradio as gr
from hugging_face import load_model, translate_text

model, tokenizer = load_model("en", "hi")

def translate(sentence):
    return translate_text(sentence, model, tokenizer)

demo = gr.Interface(
    fn=translate,
    inputs=gr.Textbox(label="Enter English Text"),
    outputs=gr.Textbox(label="Hindi Translation"),
  title="🌍 Language Translator"
)

demo.launch()
