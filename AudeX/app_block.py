from typing import List

import pytesseract
from PIL import Image

import gradio as gr

def tesseract_ocr(filepath: str, languages: List[str]=None):
    image = Image.open(filepath)
    return pytesseract.image_to_string(image=image, lang=', '.join(languages) if languages else None)

with gr.Blocks(title=title) as demo:
    gr.Markdown(f'<h1 style="text-align: center; margin-bottom: 1rem;">{title}</h1>')
    gr.Markdown(description)
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="filepath", label="Input")
            language_choices = pytesseract.get_languages()
            with gr.Accordion("Languages", open=False):
                languages = gr.CheckboxGroup(language_choices, type="value", value=["eng"], label='language')
            with gr.Row():
                btn_clear = gr.ClearButton([image, languages])
                btn_submit = gr.Button(value="Submit", variant="primary")
        with gr.Column():
            text = gr.Textbox(label="Output")

    btn_submit.click(tesseract_ocr, inputs=[image, languages], outputs=text, api_name="tesseract-ocr")
    btn_clear.add(text)


    gr.Markdown(article)

if __name__ == '__main__':
    demo.launch()