from typing import List
import pytesseract
from PIL import Image
import re
import gradio as gr
from transformers import AutoProcessor, AutoModelForTextToSpectrogram, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf

def tesseract_ocr(filepath: str) -> str:
    image = Image.open(filepath)
    combined_languages = 'eng+hin'
    extracted_text = pytesseract.image_to_string(image=image, lang=combined_languages)
    return extracted_text

def search_and_highlight(text: str, keyword: str) -> str:
    if keyword:
        highlighted_text = re.sub(f"({keyword})", r"<mark>\1</mark>", text, flags=re.IGNORECASE)
        return highlighted_text
    return text

def ocr_and_tts(filepath: str, keyword: str) -> (str, str):
    if filepath is None:
        return "Please upload an image.", None

    # OCR and keyword highlighting
    extracted_text = tesseract_ocr(filepath)
    highlighted_text = search_and_highlight(extracted_text, keyword)

    # Convert text to speech
    audio_path = text_to_speech(extracted_text)
    
    return highlighted_text, audio_path

# Load model
processor = AutoProcessor.from_pretrained("Aumkeshchy2003/speecht5_finetuned_Aumkesh_English_tts")
model = AutoModelForTextToSpectrogram.from_pretrained("Aumkeshchy2003/speecht5_finetuned_Aumkesh_English_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Load speaker embedding
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# Move models to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
vocoder = vocoder.to(device)
speaker_embeddings = speaker_embeddings.to(device)

@torch.inference_mode()
def text_to_speech(text: str) -> str:
    inputs = processor(text=text, return_tensors="pt").to(device)
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    output_path = "output.wav"
    sf.write(output_path, speech.cpu().numpy(), samplerate=16000)
    return output_path

demo = gr.Interface(
    fn=ocr_and_tts,
    inputs=[
        gr.Image(type="filepath", label="Upload Image for OCR"),
        gr.Textbox(label="Keyword to Highlight", placeholder="Enter a keyword...")
    ],
    outputs=[
        gr.HTML(label="Extracted and Highlighted Text"),
        gr.Audio(label="Generated Speech")
    ],
    title="OCR to TTS",
    description="Upload an image for OCR. The extracted text will be highlighted if a keyword is provided and the whole text will be converted to speech."
)

if __name__ == "__main__":
    demo.launch()
    