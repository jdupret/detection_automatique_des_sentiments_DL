import gradio as gr
import sys
import os

# Chemin vers le pipeline et appel des fonction
sys.path.append(os.path.abspath("pipeline"))

from transcription import audio_transcrit
from text_analysis import analyse_text

# === Pipeline ===
def process_pipeline(audio_path):
    transcription = audio_transcrit(audio_path)
    sentiment, confidence = analyse_text(transcription)
    sentiment_str = f"{sentiment} (confiance : {confidence:.2f})"
    return transcription, sentiment_str

# === Gradio ===
with gr.Blocks() as demo:
    gr.Markdown(
        "# 🎙️ Détection Automatique de Sentiment dans des Appels Vocaux à l'aide de Wav2Vec 2.0 pour la transcription et BERT pour l’analyse de sentiment"
    )
    gr.Markdown("Téléversez un fichier audio en francais (.wav), pour obtenir une transcription détecter le sentiment.")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="🎧 Audio (.wav)")
            btn = gr.Button("Analyser")
        with gr.Column():
            transcription_output = gr.Textbox(label="📝 Transcription")
            sentiment_output = gr.Textbox(label="📊 Sentiment détecté")

    btn.click(
        fn=process_pipeline,
        inputs=audio_input,
        outputs=[transcription_output, sentiment_output]
    )

# === Lancement ===
if __name__ == "__main__":
    demo.launch()