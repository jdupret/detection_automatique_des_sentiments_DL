import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os

# Chargement des modèles préentrainés
processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-french")
model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-french")

# Définition d'une fonction de transcription des audio 
def audio_transcrit(audio_path):
    """
    Transcrit un fichier audio en texte et le sauvegarde dans `transcriptions/`.
    Retourne la transcription.
    """
    # Chargement de l’audio
    speech, sr = librosa.load(audio_path, sr=16000)
    input_values = processor(speech, sampling_rate=sr, return_tensors="pt").input_values

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0]).lower()

    # Sauvegarde de la transcription dans un fichier texte
    doc_transcription = "transcriptions"
    os.makedirs(doc_transcription, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_txt = os.path.join(doc_transcription, base_name + "_transcription.txt")

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(transcription)

    print(f"✅ Transcription sauvegardée dans : {output_txt}")
    return transcription