from fastapi import FastAPI, UploadFile, File
import shutil
import os
import uuid
import sys

# pour accéder aux modules pipeline
sys.path.append(os.path.abspath("pipeline"))

from transcription import transcribe_from_file
from text_analysis import analyze_sentiment_from_text

app = FastAPI(
    title="Détection de Sentiment dans un Appel Vocal",
    description="Il s'agit d'un API qui permet de transcrire un fichier audio et prédire le sentiment du client (négatif, neutre, positif).",
    version="1.0"
)

@app.post("/analyse/")
async def analyse_audio(file: UploadFile = File(...)):
    # sauvegarder temporairement le fichier reçu
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_filename = os.path.join(temp_dir, f"{uuid.uuid4()}.wav")

    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # pipeline
    transcription = transcribe_from_file(temp_filename)
    sentiment, confidence = analyze_sentiment_from_text(transcription)

    # nettoyer
    os.remove(temp_filename)

    return {
        "transcription": transcription,
        "sentiment": sentiment,
        "confiance": round(confidence, 2)
    }