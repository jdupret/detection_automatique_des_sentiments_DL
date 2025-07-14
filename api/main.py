from fastapi import FastAPI, UploadFile, File
import shutil
import os
import uuid
import sys

# Chemin vers le pipeline et appel des fonction
sys.path.append(os.path.abspath("pipeline"))
from transcription import audio_transcrit
from text_analysis import analyse_text

# Titre et description de l'API
app = FastAPI(
    title="Détection de Sentiment dans un Appel Vocal",
    description="Il s'agit d'un API qui permet de transcrire un fichier audio et prédire le sentiment du client (négatif, neutre, positif).",
    version="1.0"
)
# Définition des composante du pipeline
@app.post("/analyse/")
async def analyse_audio(file: UploadFile = File(...)):
    # sauvegarde du fichier charger
    fichier_emp = "fichier_charge"
    os.makedirs(fichier_emp, exist_ok=True)
    temp_filename = os.path.join(fichier_emp, f"{uuid.uuid4()}.wav")

    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # pipeline
    transcription = audio_transcrit(temp_filename)
    sentiment, confidence = analyse_text(transcription)

    # nettoyer
    os.remove(temp_filename)

    return {
        "transcription": transcription,
        "sentiment": sentiment,
        "confiance": round(confidence, 2)
    }