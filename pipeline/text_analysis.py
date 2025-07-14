from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import sys
import os

# Chargement des modèles
model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

labels = ["négatif", "neutre", "positif"]

def analyse_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)

    scores = torch.softmax(outputs.logits, dim=1).squeeze()
    predicted_idx = torch.argmax(scores).item()
    predicted_label = labels[predicted_idx]
    confidence = scores[predicted_idx].item()

    return predicted_label, confidence

# Analyse du sentiment de la transcription
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python sentiment_analyzer.py <transcriptions/transcription.txt>")
        sys.exit(1)

    txt_file = sys.argv[1]

    if not os.path.isfile(txt_file):
        print(f"Le fichier {txt_file} n'existe pas.")
        sys.exit(1)

    # Lire le texte
    with open(txt_file, "r", encoding="utf-8") as f:
        text = f.read().strip()

    sentiment, confidence = analyze_sentiment_from_text(text)
    print(f"📝 Texte analysé : {text}")
    print(f"📊 Sentiment : {sentiment} (confiance : {confidence:.2f})")