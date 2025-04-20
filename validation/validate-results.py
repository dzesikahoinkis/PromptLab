from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import os
import json
from bert_score import score as bert_score
from bleurt import score as bleurt_score

# Initialize BLEURT once globally
bleurt_model_path = "../BLEURT-20"  
bleurt_scorer = bleurt_score.BleurtScorer(bleurt_model_path)

# Load environment and model
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def validate_with_sentence_transformer(answer: str, response: str) -> float:
    emb1 = model.encode(answer, convert_to_tensor=True)
    emb2 = model.encode(response, convert_to_tensor=True)
    similarity = util.cos_sim(emb1, emb2).item() 
    return similarity


def validate_with_bertscore(answer: str, response: str) -> dict:
    try:
        P, R, F1 = bert_score([response], [answer], lang="pl", verbose=True)
        return {
            "precision": P[0].item(),
            "recall": R[0].item(),
            "f1": F1[0].item()
        }
    except Exception as e:
        print(f"[ERROR] BERTScore failed: {e}")
        return {
            "precision": -1.0,
            "recall": -1.0,
            "f1": -1.0
        }
    
def validate_with_bleurt(answer: str, response: str) -> float:
    try:
        scores = bleurt_scorer.score(references=[answer], candidates=[response])
        return scores[0]
    except Exception as e:
        print(f"[ERROR] BLEURT failed: {e}")
        return -1.0

def validate_with_gpt(answer: str, response: str) -> dict:
    prompt = f"""
Porównaj dwie odpowiedzi dotyczące tego samego pytania.

Odpowiedź oficjalna:
"{answer}"

Odpowiedź modelu:
"{response}"

Oceń odpowiedź modelu na podstawie następujących kryteriów (w skali 0–5, gdzie 0 = bardzo słabo, 5 = doskonale):

1. Zgodność faktów – Czy odpowiedź zawiera poprawne i zgodne z oficjalną odpowiedzią informacje?
2. Pokrycie treści – Czy odpowiedź zawiera wszystkie kluczowe elementy z oficjalnej odpowiedzi?
3. Trafność – Czy odpowiedź odnosi się do tematu i pytania?
4. Zwiezlosc – Czy odpowiedź jest konkretna i nie zawiera zbędnych informacji?
5. Wartosc dodana – Czy zawiera dodatkowe, przydatne informacje, które nie były w oficjalnej odpowiedzi?

Zwróć wynik w formacie JSON:
{{
  "zgodnosc_faktow": int,
  "pokrycie_tresci": int,
  "trafnosc": int,
  "zwiezlosc": int,
  "wartosc_dodana": int,
  "uzasadnienie": "krótkie podsumowanie oceny"
}}
"""
    gpt_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Jesteś ekspertem oceniającym jakość odpowiedzi modelu względem wzorcowej odpowiedzi."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    content = gpt_response.choices[0].message.content
    try:
        result = json.loads(content)
        return {
            "gpt_zgodnosc_faktow": result.get("zgodnosc_faktow"),
            "gpt_pokrycie_tresci": result.get("pokrycie_tresci"),
            "gpt_trafnosc": result.get("trafnosc"),
            "gpt_zwiezlosc": result.get("zwiezlosc"),
            "gpt_wartosc_dodana": result.get("wartosc_dodana"),
            "gpt_uzasadnienie": result.get("uzasadnienie")
        }
    except json.JSONDecodeError:
        return {
            "gpt_zgodnosc_faktow": None,
            "gpt_pokrycie_tresci": None,
            "gpt_trafnosc": None,
            "gpt_zwiezlosc": None,
            "gpt_wartosc_dodana": None,
            "gpt_uzasadnienie": content  # Save raw GPT output for debugging
        }


os.chdir("validation")
df = pd.read_csv('../analysis/step1-results.csv')
df = df.iloc[0:2, ]


for idx, row in df.iterrows():
    answer = row['answer']
    response = row['step1-response']

    # # Validate using sentence transformer
    similarity = validate_with_sentence_transformer(answer, response)
    df.at[idx, 'sentencetr'] = similarity

    # # Validate using GPT
    gpt_result = validate_with_gpt(answer, response)
    for key, value in gpt_result.items():
        df.at[idx, key] = value

    # Validate using BERTScore
    bert_scores = validate_with_bertscore(answer, response)
    df.at[idx, 'bertscore_precision'] = bert_scores['precision']
    df.at[idx, 'bertscore_recall'] = bert_scores['recall']
    df.at[idx, 'bertscore_f1'] = bert_scores['f1']

    # Validate using BLEURT
    bleurt_result = validate_with_bleurt(answer, response)
    df.at[idx, 'bleurt'] = bleurt_result

df.to_csv(
    'step1-validated.csv',
    index=False,
    quoting=1, 
    quotechar='"',
    escapechar='\\'
)

