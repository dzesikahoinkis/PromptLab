from openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key= OPENAI_API_KEY)

def ask_chatgpt(prompt, model="gpt-4", temperature=0.7, max_tokens=500, system_message="You are a helpful assistant."):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content


df = pd.read_csv('../data/czyste-powietrze_FAQ.csv')
df = df.iloc[1:6, ]

df['step1-response'] = '' 

for row_idx, row in df.iterrows():
    response = ask_chatgpt(
        prompt=row['question'],  
        temperature=0.2,
        system_message="Jesteś urzędnikiem państwowym i twoim zadaniem jest odpowiadanie na pytania dotyczące programu Czyste powietrze"
    )
    df.at[row_idx, 'step1-response'] = response



df.to_csv(
    'step1-results.csv',
    index=False,
    quoting=1, 
    quotechar='"',
    escapechar='\\'
)