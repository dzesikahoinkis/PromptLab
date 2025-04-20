import os
import argparse
import logging
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
from openai import OpenAI


def load_api_key():
    load_dotenv()
    return os.getenv("OPENAI_API_KEY")

def ask_chatgpt(prompt, client, model="gpt-4", temperature=0.7, max_tokens=500,
                system_message="You are a helpful assistant."):
    try:
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
    except Exception as e:
        logging.error(f"OpenAI API error: {e}")
        return "ERROR: GPT request failed."


def main(input_path, output_path):
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    # Load API key and create client
    api_key = load_api_key()
    if not api_key:
        logging.error("OPENAI_API_KEY not found in .env file.")
        return
    client = OpenAI(api_key=api_key)

    # Load input CSV
    if not os.path.isfile(input_path):
        logging.error(f"Input file not found: {input_path}")
        return

    df = pd.read_csv(input_path)
    if 'question' not in df.columns:
        logging.error(f"The input CSV must contain a 'question' column.")
        return

    step1_results = pd.DataFrame()

    logging.info(f"Processing {len(df)} questions...")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Asking ChatGPT"):
        # response = ask_chatgpt(
        #     prompt=row['question'],
        #     client=client,
        #     temperature=0.2,
        #     system_message="Jesteś urzędnikiem państwowym i twoim zadaniem jest odpowiadanie na pytania dotyczące programu Czyste powietrze"
        # )
        step1_results.at[idx, 'step1-response'] = "response"

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save results
    step1_results.to_csv(
        output_path,
        index=False,
        quoting=1,       
        quotechar='"',
        escapechar='\\'
    )

    logging.info(f"Saved results to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ask ChatGPT to answer questions from a CSV and save the responses."
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Path to input CSV file (must contain 'question' column)"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Path to save output CSV with 'step1' response"
    )

    args = parser.parse_args()
    main(args.input, args.output)
