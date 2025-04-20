import os
import argparse
import zipfile
import requests
import fitz  
from openai import OpenAI
import faiss
import numpy as np
import logging
from tqdm import tqdm
from dotenv import load_dotenv


def load_api_key():
    load_dotenv()
    return os.getenv("OPENAI_API_KEY")

def download_zip(url, save_path):
    response = requests.get(url, verify=False)
    with open(save_path, "wb") as f:
        f.write(response.content)
    logging.info(f"ZIP downloaded to: {save_path}")
    return save_path

def extract_pdfs(zip_path, extract_to):
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    logging.info(f"ZIP extracted to: {extract_to}")

    pdfs = []
    for root, _, files in os.walk(extract_to):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, file))
    return pdfs


def extract_chunks_from_pdfs(pdf_paths, max_chars=1000, overlap=200):
    chunks = []

    for pdf_path in pdf_paths:
        doc = fitz.open(pdf_path)
        full_text = ""

        for page in doc:
            full_text += page.get_text()

        doc.close()

        for i in range(0, len(full_text), max_chars - overlap):
            chunk = full_text[i:i + max_chars]
            if len(chunk.strip()) > 100:
                chunks.append(chunk)

    logging.info(f"Extracted {len(chunks)} text chunks.")
    return chunks


def embed_text(text, client):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding


def build_faiss_index(chunks, client):
    """Build FAISS index from text chunks."""
    dimension = 1536  # Embedding size for ada-002
    index = faiss.IndexFlatL2(dimension)
    metadata = []

    for chunk in tqdm(chunks, desc="Embedding chunks"):
        try:
            emb = embed_text(chunk, client)
            index.add(np.array([emb], dtype='float32'))
            metadata.append(chunk)
        except Exception as e:
            logging.warning(f"Error embedding chunk: {e}")

    logging.info("FAISS index built.")
    return index, metadata


def save_index(index, metadata, folder):
    """Save FAISS index and metadata to disk."""
    os.makedirs(folder, exist_ok=True)

    faiss.write_index(index, os.path.join(folder, "index.faiss"))

    with open(os.path.join(folder, "chunks.txt"), "w", encoding="utf-8") as f:
        for chunk in metadata:
            f.write(chunk.replace("\n", " ") + "\n---\n")

    logging.info(f"Index and metadata saved to: {folder}")


def main(zip_url, base_path):
    """Main processing pipeline."""
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    api_key = load_api_key()
    if not api_key:
        logging.error("OPENAI_API_KEY not found in .env file.")
        return
    client = OpenAI(api_key=api_key)

    base_path   = os.path.abspath(base_path)
    zip_path    = f"{base_path}.zip"
    extract_dir = f"{base_path}_pdf"
    faiss_dir   = f"{base_path}_faiss-store"

    logging.info(f"Base path: {base_path}")
    logging.info(f"Downloading ZIP from: {zip_url}")

    # Execute pipeline
    download_zip(zip_url, zip_path)
    pdf_files = extract_pdfs(zip_path, extract_dir)
    pdf_chunks = extract_chunks_from_pdfs(pdf_files)
    faiss_index, chunk_texts = build_faiss_index(pdf_chunks, client)
    save_index(faiss_index, chunk_texts, faiss_dir)

    logging.info("Done. All files saved and indexed successfully.")


# === CLI ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download, extract, chunk, embed and index PDF documents for semantic search."
    )

    parser.add_argument(
        "-u", "--url",
        required=True,
        help="URL to the ZIP file containing PDF documents."
    )

    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Base path for saving ZIP, extracted PDFs, and FAISS index (no extension)."
    )

    args = parser.parse_args()
    main(args.url, args.output)
