import pandas as pd
import html, re, os
from tqdm import tqdm
tqdm.pandas()

RAW = 'D:/NIENLUAN/drug-recommender/data/raw/drugsComTrain_clean.csv'
PROCESSED = 'data/processed/drug_reviews_clean.csv'

def clean_html_entities(text):
    if pd.isna(text):
        return ''
    t = html.unescape(str(text))
    t = re.sub(r'[\r\n]+', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def main():
    os.makedirs('data/processed', exist_ok=True)
    df = pd.read_csv(RAW, dtype=str, on_bad_lines='skip')
    # Ensure expected columns
    for col in ['uniqueID','drugName','condition','review','rating','date','usefulCount']:
        if col not in df.columns:
            df[col] = None
    df['review'] = df['review'].progress_apply(clean_html_entities)
    df['drugName'] = df['drugName'].fillna('Unknown').str.strip()
    df['condition'] = df['condition'].fillna('Unknown').str.strip()
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0).astype(int)
    df.to_csv(PROCESSED, index=False)
    print(f"Processed saved to {PROCESSED}")

if __name__ == '__main__':
    main()
