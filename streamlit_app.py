import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import tempfile

# --- CORE FUNCTIONS FROM YOUR SCRIPT ---

def extract_grade_from_pdf_title(pdf_path):
    with fitz.open(pdf_path) as doc:
        for i in range(min(5, len(doc))):
            page_text = doc[i].get_text().upper()
            match = re.search(r'GRADE\s+(\d+)', page_text)
            if match:
                return match.group(1)
    return None

def load_gpf_data(csv_file):
    df = pd.read_csv(csv_file)
    grade_1_col = df.columns[8]  # 0-indexed
    matched_rows = [
        {
            'Knowledge or Skill': str(row['Knowledge or Skill']).strip(),
            'Grade': '1'
        }
        for _, row in df.iterrows()
        if str(row.get(grade_1_col, '')).strip().lower() == 'x'
    ]
    return pd.DataFrame(matched_rows)

def extract_kenyan_outcomes(pdf_path):
    outcomes = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            blocks = page.get_text("blocks")
            for block in blocks:
                block_text = block[4].strip()
                if "specific lesson learning outcome" in block_text.lower():
                    match = re.search(r"Specific Lesson Learning Outcome\s*(.*)", block_text, re.IGNORECASE | re.DOTALL)
                    if match:
                        outcomes.append(match.group(1).strip())
    return outcomes

def embed_sentences(sentences, model):
    return model.encode(sentences, convert_to_tensor=True)

def match_outcomes(gpf_df, kenyan_outcomes, model, threshold=0.5):
    gpf_df = gpf_df[gpf_df['Knowledge or Skill'].str.strip() != ""].reset_index(drop=True)
    gpf_sentences = gpf_df['Knowledge or Skill'].tolist()
    gpf_embeddings = embed_sentences(gpf_sentences, model)
    kenyan_embeddings = embed_sentences(kenyan_outcomes, model)

    matches = []
    matched_count_per_grade = {}

    for i, gpf_row in gpf_df.iterrows():
        gpf_text = gpf_row['Knowledge or Skill']
        grade = gpf_row['Grade']
        gpf_emb = gpf_embeddings[i]

        similarities = cosine_similarity([gpf_emb], kenyan_embeddings)[0]
        max_sim = max(similarities)
        best_match_index = similarities.argmax()

        matched = max_sim >= threshold
        if matched:
            matched_count_per_grade[grade] = matched_count_per_grade.get(grade, 0) + 1

        matches.append({
            'Grade': grade,
            'GPF Outcome': gpf_text,
            'Best Kenyan Match': kenyan_outcomes[best_match_index],
            'Similarity': round(float(max_sim), 2),
            'Matched': matched
        })

    return pd.DataFrame(matches), matched_count_per_grade

# --- STREAMLIT UI ---
st.set_page_config(page_title="GPF-Kenyan Curriculum Matcher", layout="wide")
st.title("📚 GPF & Kenyan Curriculum Matching Tool")

# Upload GPF CSV
st.subheader("1. Upload Global Proficiency Framework CSV")
gpf_file = st.file_uploader("Choose the GPF .csv file", type="csv")

if gpf_file:
    gpf_df = load_gpf_data(gpf_file)
    st.dataframe(gpf_df, use_container_width=True)

    # Upload Kenyan Curriculum PDF
    st.subheader("2. Upload Kenyan Curriculum PDF")
    pdf_file = st.file_uploader("Choose the Kenyan curriculum .pdf file", type="pdf")

    if pdf_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            tmp_pdf.write(pdf_file.read())
            tmp_pdf_path = tmp_pdf.name

        st.info("Analyzing PDF... Please wait")

        grade = extract_grade_from_pdf_title(tmp_pdf_path)
        gpf_filtered = gpf_df[gpf_df['Grade'] == grade].reset_index(drop=True)
        kenyan_outcomes = extract_kenyan_outcomes(tmp_pdf_path)

        model = SentenceTransformer("all-MiniLM-L6-v2")
        match_df, _ = match_outcomes(gpf_filtered, kenyan_outcomes, model)

        st.success(f"✅ Matching complete for Grade {grade}. Showing {len(match_df)} outcomes.")
        st.dataframe(match_df, use_container_width=True)

        # Download link
        st.download_button(
            label="📥 Download Matched Outcomes as Excel",
            data=match_df.to_excel(index=False, engine='openpyxl'),
            file_name="matched_outcomes.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
