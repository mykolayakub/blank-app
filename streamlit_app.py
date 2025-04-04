import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import re
import tempfile
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px


# Detect grade level from the PDF title (optional)
def extract_grade_from_pdf_title(pdf_path):
    with fitz.open(pdf_path) as doc:
        for i in range(min(5, len(doc))):
            page_text = doc[i].get_text().upper()
            match = re.search(r'GRADE\s+(\d+)', page_text)
            if match:
                return match.group(1)
    return None

# Load GPF data and organize it
# Load GPF data and organize it
def load_gpf_data(csv_file):
    df = pd.read_csv(csv_file)
    
    # Strip leading/trailing spaces from column names
    df.columns = df.columns.str.strip()

    # Forward-fill hierarchical columns to handle blank cells
    df[['Domain', 'Construct', 'Subconstruct']] = df[['Domain', 'Construct', 'Subconstruct']].fillna(method='ffill')

    # Validate columns
    if "Domain" not in df.columns:
        raise ValueError("Missing 'Domain' column!")
    if "Knowledge or Skill" not in df.columns:
        raise ValueError("Missing 'Knowledge or Skill' column!")
    
    # Organize data into hierarchy (existing code)
    hierarchy = {}
    for _, row in df.iterrows():
        domain = row['Domain']
        construct = row['Construct']
        subconstruct = row['Subconstruct']
        knowledge_or_skill = row['Knowledge or Skill']
        
        if domain not in hierarchy:
            hierarchy[domain] = {}
        if construct not in hierarchy[domain]:
            hierarchy[domain][construct] = {}
        if subconstruct not in hierarchy[domain][construct]:
            hierarchy[domain][construct][subconstruct] = []
        
        hierarchy[domain][construct][subconstruct].append(knowledge_or_skill)
    
    return df, hierarchy

def split_label_into_two_lines(text, max_length=40):
    """Split text into two lines if it exceeds max_length characters."""
    if len(text) > max_length:
        midpoint = len(text) // 2
        # Find the nearest space to split
        split_index = text.rfind(' ', 0, midpoint)
        if split_index == -1:
            split_index = midpoint
        return f"{text[:split_index]}<br>{text[split_index:].lstrip()}"
    return text

# Extract learning outcomes from the Kenyan PDF
def extract_kenyan_outcomes(pdf_path):
    sentences = []
    sentence_endings = re.compile(r'(?<=[.!?;:])\s+(?=[A-Z"“])')  # Handle punctuation like :, ;, ., ?, etc.
    abbreviation_pattern = re.compile(r'\b(Mr|Mrs|Ms|Dr|Prof|St|Ave|etc|e\.g|i\.e)\.$', re.IGNORECASE)

    with fitz.open(pdf_path) as doc:
        for page in doc:
            text = page.get_text()
            text = re.sub(r'\n+', ' ', text)  # Replace newlines with spaces
            text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces

            candidates = sentence_endings.split(text)

            merged = []
            buffer = ''
            for candidate in candidates:
                if buffer:
                    candidate = buffer + ' ' + candidate
                    buffer = ''
                
                # Check if ends with potential abbreviation
                if abbreviation_pattern.search(candidate):
                    buffer = candidate
                else:
                    merged.append(candidate)

            if buffer:
                merged.append(buffer)

            # Filter and clean sentences
            for sentence in merged:
                cleaned = sentence.strip()
                if len(cleaned) > 15:  # Filter short fragments
                    sentences.append(cleaned)

    return sentences


# Embed sentences
def embed_sentences(sentences, model):
    return model.encode(sentences, convert_to_tensor=True)

# Match outcomes
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
            'Domain': gpf_row['Domain'],  # <-- Add Domain
            'Construct': gpf_row['Construct'],  # <-- Add Construct
            'Subconstruct': gpf_row['Subconstruct'],  # <-- Add Subconstruct
            'Grade': grade,
            'GPF Outcome': gpf_text,
            'Best Match': kenyan_outcomes[best_match_index],
            'Similarity': round(float(max_sim), 2),
            'Matched': matched
        })

    return pd.DataFrame(matches), matched_count_per_grade


# -------------------- STREAMLIT APP --------------------

st.set_page_config(page_title="GPF-Curriculum Matcher", layout="wide")
st.title("📚 GPF & Curriculum Matching Tool")

# Step 1: Upload CSV
st.subheader("1. Upload Global Proficiency Framework CSV")
gpf_file = st.file_uploader("Upload the GPF .csv file", type="csv")

if gpf_file:
    gpf_df, gpf_hierarchy = load_gpf_data(gpf_file)

    # Step 2: Grade Selector
    st.subheader("2. Select Grade to Filter")
    selected_grade = st.selectbox("Select Grade", [str(i) for i in range(1, 10)])

    grade_col_index = 8 + (int(selected_grade) - 1)
    grade_col_name = gpf_df.columns[grade_col_index]

    filtered = []
    # Inside the loop where you populate the 'filtered' list:
    for _, row in gpf_df.iterrows():
        if str(row[grade_col_name]).strip().lower() == 'x':
            filtered.append({
                'Domain': row['Domain'],  # Add Domain
                'Construct': row['Construct'],  # Add Construct
                'Subconstruct': row['Subconstruct'],  # Add Subconstruct
                'Knowledge or Skill': str(row['Knowledge or Skill']).strip(),
                'Grade': selected_grade
            })
    gpf_filtered_df = pd.DataFrame(filtered)

    st.success(f"✅ Found {len(gpf_filtered_df)} outcomes for Grade {selected_grade}")
    st.dataframe(gpf_filtered_df, use_container_width=True)

    # Step 3: Upload PDF
    st.subheader("3. Upload Curriculum PDF")
    pdf_file = st.file_uploader("Upload the curriculum PDF", type="pdf")

    if pdf_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            tmp_pdf.write(pdf_file.read())
            tmp_pdf_path = tmp_pdf.name

        st.info("🔍 Extracting and matching outcomes...")

        kenyan_outcomes = extract_kenyan_outcomes(tmp_pdf_path)
        model = SentenceTransformer("all-MiniLM-L6-v2")

        match_df, matched_count_per_grade = match_outcomes(gpf_filtered_df, kenyan_outcomes, model)

        st.success(f"🎯 Matching complete. Showing {len(match_df)} matched outcomes.")
        st.dataframe(match_df, use_container_width=True)

        # Visualization: Sunburst chart
        # Visualization: Sunburst chart
        st.subheader("4. Visualization: Sunburst of Matched Outcomes")

        # Create hierarchical structure for Sunburst
        data = []
        for index, row in match_df.iterrows():
            if row['Matched']:
                gpf_outcome = split_label_into_two_lines(row['GPF Outcome'])
                best_match = split_label_into_two_lines(row['Best Match'])
                data.append({
                    "Domain": row['Domain'],
                    "Construct": row['Construct'],
                    "Subconstruct": row['Subconstruct'],
                    "Knowledge or Skill": gpf_outcome,  # Modified with line breaks
                    "Best Match": best_match,  # Modified with line breaks
                    "Similarity": row['Similarity']
                })

        sunburst_df = pd.DataFrame(data)

        # Plot Sunburst with correct hierarchy
        fig = px.sunburst(
            sunburst_df,
            path=['Domain', 'Construct', 'Subconstruct', 'Knowledge or Skill', 'Best Match'], 
            values='Similarity',
            color='Similarity',
            hover_data=['Similarity']
        )

        fig.update_layout(
            title="Sunburst Chart of Matched Outcomes",
            margin={"t": 0, "b": 0},
            height=800,  # Increase chart size
            width=1200
        )
        fig.update_traces(
            textfont_size=14,  # Increase font size
            insidetextorientation='horizontal'  # Keep text horizontal
        )
        st.plotly_chart(fig)
