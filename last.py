import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import tempfile
import os
from pypdf import PdfReader

st.set_page_config(page_title="Company Handbook Assistant", page_icon="📋")

try:
    genai.configure(api_key=st.secrets["AIzaSyA0Tr25hmhf5uB9yShjF-k3j7KdwJ1AqWo"])
    st.sidebar.success("✅ Gemini API Connected")
except Exception as e:
    st.error(f"Failed to configure Gemini API. Did you add it to secrets? Error: {e}")
    st.stop()

if "question" not in st.session_state:
    st.session_state["question"] = ""

st.title("📋 Company Handbook Assistant")
st.markdown("Your 24/7 HR policy guide — powered by AI")

uploaded_file = st.file_uploader("Upload your Company Handbook PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        reader = PdfReader(tmp_path)
        full_text_list = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text_list.append(text)
        full_text = "\n".join(full_text_list)

        def chunk_text(text, chunk_size=500, overlap=50):
            chunks = []
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunks.append(text[start:end])
                start += chunk_size - overlap
            return chunks

        texts = chunk_text(full_text)

        model_emb = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model_emb.encode(texts).astype("float32")
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        st.success(f"✅ PDF processed! {len(texts)} chunks created.")

        st.markdown("💬 Try a common question:")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("How many casual leaves do I get?"):
                st.session_state["question"] = "How many casual leaves do I get?"
                st.rerun()
        with col2:
            if st.button("What is the notice period?"):
                st.session_state["question"] = "What is the notice period?"
                st.rerun()
        with col3:
            if st.button("How do I apply for work from home?"):
                st.session_state["question"] = "How do I apply for work from home?"
                st.rerun()

        question = st.text_input(
            "Ask your HR question:",
            value=st.session_state.get("question", ""),
            placeholder="e.g. How many sick leaves can I take?"
        )

        if st.button("Get Answer"):
            if question.strip() == "":
                st.warning("Please enter a question")
            else:
                q_emb = model_emb.encode([question]).astype("float32")
                distances, indices = index.search(q_emb, k=4)
                context = "\n\n".join([texts[i] for i in indices[0]])

                prompt = f"""
You are HRBot, a warm and employee-friendly Company Handbook Assistant.

Answer ONLY using the context below. If the answer is not found in the context, respond exactly with:
"I couldn't find that in the company handbook. Please contact your HR team directly."

FORMAT your answer exactly like this:
✅ Direct Answer: (1-2 sentences)
📋 Policy Details: (from the handbook, mention section/page if visible)
💡 Next Step: (a helpful tip or action for the employee)

Rules:
- Use plain, friendly language
- Address the employee as "you"
- Never use outside knowledge
- For sensitive topics (harassment, legal, medical) always add:
  "Please reach out to your HR Business Partner directly."

Context:
{context}

Question:
{question}

Answer:
"""

                try:
                    llm = genai.GenerativeModel("gemini-2.5-flash-lite")
                    with st.spinner("HRBot is thinking..."):
                        response = llm.generate_content(prompt)
                        st.subheader("💬 Answer")
                        st.write(response.text)

                    with st.expander("📋 Source Sections Used"):
                        for i, idx in enumerate(indices[0]):
                            st.markdown(f"**Chunk {i+1}** — Score: {distances[0][i]:.2f}")
                            st.info(texts[idx])

                except Exception as e:
                    st.error(f"Gemini Error: {e}")

    finally:
        os.unlink(tmp_path)
