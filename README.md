# 📊 FinSight — Multimodal Financial (SEC 10-K / Basel III / RBI) & Legal Hybrid RAG Platform

**FinSight** is an institutional **Hybrid Retrieval-Augmented Generation (RAG) & RAGAS Evaluation Platform** engineered for high-precision financial filings (`SEC Form 10-K`, `Basel III Pillar 3 Capital Disclosures`, `RBI Digital Lending Circulars`) and enterprise legal contracts (`Master Services Agreements`).

---

## 🔬 Why Hybrid RRF + Cross-Encoder Reranking Beats Naive RAG

Naive vector-only RAG fails in financial and legal workflows because dense embeddings smooth over exact alphanumeric identifiers (`CET1`, `DOR.CRE.REC.21`, `Section 9.2`, `$115.2B`, `CoWoS-L`). FinSight resolves this via a **4-stage retrieval & verification pipeline**:

1. **Sparse Lexical Engine (`BM25Okapi`, $k_1 = 1.5, b = 0.75$)**:
   - Computes Inverse Document Frequency ($\text{IDF}$) and term-frequency saturation to guarantee exact matches on regulatory circular numbers, GAAP line items, and legal clause numbers.
2. **Dense Semantic Vector Engine**:
   - Computes L2-normalized word unigram/bigram + character subword n-gram embeddings with domain-specific financial synonym expansion.
3. **Reciprocal Rank Fusion (RRF)**:
   - Combines Sparse BM25 rank ($r_{\text{BM25}}$) and Dense Vector rank ($r_{\text{Dense}}$) without scale distortion:
     $$\text{RRF}(d) = \frac{1 - \alpha}{k + r_{\text{BM25}}(d)} + \frac{\alpha}{k + r_{\text{Dense}}(d)}$$
4. **Cross-Encoder Reranker + Multimodal Markdown Table Extractor + RAGAS Evaluation Triad**:
   - Re-scores candidate chunks jointly with the query, parses embedded GAAP & Basel III tables into structured matrices, and audits every response across **Faithfulness (Groundedness)**, **Context Precision**, and **Answer Relevance**.

---

## 🚀 Quickstart (Local Execution)

```bash
cd backend
python3 -m uvicorn app:app --host 0.0.0.0 --port 8002
```
Open **`http://localhost:8002`** in your browser.
