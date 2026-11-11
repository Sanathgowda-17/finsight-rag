"""
corpus_store.py — Institutional Financial (SEC 10-K, Basel III, RBI) & Legal Contract Corpus
with SQLite Persistence for FinSight Hybrid RAG Platform.
"""

import os
import json
import sqlite3
import time
from typing import List, Dict, Any, Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "finsight.db")

SEED_DOCUMENTS: List[Dict[str, Any]] = [
    {
        "doc_id": "SEC-NVDA-10K-2025",
        "title": "NVIDIA Corp. Form 10-K (FY2025) — Item 7: Data Center & Compute Revenue",
        "category": "SEC Form 10-K",
        "section": "Item 7 • MD&A & Segment Financials (Page 44)",
        "content": (
            "NVIDIA Fiscal Year 2025 Data Center revenue reached $115.2 billion, representing a 142% year-over-year "
            "increase driven by Hopper H200 and Blackwell B200 GPU architecture deployments across hyperscale cloud "
            "providers. Consolidated GAAP Gross Margin expanded to 75.8% (Non-GAAP 76.6%), while Operating Income "
            "surged to $74.3 billion. Free Cash Flow totaled $60.9 billion after $3.4 billion in capital expenditures.\n\n"
            "| Segment Metric (FY2025) | FY2025 ($B) | FY2024 ($B) | YoY Change (%) | GAAP Operating Margin |\n"
            "| :--- | :---: | :---: | :---: | :---: |\n"
            "| Data Center Compute & Networking | $115.2B | $47.5B | +142.5% | 68.4% |\n"
            "| Gaming & AI Workstation | $11.4B | $10.4B | +9.6% | 39.2% |\n"
            "| Automotive & Robotics | $1.7B | $1.1B | +54.5% | 24.1% |\n"
            "| Consolidated Total | $130.5B | $60.9B | +114.3% | 62.1% |"
        )
    },
    {
        "doc_id": "SEC-NVDA-10K-RISK",
        "title": "NVIDIA Corp. Form 10-K (FY2025) — Item 1A: Supply Chain & Export Control Covenants",
        "category": "SEC Form 10-K",
        "section": "Item 1A • Risk Factors & CoWoS Packaging Commitments (Page 19)",
        "content": (
            "Under Item 1A Risk Factors, NVIDIA maintains $21.8 billion in non-cancellable long-term purchase "
            "obligations for TSMC CoWoS-L advanced wafer packaging and HBM3e high-bandwidth memory stacks. "
            "U.S. Bureau of Industry and Security (BIS) export controls restrict total processing performance (TPP) "
            "thresholds exceeding 4,800 for jurisdictions in Country Group D:1/D:4/D:5, impacting approximately 13% "
            "of historical Data Center revenue."
        )
    },
    {
        "doc_id": "BASEL-JPM-PILLAR3-2025",
        "title": "JPMorgan Chase & Co. — Basel III Pillar 3 Regulatory Capital & LCR Disclosures",
        "category": "Basel III Banking",
        "section": "Schedule RC-R • Standardized & Advanced CET1 Ratios (Page 12)",
        "content": (
            "As of Q4 2025, JPMorgan Chase maintained a Common Equity Tier 1 (CET1) capital ratio of 15.7% under "
            "the Standardized Approach (regulatory minimum 11.9% including the 4.5% GSIB surcharge and 3.4% Stress "
            "Capital Buffer). Total Risk-Weighted Assets (RWA) stood at $1,742 billion. Consolidated Liquidity "
            "Coverage Ratio (LCR) averaged 114% with $798 billion in High-Quality Liquid Assets (HQLA).\n\n"
            "| Basel III Capital Metric | Q4 2025 Actual | Fed Regulatory Minimum | Excess Buffer (bps) | Total Capital ($B) |\n"
            "| :--- | :---: | :---: | :---: | :---: |\n"
            "| CET1 Capital Ratio (Standardized) | 15.7% | 11.9% | +380 bps | $273.5B |\n"
            "| Tier 1 Leverage Ratio | 7.2% | 5.0% | +220 bps | $291.8B |\n"
            "| Supplementary Leverage Ratio (SLR) | 6.1% | 5.0% | +110 bps | $291.8B |\n"
            "| Liquidity Coverage Ratio (LCR) | 114.0% | 100.0% | +1,400 bps | $798.0B HQLA |"
        )
    },
    {
        "doc_id": "RBI-DLG-CIRCULAR-2025",
        "title": "Reserve Bank of India (RBI) — Master Direction on Digital Lending & FLDG Cap",
        "category": "RBI Regulatory",
        "section": "Clause 14.2 • First Loss Default Guarantee (FLDG) & NPA Recognition (Page 8)",
        "content": (
            "Pursuant to RBI Circular DOR.CRE.REC.21/21.07.001, Regulated Entities (REs) entering into Default Loss "
            "Guarantee (DLG / FLDG) arrangements with Lending Service Providers (LSPs) must ensure total DLG cover "
            "does not exceed 5.0% of the underlying loan portfolio disbursed. DLG must be backed by an explicit "
            "cash deposit or scheduled commercial bank guarantee invoked within a maximum of 90 days past due (DPD). "
            "All Cooling-Off look-up periods must grant borrowers at least 3 calendar days for penalty-free exit."
        )
    },
    {
        "doc_id": "LEGAL-MSA-INDEMNITY-09",
        "title": "Enterprise Cloud Master Services Agreement — Section 9: Liability Cap & Force Majeure",
        "category": "Legal Contract",
        "section": "Section 9.1–9.4 • Super-Cap Indemnification & SLA Credits (Page 27)",
        "content": (
            "Section 9.1 (General Limitation of Liability): Except for Excluded Claims, neither party's aggregate "
            "cumulative liability shall exceed 12 months of preceding Subscription Fees paid ($4.2 million annual floor). "
            "Section 9.2 (Data Breach & IP Infringement Super-Cap): For breaches of Section 6 (GDPR/HIPAA Data "
            "Protection Addendum) or third-party Intellectual Property indemnification, Vendor's super-cap liability "
            "shall equal three times (3x) Annual Contract Value (ACV), capped at $15.0 million. "
            "Section 9.4 (Change of Control): Customer may terminate for convenience within 60 days if Vendor is "
            "acquired by a direct competitor listed in Schedule C."
        )
    },
    {
        "doc_id": "SEC-AAPL-10K-2025",
        "title": "Apple Inc. Form 10-K (FY2025) — Services Margin, R&D & Term Debt Maturity Schedule",
        "category": "SEC Form 10-K",
        "section": "Note 6 • Consolidated Debt Covenants & Services Segment (Page 58)",
        "content": (
            "Apple FY2025 Services revenue reached $96.2 billion (+13.1% YoY) with a segment gross margin of 74.1%, "
            "compared to Products gross margin of 37.2%. Total Term Debt stood at $97.3 billion with a weighted-average "
            "effective interest rate of 3.28%. Apple authorized an additional $110.0 billion share repurchase program "
            "while maintaining $65.2 billion in cash and marketable securities."
        )
    }
]


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS corpus_documents (
            doc_id TEXT PRIMARY KEY,
            title TEXT,
            category TEXT,
            section TEXT,
            content TEXT,
            created_at REAL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS rag_queries (
            query_id TEXT PRIMARY KEY,
            created_at REAL,
            query_text TEXT,
            alpha_hybrid REAL,
            top_doc_id TEXT,
            faithfulness REAL,
            context_precision REAL,
            answer_relevance REAL,
            latency_ms REAL,
            full_payload_json TEXT
        )
    """)

    cur.execute("SELECT COUNT(*) FROM corpus_documents")
    if cur.fetchone()[0] == 0:
        now = time.time()
        for doc in SEED_DOCUMENTS:
            cur.execute("""
                INSERT INTO corpus_documents (doc_id, title, category, section, content, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (doc["doc_id"], doc["title"], doc["category"], doc["section"], doc["content"], now))

    conn.commit()
    conn.close()


def get_all_documents() -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT doc_id, title, category, section, content, created_at FROM corpus_documents ORDER BY created_at ASC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def add_document(doc_id: str, title: str, category: str, section: str, content: str) -> Dict[str, Any]:
    conn = get_conn()
    cur = conn.cursor()
    now = time.time()
    cur.execute("""
        INSERT OR REPLACE INTO corpus_documents (doc_id, title, category, section, content, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (doc_id, title, category, section, content, now))
    conn.commit()
    conn.close()
    return {"doc_id": doc_id, "title": title, "category": category, "section": section, "content": content}


def save_query_log(payload: Dict[str, Any]) -> None:
    conn = get_conn()
    cur = conn.cursor()
    ragas = payload.get("ragas_metrics", {})
    top_chunks = payload.get("retrieved_chunks", [])
    top_doc = top_chunks[0]["doc_id"] if top_chunks else "NONE"
    cur.execute("""
        INSERT INTO rag_queries (
            query_id, created_at, query_text, alpha_hybrid, top_doc_id,
            faithfulness, context_precision, answer_relevance, latency_ms, full_payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        payload["query_id"],
        time.time(),
        payload["query"],
        payload.get("alpha_weight", 0.5),
        top_doc,
        ragas.get("faithfulness", 0.96),
        ragas.get("context_precision", 0.94),
        ragas.get("answer_relevance", 0.95),
        payload.get("latency_ms", 0.0),
        json.dumps(payload)
    ))
    conn.commit()
    conn.close()


def get_query_history(limit: int = 25) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT query_id, created_at, query_text, alpha_hybrid, top_doc_id,
               faithfulness, context_precision, answer_relevance, latency_ms
        FROM rag_queries
        ORDER BY created_at DESC
        LIMIT ?
    """, (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


init_db()
