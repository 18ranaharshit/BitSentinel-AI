# ⚡ AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

An end-to-end Graph Neural Network (GNN), Multimodal Cross-Layer Correlation, and Machine Learning platform for real-time Bitcoin transaction monitoring, address entity resolution, and fraud risk detection.

📄 **[Full Technical Write-Up](docs/TECHNICAL_WRITEUP.md)** — Comprehensive architecture, model choices, explainability methodology, and honest limitations disclosure.

---

## 🌟 Architecture & Highlights

- **Graph Neural Network Pretraining**: 2-Layer **GraphSAGE Encoder** (`torch_geometric`) pretrained self-supervised via link prediction across **203,769 transactions and 234,355 directed graph edges**.
- **Multimodal Network↔Blockchain Correlation**: Cross-layer fusion engine combining on-chain graph suspicion with BGP ASN routing density and rolling 6-hour `/24` subnet peer bursts, boosting Test F1 to **`0.8041`** while eliminating **100% of false alarms on exchange hot-wallet bursts**.
- **Address Entity Resolution**: **Common-Input Co-Spend Heuristics** (Union-Find) clustering co-spent addresses into 5,900+ multi-address entity wallet clusters.
- **Explainable AI (XAI)**: Decision-tree path probability attribution providing transparent plain-English reasons for every flagged alert.
- **Leakage-Free Temporal & Account Splitting**:
  - Elliptic dataset split using standard **Temporal Protocol** (Train: steps 1–29, Val: steps 30–34, Test: steps 35–49).
  - BABD-13 dataset split using **Account-Disjoint Stratified Splitting** (70/15/15) ensuring zero account leakage.
- **Full-Stack Monitoring Application**:
  - **FastAPI Backend** (`backend/main.py`, `backend/network_alerts.py`): REST API endpoints + Real-Time WebSocket stream (`/ws/stream`).
  - **Vite + React Dashboard** (`frontend/`): Dark glassmorphism UI with live block stream ticker, interactive search tool, network correlation inspector, wallet cluster browser, and benchmark visualizers.

---

## 📊 Benchmark Results

### 1. Elliptic Transaction Fraud Classification
| Model | Split | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Raw)** | **Val** | **0.9875** | **0.9323** | **0.9591** | **0.9964** | **0.9889** |
| **Random Forest (Raw)** | **Test** | **0.9808** | **0.6621** | **0.7905** | **0.9219** | **0.7820** |
| **GraphSAGE (Fine-Tuned)** | Val | 0.5953 | 0.9509 | 0.7322 | 0.9619 | 0.7427 |

### 2. BABD-13 Multi-Class Address Category Classification
| Model | Split | Accuracy | Macro F1 | Weighted F1 | Macro Precision | Macro Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **Val** | **0.9647** | **0.8271** | **0.9636** | **0.8628** | **0.8105** |
| **Random Forest** | **Test** | **0.9637** | **0.8347** | **0.9626** | **0.8904** | **0.8150** |

### 3. Network↔Blockchain Cross-Layer Correlation Benchmark
*Evaluated on synthetic network telemetry injected onto real Elliptic TXIDs (as real seized network-layer pcap data is unavailable per the problem statement). Note: Elliptic transactions are PCA-anonymized TXID graph nodes and do not include real wallet addresses; address-level linkage is handled separately via BABD-13 and raw block heuristics.*

| Architecture Layer | Overall Test Prec | Overall Test Rec | Overall Test F1 | Test Botnet Recall | Test Exchange FPs (Hard Negatives) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Network-Only Correlation** | 55.40% | 14.68% | 0.2321 | 159 / 159 (100.0%) | 128 / 128 (100.0% False Alarms) |
| **Blockchain-Only RF (165 Feats)** | 98.08% | 66.20% | 0.7905 | 105 / 159 (66.0%) | 0 / 128 (0.0% Cleanly Ignored) |
| **★ Multimodal Fusion Model** | **96.75%** | **68.79%** | **0.8041** | **112 / 159 (70.4%)** | **0 / 128 (0.0% FPs — 100% Cleared)** |

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Environment Setup

```bash
# Clone the repository
git clone https://github.com/18ranaharshit/BitSentinel-AI.git
cd BitSentinel-AI

# Install Python dependencies
pip install pandas numpy scikit-learn torch torch-geometric xgboost lightgbm matplotlib seaborn fastapi uvicorn websockets
```

### 2. Run Preprocessing, Pretraining & Benchmarks

```bash
# 1. Dataset Exploration
python explore_datasets.py

# 2. BABD-13 Data Cleaning & Account-Disjoint Splitting
python fix_babd13_split.py

# 3. Verify Elliptic Temporal Splits & Graph Edges
python verify_elliptic_prep.py

# 4. Self-Supervised GraphSAGE Pretraining (100 Epochs)
python train_elliptic_graphsage.py

# 5. Downstream Elliptic Fraud Classification
python train_elliptic_classifier.py

# 6. Downstream BABD-13 Multi-Class Address Classification
python train_babd13_classifier.py

# 7. Generate Synthetic Network Telemetry Metadata
python generate_network_metadata.py

# 8. Genuine GeoIP & BGP ASN Resolution
python add_geoip_resolution.py

# 9. Cross-Layer Correlation & Rolling Subnet Temporal Density
python correlate_network_blockchain.py

# 10. Train Multimodal Cross-Layer Risk Fusion Model & Export Alerts
python train_combined_risk_model.py

# 11. Co-Spend Address Entity Clustering (Union-Find)
python bitcoin_heuristics.py

# 12. Real-Time Raw Block JSON Feature Extraction & Scoring
python parse_raw_blocks_and_predict.py

# 13. Generate Benchmark Visualization Charts
python generate_benchmark_report.py

# 14. Pipeline Integrity Verification Audit
python verify_pipeline_integrity.py
```

---

## 🖥️ Running the Web Monitoring Application

### Launch FastAPI Backend
```bash
python -m uvicorn backend.main:app --port 8005 --reload
```
- API Base: `http://localhost:8005`
- Swagger Docs: `http://localhost:8005/docs`

### Launch React Frontend
```bash
cd frontend
npm install
npm run dev
```
- React Dashboard: `http://localhost:3005`

---

## 📁 Repository Structure

```
├── docs/
│   └── TECHNICAL_WRITEUP.md        # Comprehensive technical report & PS deliverable
├── backend/
│   ├── main.py                     # FastAPI REST & WebSocket Application
│   └── network_alerts.py           # Network correlation REST API router
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   └── src/
│       ├── index.css               # Dark glassmorphism stylesheet
│       ├── main.jsx                # Entrypoint
│       └── App.jsx                 # React dashboard UI component
├── generate_network_metadata.py    # Synthetic network telemetry generator with randomized sizing
├── add_geoip_resolution.py         # Pure-Python BGP routing table resolver (ip2asn-v4.tsv.gz)
├── correlate_network_blockchain.py # Cross-layer correlation & rolling temporal burst engine
├── train_combined_risk_model.py    # Multimodal fusion classifier & alert persistence
├── explainability.py               # SHAP tree attribution and heuristic rule explainability engine
├── bitcoin_heuristics.py           # Address co-spend clustering heuristics engine
├── live_stream_simulator.py        # Real-time block stream & fraud alert simulator
├── explore_datasets.py             # EDA & dataset statistics
├── fix_babd13_split.py             # BABD-13 artifact cleaning & account split
├── verify_elliptic_prep.py         # Elliptic disjointness & edge diagnostics
├── train_elliptic_graphsage.py     # Self-supervised GraphSAGE pretraining
├── train_elliptic_classifier.py    # Elliptic downstream benchmarking
├── train_babd13_classifier.py      # BABD-13 multi-class classifier
├── parse_raw_blocks_and_predict.py # Raw block JSON parser & live inference
├── generate_benchmark_report.py    # Generates PNG benchmark evaluation charts
└── verify_pipeline_integrity.py    # Integrity verification & checkpoint audit script
```
