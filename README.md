# YT_Live_comments_bot

Real-time sentiment and topic analysis for YouTube live streams, built specifically for **Hinglish (Hindi + English)** chat.  
LivePulse uses a **fine-tuned MuRIL ensemble** to classify chat messages and visualize insights on an interactive dashboard.

---

## 🧠 How It Works

```mermaid
flowchart LR
    A[YouTube Live Chat] --> B[Scraper - pytchat]
    B --> C[Redis Queue]
    C --> D[ML Pipeline]
    D --> E[FastAPI Backend]
    E --> F[Streamlit Dashboard]
---

## 🗂 Project Structure

```
├── backend/
│   ├── config.py          # Redis config + YouTube video ID
│   ├── main.py            # FastAPI server
│   └── scraper.py         # Live chat scraper + ML pipeline
├── frontend/
│   └── streamlit_app.py   # Dashboard UI
├── ml/
│   ├── sentiment_model.py # MuRIL + XLM-R + Multilingual ensemble
│   ├── topic_model.py     # Zero-shot BART topic classifier
│   └── train_muril.py     # Fine-tuning script for MuRIL
├── new_trained_data/
│   └── muril-sentimix/    # Fine-tuned MuRIL model weights
├── Redis-x64-5.0.14.1/    # Redis binaries (Windows)
└── requirements.txt
```

---

## ⚙️ Prerequisites

- Python 3.10+
- Redis (Windows binary included in `Redis-x64-5.0.14.1/`, or install via your OS package manager)
- A **live** YouTube stream video ID

---

## 🚀 Setup & Run

### 1. Clone the repo

```bash
git clone https://github.com/your-username/livepulse.git
cd livepulse
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Redis

**Windows** (using the included binary):
```bash
./Redis-x64-5.0.14.1/redis-server.exe
```

**Linux/macOS:**
```bash
redis-server
```

### 4. Set your YouTube video ID

Edit `backend/config.py`:

```python
VIDEO_ID = "YOUR_VIDEO_ID_HERE"  # e.g. "xQ04RZB12mY"
```

You can use a full URL or just the 11-character video ID.

### 5. Start the scraper

In a terminal:
```bash
python -m backend.scraper
```

This connects to the live chat, runs ML inference on each message, and stores results in Redis.

### 6. Start the FastAPI backend

In a second terminal:
```bash
uvicorn backend.main:app --reload --port 8000
```

API docs available at: `http://localhost:8000/docs`

### 7. Launch the Streamlit dashboard

In a third terminal:
```bash
streamlit run frontend/streamlit_app.py
```

Dashboard opens at: `http://localhost:8501`

---

## 🔌 API Endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Redis connectivity check |
| `GET /get_messages` | Last 50 raw chat messages |
| `GET /sentiment_trend` | Time-series sentiment data (last 200) |
| `GET /sentiment_summary` | Aggregate sentiment counts + avg confidence |
| `GET /topic_stats` | Per-topic message counts (last 100) |
| `GET /live_stats` | Combined snapshot for dashboard widgets |

---

## 🤖 Models Used

| Model | Purpose | Weight |
|---|---|---|
| `google/muril-base-cased` (fine-tuned) | Hinglish sentiment | 40% |
| `cardiffnlp/twitter-xlm-roberta-base-sentiment` | Multilingual sentiment | 35% |
| `tabularisai/multilingual-sentiment-analysis` | Multilingual sentiment | 25% |
| `facebook/bart-large-mnli` | Zero-shot topic classification | — |

### Fine-tuning MuRIL (optional)

To retrain the MuRIL model on the SentiMix dataset:

```bash
python ml/train_muril.py
```

Trained weights will be saved to `./muril-sentimix/`. Move them to `new_trained_data/muril-sentimix/` when done.

---

## 📦 Requirements

Key packages:
- `torch`, `transformers`, `sentencepiece` — ML inference
- `pytchat` — YouTube live chat scraping
- `redis` — message storage
- `fastapi`, `uvicorn` — REST API
- `streamlit`, `plotly`, `pandas` — dashboard

---

## ⚠️ Notes

- The scraper only works with **currently live** YouTube streams. VODs and premieres are not supported.
- First run will download XLM-R and BART models from Hugging Face (~2–3 GB). Ensure you have a stable internet connection.
- GPU is used automatically if CUDA is available, otherwise falls back to CPU.
