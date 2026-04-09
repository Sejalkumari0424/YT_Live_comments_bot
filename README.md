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
