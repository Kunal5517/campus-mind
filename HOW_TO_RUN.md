# 🎓 Medicaps University — College Enquiry Chatbot
## How to Run Guide

---

### 📁 Project Structure
```
college-chatbot/
├── app.py          ← Flask backend (NLP engine)
├── index.html      ← Frontend UI (chat interface)
├── data.json       ← Knowledge base (intents)
└── requirements.txt  ← (see below)
```

---

### ⚙️ Step 1 — Install Python Dependencies

```bash
pip install flask nltk
```

For Python 3.11+, also run once to download NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
```
> ✅ **Note:** The app auto-downloads NLTK data on first run.

---

### 📂 Step 2 — Place Files Together

Make sure all three files are in the **same folder**:
```
app.py      ← Flask server
index.html  ← Chat UI (served by Flask)
data.json   ← Intent knowledge base
```

---

### ▶️ Step 3 — Start the Flask Server

```bash
python app.py
```

You should see:
```
=======================================================
  🎓 Medicaps University Chatbot – Starting Up
  Visit: http://127.0.0.1:5000
=======================================================
[INFO] Chatbot trained on 80+ patterns across 13 intents.
 * Running on http://0.0.0.0:5000
```

---

### 🌐 Step 4 — Open in Browser

Visit: **http://127.0.0.1:5000**

Click the 💬 chat bubble at the **bottom-right corner** to start chatting!

---

### 💬 Sample Questions to Try

| Category | Example Queries |
|----------|-----------------|
| Greetings | "Hi", "Hello", "Good morning" |
| Admissions | "How to apply?", "What is the admission process?" |
| Courses | "What courses are offered?", "List all programs" |
| Fees | "What is the fee structure?", "How much does B.Tech cost?" |
| Scholarships | "Tell me about scholarships", "Merit-based fee waiver?" |
| Placements | "What is the highest package?", "Which companies recruit?" |
| Hostel | "Is hostel available?", "Hostel fees?" |
| Facilities | "What are the campus facilities?", "Is there a library?" |
| Entrance Exam | "Tell me about the entrance test", "MUET exam pattern" |
| Contact | "What is your address?", "How to contact the college?" |

---

### 🧠 How the NLP Works

```
User Input
    ↓
Preprocessing: lowercase → tokenize → remove stopwords → stem
    ↓
TF-IDF Vectorization (custom, pure Python — no sklearn needed)
    ↓
Cosine Similarity vs all training patterns
    ↓
Best matching Intent (threshold: 0.15)
    ↓
Random response from matched intent's responses list
```

---

### 🔧 Customization

**Add new intents** — Edit `data.json`:
```json
{
  "tag": "my_new_topic",
  "patterns": ["keyword 1", "ask about X", "how to X"],
  "responses": ["Your custom answer here."]
}
```

**Change college name** — Edit `data.json` responses and `index.html` branding sections.

**Tune sensitivity** — In `app.py`, adjust the `threshold` parameter in `matcher.predict()`:
- Lower (e.g., `0.1`) → more matches, less precise
- Higher (e.g., `0.25`) → fewer matches, more accurate

---

### 📦 requirements.txt

```
flask>=2.3.0
nltk>=3.8.0
```

Install with: `pip install -r requirements.txt`

---

### 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: flask` | Run `pip install flask` |
| `ModuleNotFoundError: nltk` | Run `pip install nltk` |
| NLTK download error | Run the NLTK downloads manually (Step 1) |
| Port 5000 already in use | Change `port=5000` to `port=5001` in `app.py` |
| "Couldn't reach the server" in UI | Make sure `python app.py` is running |

---

*Built with Flask + NLTK + Pure CSS/JS | No React, no webpack, no fuss.*
