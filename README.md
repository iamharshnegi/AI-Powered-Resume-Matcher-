# 🚀 AI-Powered Resume Matcher Tool

An intelligent system for **HR professionals** and **Job Seekers** that uses AI to streamline resume evaluation and optimization.

---

## 🔧 Tech Stack

- **Backend:** Python, Flask  
- **Frontend:** HTML, CSS, JavaScript (Dark Mode)  
- **AI:** Gemini API 2.5 (primary) + TF-IDF & Cosine Similarity (fallback)  
- **File Handling:** pdfplumber, docx2txt, PyMuPDF  
- **Database:** SQLite via SQLAlchemy  

---

## 📝 Setup

1. Clone the repo  
2. Create a `.env` file:

```
GEMINI_API_KEY=your_api_key_here
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
python final.py
```

5. Open: `http://127.0.0.1:5000`

---

## ✨ Features

### HR Module

- Upload multiple resumes  
- AI analyzes each resume against the Job Description  
- **Final Verdict:** Strong Match / Consider / Not a Fit  
- **Suggestions:** How much improvement is needed per resume to better match the JD  
- Matching score (0–100)  
- Fallback to TF-IDF & Cosine Similarity if AI unavailable  

### Job Seeker Module

- Upload a single resume  
- Align resume with Job Description  
- Personalized improvement suggestions for:  
  - Summary  
  - Skills  
  - Experience  
  - Keywords  
- Matching score (0–100)  
- Fallback to TF-IDF & Cosine Similarity if AI unavailable  

### Common Features

- PDF, DOCX, TXT supported  
- Markdown-to-HTML conversion for better readability of suggestions  
- Dark Mode UI  
- Real-time matching insights  

---

## 💼 Use Cases

- **HR & Recruiters:** Automate resume screening, save time, make data-driven decisions  
- **Job Seekers:** Optimize resumes to improve chances for a job  

---

## 🧠 What I Learned

- Flask routing and templates  
- Gemini 2.5 API integration for semantic resume matching  
- NLP techniques: TF-IDF + Cosine Similarity  
- File parsing and handling (PDF, DOCX, TXT)  
- UI/UX design: Dark Mode and suggestion formatting  
- Markdown-to-HTML conversion  

---

## ⚠️ Notes

- Keep `.env` private  
- Ensure all HTML files are inside `templates/`  
- `uploads/` folder is for local testing only
