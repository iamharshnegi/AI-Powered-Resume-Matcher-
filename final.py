from flask import Flask, request, render_template
from flask_sqlalchemy import SQLAlchemy
import os
import re
import docx2txt
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import socket
import concurrent.futures

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
print("Gemini Key:", GEMINI_API_KEY)  # For debug only

import google.generativeai as genai

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///resumematcher.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db = SQLAlchemy(app)

# Allowed extensions
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

# ---------- Database Models ----------

class Resume(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(120), nullable=False)
    content = db.Column(db.Text, nullable=False)
    score = db.Column(db.Integer)
    suggestion = db.Column(db.String(200))

class JobDescription(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)

# ---------- File Type Handling ----------

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_from_txt(file_path)
    return ""

def is_connected():
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=10)
        return True
    except OSError:
        return False

def call_gemini(prompt):
    return gemini_model.generate_content(prompt)

# ---------- Routes ----------

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/upload/jobseeker")
def jobseeker_form():
    return render_template("test_1.html")

@app.route("/upload/hr")
def hr_form():
    return render_template("hr.html")

@app.route('/analyze_resume', methods=['POST'])
def analyze_resume():
    if request.method == 'POST':
        job_description = request.form['job_description']
        resume_file = request.files.get('resumes')

        if not job_description or not resume_file or not allowed_file(resume_file.filename):
            return render_template('test_1.html', message="Please upload a resume and enter a job description.")

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
        resume_file.save(filepath)
        resume_text = extract_text(filepath)

        online = is_connected()

        try:
            if online:
                # Gemini prompt for Job Seeker - focused on improving their resume for the given job description
                prompt = (
                    f"You are a career coach. Analyze the following resume and job description. "
                    f"Provide personalized, actionable advice to help the job seeker improve their resume for the role. "
                    f"Address the summary, skills, experience, and keywords sections separately.\n\n"
                    f"Job Description:\n{job_description}\n\nResume:\n{resume_text}"
                )

                print("Sending to Gemini (job seeker):\n", prompt)
                response_obj = gemini_model.generate_content(prompt)
                advice_raw = response_obj.text if response_obj and response_obj.text else "No advice provided."
                advice_html = convert_markdown_to_html(advice_raw)

                # Optional: Calculate similarity score too, like in matcher
                score_prompt = (
                    f"Rate from 0 to 100 how well this resume matches the job description.\n\n"
                    f"Job Description:\n{job_description}\n\nResume:\n{resume_text}"
                )
                score_response_obj = gemini_model.generate_content(score_prompt)
                score_text = score_response_obj.text if score_response_obj and score_response_obj.text else ""
                match = re.search(r'\b\d{1,3}\b', score_text)
                score = int(match.group(0)) if match else 0

            else:
                raise ConnectionError("No internet connection.")

        except Exception as e:
            print(f"Gemini failed for job seeker resume: {e}")
            try:
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform([job_description, resume_text])
                cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
                score = int(cosine_sim[0][0] * 100)
                advice_html = (
                    "Gemini was unavailable. Score calculated using TF-IDF similarity. "
                    "Improve keyword alignment and tailor your resume to the job description."
                )
            except Exception as ve:
                print(f"TF-IDF fallback failed: {ve}")
                score = 0
                advice_html = "Unable to analyze resume due to processing error."

        # Save resume and results to DB
        resume_entry = Resume(
            filename=resume_file.filename,
            content=resume_text,
            score=score,
            suggestion=advice_html
        )
        db.session.add(resume_entry)
        db.session.commit()

        return render_template(
            'test_1.html',
            message="Resume analysis complete:",
            top_resumes=[resume_file.filename],
            similarity_scores=[score],
            suggestions=[advice_html]
        )

    return render_template('test_1.html')


@app.route('/matcher', methods=['POST'])
def matcher():
    if request.method == 'POST':
        job_description = request.form['job_description']
        resume_files = request.files.getlist('resumes')

        if not job_description or not resume_files:
            return render_template('hr.html', message="Please upload resumes and enter a job description.")

        job = JobDescription(content=job_description)
        db.session.add(job)
        db.session.commit()

        top_resumes = []
        similarity_scores = []
        suggestions = []

        online = is_connected()

        for resume_file in resume_files:
            if resume_file and allowed_file(resume_file.filename):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
                resume_file.save(filepath)
                resume_text = extract_text(filepath)

                try:
                    if online:
                        # --- Score prompt ---
                        score_prompt = (
                            f"Rate how well this resume matches the job description from 0 to 100.\n\n"
                            f"Job Description:\n{job_description}\n\nResume:\n{resume_text}"
                        )
                        score_response = gemini_model.generate_content(score_prompt)
                        score_text = score_response.text if score_response and score_response.text else ""
                        match = re.search(r'\b\d{1,3}\b', score_text)
                        score = int(match.group(0)) if match else 0

                        # --- HR suggestion prompt ---
                        suggestion_prompt = (
                            f"As an HR expert, analyze the following resume in comparison to the provided job description. "
                            f"Evaluate how well the resume aligns with the job description, specifically highlighting any missing skills or qualifications. "
                            f"Provide clear, actionable suggestions for improvement.\n\n"
                            f"Finally, provide a clear final verdict on whether this candidate is a strong match for the role, "
                            f"including a brief reason for your conclusion (e.g., 'Strong Match due to relevant skills and experience', "
                            f"'Consider with Reservations due to missing key skills in X', 'Not a Fit due to lack of experience')."
                            f"\n\nJob Description:\n{job_description}\n\nResume:\n{resume_text}"
                        )
                        suggestion_response = gemini_model.generate_content(suggestion_prompt)
                        raw_suggestion = suggestion_response.text.strip() if suggestion_response and suggestion_response.text else "No suggestions provided."
                        suggestion_html = convert_markdown_to_html(raw_suggestion)

                    else:
                        raise ConnectionError("No internet connection.")

                except Exception as e:
                    print(f"[Fallback] Error with Gemini for {resume_file.filename}: {e}")
                    try:
                        vectorizer = TfidfVectorizer(stop_words='english')
                        tfidf_matrix = vectorizer.fit_transform([job_description, resume_text])
                        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
                        score = int(cosine_sim[0][0] * 100)
                        suggestion_html = (
                            "Gemini was unavailable. Score calculated using TF-IDF similarity. "
                            "Improve keyword alignment with job description."
                        )
                    except Exception as ve:
                        print(f"TF-IDF fallback failed for {resume_file.filename}: {ve}")
                        score = 0
                        suggestion_html = "Unable to analyze resume due to processing error."

                # Store result
                top_resumes.append(resume_file.filename)
                similarity_scores.append(score)
                suggestions.append(suggestion_html)

                # Save in DB
                resume_entry = Resume(
                    filename=resume_file.filename,
                    content=resume_text,
                    score=score,
                    suggestion=suggestion_html
                )
                db.session.add(resume_entry)

        db.session.commit()

        return render_template(
            'hr.html',
            message="Top matching resumes (Gemini-enhanced with fallback):",
            top_resumes=top_resumes,
            similarity_scores=similarity_scores,
            suggestions=suggestions
        )

    return render_template('hr.html')

# ---------- Admin View ----------

@app.route("/admin/resumes")
def view_resumes():
    resumes = Resume.query.order_by(Resume.score.desc()).all()
    return render_template("admin_view.html", resumes=resumes)

# ---------- Run App ----------

if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    with app.app_context():
        db.create_all()

    app.run(debug=True)

