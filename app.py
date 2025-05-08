import os
import cv2
import numpy as np
import types
import sys
import fitz  
from flask import Flask, render_template, request, flash, session
from dotenv import load_dotenv
try:
    import albumentations
    sys.modules['albumentations.pytorch'] = types.ModuleType('albumentations.pytorch')
except ImportError:
    pass
from PyPDF2 import PdfReader
from docx import Document
from paddleocr import PaddleOCR
from cv2 import dnn_superres
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
secret_key = os.getenv("SECRET_KEY")
app = Flask(__name__)
app.secret_key = secret_key or os.urandom(24)
base_dir = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = os.path.join(base_dir, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False)
sr = None
try:
    tmp = dnn_superres.DnnSuperResImpl_create()
    tmp.readModel("EDSR_x3.pb")
    tmp.setModel("edsr", 3)
    sr = tmp
    print("SR model loaded")
except Exception:
    app.logger.warning("SR unavailable, continuing without it")
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
llm = ChatGroq(groq_api_key=groq_api_key, model_name='llama3-8b-8192', temperature=0.0)
template = (
    "You are an expert summarizer. Read the following text carefully and write a clear, concise, and informative summary. "
    "Focus on the main ideas, key points, and important details. Remove repetition or filler. Keep it neutral and clear. "
    "Length should be about 1/3 of the original content.\n\nText: {question}"
)
prompt = ChatPromptTemplate.from_template(template)
output_parser = StrOutputParser()
chain = ({"question": RunnablePassthrough()} | prompt | llm | output_parser)
def preprocess_image(img: np.ndarray) -> np.ndarray:
    if sr:
        try:
            img = sr.upsample(img)
        except:
            pass
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3),0)
    _, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return thresh

def ocr_from_image(path: str) -> str:
    img = cv2.imread(path)
    proc = preprocess_image(img)
    if proc.ndim == 2:
        proc = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)
    res = ocr.ocr(proc, cls=True)
    return "\n".join([l[1][0] for p in res for l in p])

def extract_from_pdf(path: str) -> str:
    texts = []
    try:
        doc = fitz.open(path)
        for page in doc:
            pix = page.get_pixmap()
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            proc = preprocess_image(arr)
            if proc.ndim == 2:
                proc = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)
            res = ocr.ocr(proc, cls=True)
            texts.append("\n".join([l[1][0] for p in res for l in p]))
        return "\n\n".join(texts)
    except Exception:
        reader = PdfReader(path)
        full_text = []
        for p in reader.pages:
            txt = p.extract_text() or ''
            full_text.append(txt)
        return "\n\n".join(full_text)

def extract_from_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join([para.text for para in doc.paragraphs if para.text])

def extract_text(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ['.png', '.jpg', '.jpeg']:
        return ocr_from_image(filepath)
    if ext == '.pdf':
        return extract_from_pdf(filepath)
    if ext in ['.docx', '.doc']:
        return extract_from_docx(filepath)
    return ''
def summarize(text: str) -> str:
    try:
        return chain.invoke(text)
    except Exception as e:
        return f"Summarization error: {e}"
@app.route('/', methods=['GET','POST'])
def index():
    message = ''
    extracted = session.get('extracted','')
    summary = session.get('summary','')
    if request.method=='POST':
        action = request.form.get('action')
        f = request.files.get('file') or request.files.get('image')
        if action=='extract' and f:
            path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
            f.save(path)
            text = extract_text(path)
            if not text.strip():
                flash('No readable text found.')
                session.pop('extracted',None); session.pop('summary',None)
            else:
                session['extracted']=text; session.pop('summary',None)
                message='Extraction completed.'; extracted=text; summary=''
        elif action=='summarize':
            if extracted:
                summary=summarize(extracted); session['summary']=summary; message='Summary completed.'
            else:
                flash('Please extract first.')
    return render_template('web.html',message=message,output=extracted,summary=summary)
app.add_url_rule('/','index',index,methods=['GET','POST'])
if __name__=='__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)