import os
from fastapi import FastAPI, Form, HTTPException, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import psycopg2
import numpy as np
import json
import PyPDF2
from docx import Document
import openai
import io
from fastapi.middleware.cors import CORSMiddleware




app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

DB_CONNECTION = "dbname=postgres user=postgres password=mysecretpassword host=db port=5432"
def get_db_connection():
    """Restituisce una connessione al database."""
    return psycopg2.connect(DB_CONNECTION)

def execute_query(query, params=None, fetch=False):
    """Esegue una query e, se richiesto, restituisce i risultati."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params or ())
            if fetch:
                return cur.fetchall()
            conn.commit()

load_dotenv()



openai.api_key = os.getenv("OPENAI_API_KEY")

# def init_db():

#     execute_query("CREATE EXTENSION IF NOT EXISTS vector;")

#     execute_query("""
#         CREATE TABLE IF NOT EXISTS documents (
#             id SERIAL PRIMARY KEY,
#             title TEXT NOT NULL,
#             content TEXT NOT NULL,
#             embedding VECTOR(1536) NOT NULL
#         );
#     """)

# # Esegui all'avvio
# init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserPrompt(BaseModel):
    prompt: str

@app.get("/")
async def read_root():
    return FileResponse("static/dashboard.html")

def chunk_pdf(file):
    reader = PyPDF2.PdfReader(file)
    chunks = []
    for page in reader.pages:
        chunks.append(page.extract_text())
    return chunks

def chunk_docx(file):
    doc = Document(file)
    chunks = []
    for para in doc.paragraphs:
        chunks.append(para.text)
    return chunks

def generate_embedding(text: str):
    response = openai.embeddings.create(input=[text], model="text-embedding-ada-002")
    embedding = response.data[0].embedding
    return np.array(embedding)


def load_document(title: str, content: str):
    embedding = generate_embedding(content)
    content = content.replace('\x00', '')
    
    execute_query("INSERT INTO documents (title, content, embedding) VALUES (%s, %s, %s)",
                (title, content, embedding.tolist()))


def get_uni_documents_from_db(prompt: str):
    prompt_embedding = generate_embedding(prompt)
    conn= get_db_connection()
    cur= conn.cursor()
    cur.execute("SELECT title, content, embedding FROM documents")
    rows = cur.fetchall()
    best_match = None
    best_similarity = 0.75
    for row in rows:

        db_embedding = np.array(json.loads(row[2]), dtype=np.float32)
        prompt_embedding = np.array(prompt_embedding, dtype=np.float32)
        
       
        similarity = np.dot(prompt_embedding, db_embedding) / (np.linalg.norm(prompt_embedding) * np.linalg.norm(db_embedding))
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = row
    
    if best_match:
        document_title = best_match[0]
        document_path = f"/static/documents/{document_title}"
        return {
            "document_name": document_title,
            "document_path": document_path
        }
    else:
        answer =  "Nessun documento trovato"
        return {"message": answer}

class answers(BaseModel):
    document_name: str
    content_document: str

 
def conversation(prompt: str, context: str):
    prompt_enhanced=f"Prompt: {prompt}\n\nContexto: {context}Se ti viene chiesto un documento universitario esegui la funzione associata e rispondi all'utente a breve glielo mostri\n\nRisposta:"
    # functions = [
    #     {
    #     "type": "function",
    #     "function":{
    #         "name": "test",
    #         "description": "returns hello world",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {},
    #             "additionalProperties": False
    #         }
    #     }
           
    # }]

    tools = [{
    "type": "function",
    "function": {
        "name": "get_uni_documents_from_db",
        "description": "Recupera i chunk dei documenti universitari piu' vicini al prompt dell'utente",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Prompt dell'utente"
                }
            },
            "required": [
                "prompt"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
}]
    
    messages=[ {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_enhanced}
            ]
        }]

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools
    )

    if response.choices[0].message.tool_calls is not None:
        tool_call = response.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
        result = get_uni_documents_from_db(args["prompt"])
        if "document_path" in result:
            messages.append(response.choices[0].message)
            messages.append({                            
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": [
                    {"type": "text", "text": "Informa l'utente che hai trovato il documento e lo mostrerai nel frame "}
                ]
            })
            response2 = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )    
            msg=response2.choices[0].message.content
            return {
                "message": msg,
                "document_name": result["document_name"],
                "document_path": result["document_path"]
            }
        else:
            return {"message": result["message"]}
        
    else:
        return {"message": response.choices[0].message.content}

@app.post("/upload_document/")
async def upload_document(title = Form(), file: UploadFile = File(...)):
    file_content = await file.read()
    file_like = io.BytesIO(file_content)
    print(file.filename)
    with open(f'static/documents/{file.filename}', 'wb') as f:
        f.write(file_content)

    if file.filename.endswith(".pdf"):
        chunks = chunk_pdf(file_like)
    elif file.filename.endswith(".docx"):
        chunks = chunk_docx(file_like)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")
    
    for chunk in chunks:
        load_document(title, chunk)
    
    return {"message": "Document uploaded successfully"}

@app.get("/ask_question/")
async def ask_question(user_prompt: str):
    answer = conversation(user_prompt,"Sei un assistente universitario") #get_answer_from_db(user_prompt)
    return answer

async def test():
    return "ciaot test"