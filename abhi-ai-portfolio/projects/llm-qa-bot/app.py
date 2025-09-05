# Simple LLM QA Bot (RAG-lite)
import os, gradio as gr, faiss, glob, numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QA_MODEL = "deepset/roberta-base-squad2"
DOC_DIR = "./sample_docs"

def read_files(folder):
    docs = []
    for path in glob.glob(os.path.join(folder, "*")):
        if path.endswith((".txt",".md")):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                docs.append((path, f.read()))
    return docs

def chunk_text(text, size=500, overlap=100):
    return [text[i:i+size] for i in range(0, len(text), size-overlap)]

class Store:
    def __init__(self):
        self.embed = SentenceTransformer(EMBED_MODEL)
        self.index, self.meta = None, []
    def build(self, docs):
        self.meta, chunks = [], []
        for path, text in docs:
            for c in chunk_text(text):
                chunks.append(c); self.meta.append({"src":path,"chunk":c})
        if not chunks: return 0
        X = self.embed.encode(chunks, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        self.index = faiss.IndexFlatIP(X.shape[1]); self.index.add(X)
        return len(chunks)
    def query(self,q,k=3):
        if not self.index: return []
        qv = self.embed.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        D,I = self.index.search(qv,k)
        return [(self.meta[i],D[0][j]) for j,i in enumerate(I[0]) if i!=-1]

vs, qa = Store(), pipeline("question-answering", model=QA_MODEL)

def refresh(_): return f"Indexed {vs.build(read_files(DOC_DIR))} chunks"
def answer(q):
    top=vs.query(q,3)
    if not top: return "","No docs",""
    ctx="\n".join([m["chunk"] for m,_ in top])
    res=qa(question=q, context=ctx)
    return res["answer"], str(res["score"]), ctx[:1000]

with gr.Blocks() as demo:
    with gr.Row(): r=gr.Button("Refresh"); status=gr.Textbox()
    q=gr.Textbox(label="Ask"); a=gr.Button("Go")
    ans=gr.Textbox(label="Answer"); sc=gr.Textbox(label="Score"); ctx=gr.Textbox(label="Context")
    r.click(refresh, inputs=None, outputs=status)
    a.click(answer, inputs=q, outputs=[ans,sc,ctx])
if __name__=="__main__": demo.launch()
