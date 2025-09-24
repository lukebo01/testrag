"""
Modulo che implementa la catena RAG (Retrieval-Augmented Generation) usando LangChain.
"""
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# --- NUOVI IMPORT ---
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# --- FINE NUOVI IMPORT ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser # Aggiungiamo un parser di output
from typing import List
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

from operator import itemgetter
from langchain_core.runnables import RunnableParallel


def create_rag_chain(file_path: str): # Rimuoviamo model_name, non serve più
    """
    Crea una catena RAG completa usando modelli locali.
    """
    # Carica il documento
    loader = TextLoader(file_path, encoding="utf-8") # Aggiungiamo l'encoding per sicurezza
    documents = loader.load()
    
    print(f"Documento caricato: {len(documents)} documenti, {len(documents[0].page_content)} caratteri")
    
    # Dividi il documento in chunk
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"Documento diviso in {len(chunks)} chunks")
    
    # --- MODIFICA 1: USA EMBEDDING LOCALI ---
    # Scegliamo un modello di embedding leggero e performante
    embedding_model_name = "all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    # Crea il database vettoriale (la prima volta scaricherà il modello di embedding)
    print("Creazione degli embedding in corso (potrebbe richiedere tempo la prima volta)...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("Database vettoriale creato")
    
    # Crea il retriever
    retriever = vector_store.as_retriever()

    # --- MODIFICA 2: USA LLM GOOGLE ---
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    
    # Definisci il template del prompt
    template = """
    Usa le seguenti informazioni di contesto per rispondere alla domanda.
    Se non conosci la risposta basandoti sul contesto, dì che non lo sai.
    
    Contesto: {context}
    
    Domanda: {question}
    
    Risposta: """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Funzione per formattare i documenti recuperati
    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Crea la catena RAG usando LCEL
    rag_chain = (
        RunnableParallel(
            context=(itemgetter("question") | retriever | format_docs),
            question=itemgetter("question")
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("Catena RAG creata con modelli locali")
    
    return rag_chain