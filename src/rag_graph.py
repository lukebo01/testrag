# src/rag_graph.py

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings # Usiamo la versione aggiornata
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from operator import itemgetter
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

# --- 1. Definizione dello Stato del Grafo ---
# Lo stato è un oggetto che viene passato tra i nodi. 
# Deve contenere tutte le informazioni necessarie per il processo.
class GraphState(TypedDict):
    question: str       # La domanda dell'utente
    documents: List[str]# I documenti recuperati
    generation: str     # La risposta generata dall'LLM
    
# --- 2. Definizione dei Nodi del Grafo ---
# Ogni nodo è una funzione Python che esegue un'azione.

def retrieve_documents(state: GraphState):
    """
    Nodo 1: Recupera i documenti dal Vector Store.
    """
    print("--- NODO: RECUPERA DOCUMENTI ---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def grade_documents(state: GraphState):
    """
    Nodo 2: Valuta la pertinenza dei documenti recuperati.
    Usa un LLM con un prompt specifico per ottenere un output JSON.
    """
    print("--- NODO: VALUTA PERTINENZA DOCUMENTI ---")
    question = state["question"]
    documents = state["documents"]
    
    # LLM per la valutazione
    grading_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    prompt = ChatPromptTemplate.from_template(
        """Sei un valutatore esperto. Il tuo compito è valutare se i documenti recuperati sono pertinenti per rispondere alla domanda dell'utente.
        Fornisci una valutazione binaria ('yes' o 'no') in formato JSON. La chiave deve essere 'score'.

        Domanda: {question}
        Documenti:
        {documents}

        JSON di valutazione:
        """
    )
    
    grader_chain = prompt | grading_llm | JsonOutputParser()
    
    # Estraiamo il contenuto del documento per la valutazione
    docs_content = "\n\n".join([d.page_content for d in documents])
    
    result = grader_chain.invoke({"question": question, "documents": docs_content})
    grade = result['score']
    print(f"Esito valutazione: {grade}")

    # Il risultato della valutazione non viene salvato nello stato, ma verrà usato nell'arco condizionale.
    # Ritorniamo un valore che l'arco condizionale userà per decidere.
    if grade.lower() == "yes":
        return "generate" # Prossimo passo: genera risposta
    else:
        return "fallback" # Prossimo passo: usa il fallback


def generate_answer(state: GraphState):
    """
    Nodo 3: Genera la risposta basandosi sui documenti pertinenti.
    """
    print("--- NODO: GENERA RISPOSTA ---")
    question = state["question"]
    documents = state["documents"]
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, convert_system_message_to_human=True)
    
    prompt = ChatPromptTemplate.from_template(
        """Sei un assistente per domande e risposte. Usa il seguente contesto per rispondere alla domanda.
        Sii conciso e rispondi basandoti solo sulle informazioni fornite.
        
        Contesto: {context}
        Domanda: {question}
        
        Risposta:
        """
    )
    
    rag_chain = prompt | llm | StrOutputParser()
    
    docs_content = "\n\n".join([d.page_content for d in documents])
    generation = rag_chain.invoke({"context": docs_content, "question": question})
    
    return {"documents": documents, "question": question, "generation": generation}


def fallback_answer(state: GraphState):
    """
    Nodo 4: Fornisce una risposta standard quando i documenti non sono pertinenti.
    """
    print("--- NODO: FALLBACK - DOCUMENTI NON PERTINENTI ---")
    question = state["question"]
    generation = "Mi dispiace, non ho trovato informazioni sufficienti nei miei documenti per rispondere a questa domanda."
    return {"documents": [], "question": question, "generation": generation}
    

# --- 3. Costruzione del Grafo ---

# Definiamo le variabili globali (retriever) che verranno inizializzate una sola volta.
retriever = None

def create_rag_graph(file_path: str):
    """
    Funzione principale che inizializza le risorse e costruisce il grafo.
    """
    global retriever
    
    # Inizializzazione del retriever (come nel vecchio file)
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever()
    print("Retriever inizializzato.")

    # Definizione del grafo
    workflow = StateGraph(GraphState)
    
    # Aggiunta dei nodi
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("fallback", fallback_answer)
    
    # Definizione del punto di partenza
    workflow.set_entry_point("retrieve")
    
    # Aggiunta dell'arco condizionale
    workflow.add_conditional_edges(
        "retrieve", # Parte dal nodo di recupero
        grade_documents, # La funzione che decide il percorso
        {
            "generate": "generate", # Se grade_documents ritorna "generate", vai al nodo "generate"
            "fallback": "fallback"   # Se ritorna "fallback", vai al nodo "fallback"
        }
    )
    
    # Aggiunta degli archi finali
    workflow.add_edge("generate", END)
    workflow.add_edge("fallback", END)
    
    # Compilazione del grafo
    app = workflow.compile()
    print("Grafo RAG compilato e pronto.")
    return app