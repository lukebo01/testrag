"""Utility per costruire una crew RAG basata su ``documento.txt``."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from crewai import Agent, Crew, Process, Task
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models.litellm import ChatLiteLLM
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


DEFAULT_DOCUMENT_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "documento.txt"
)
"""Percorso predefinito del corpus di conoscenza."""


_vector_store: Optional[FAISS] = None
"""Cache per il database vettoriale in memoria."""


def _ensure_vector_store(file_path: Optional[str] = None) -> FAISS:
    """Carica (o crea) il database vettoriale a partire dal documento sorgente."""

    global _vector_store

    if _vector_store is not None:
        return _vector_store

    source_path = Path(file_path) if file_path else DEFAULT_DOCUMENT_PATH
    if not source_path.exists():
        raise FileNotFoundError(
            f"Il file '{source_path}' non esiste. Assicurati che documento.txt sia disponibile."
        )

    loader = TextLoader(str(source_path), encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    _vector_store = FAISS.from_documents(chunks, embeddings)
    return _vector_store


def _retrieve_relevant_chunks(
    question: str, *, top_k: int = 4, file_path: Optional[str] = None
) -> List[Document]:
    """Restituisce i chunk più rilevanti per la domanda fornita."""

    vector_store = _ensure_vector_store(file_path)
    return vector_store.similarity_search(question, k=top_k)


def _format_context(docs: List[Document]) -> str:
    """Formatta i documenti recuperati in un blocco leggibile dagli agenti."""

    formatted_sections: List[str] = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source") or DEFAULT_DOCUMENT_PATH.name
        content = " ".join(doc.page_content.split())
        if len(content) > 800:
            content = f"{content[:800].rstrip()}…"
        formatted_sections.append(
            f"[Estratto {idx} - {Path(source).name}]\n{content}"
        )

    return "\n\n".join(formatted_sections) if formatted_sections else "(Nessun contesto recuperato)"


@dataclass
class RAGCrewRun:
    """Semplice contenitore per la crew e il contesto recuperato."""

    crew: Crew
    context: str
    documents: List[Document]


def create_rag_crew(
    question: str,
    *,
    top_k: int = 4,
    file_path: Optional[str] = None,
) -> RAGCrewRun:
    """Crea una crew di agenti specializzati per rispondere usando ``documento.txt``."""

    if not question or not question.strip():
        raise ValueError("La domanda non può essere vuota.")

    docs = _retrieve_relevant_chunks(question, top_k=top_k, file_path=file_path)
    context_block = _format_context(docs)

    llm = ChatLiteLLM(
        model="gemini/gemini-1.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY"),
    )

    researcher = Agent(
        role="Analista documentale",
        goal="Estrarre fatti accurati e rilevanti dagli estratti forniti.",
        backstory=(
            "Veterano nella lettura ravvicinata di report tecnici. Sai distinguere i dettagli"
            " cruciali dalle informazioni marginali e annoti sempre le fonti."
        ),
        allow_delegation=False,
        verbose=True,
        llm=llm,
    )

    synthesizer = Agent(
        role="Coordinatore della conoscenza",
        goal="Costruire una bozza coerente collegando i fatti al quesito dell'utente.",
        backstory=(
            "Progetti percorsi RAG da anni. Sai sintetizzare velocemente grandi moli di"
            " informazioni e mantenerle coerenti con gli obiettivi."
        ),
        allow_delegation=False,
        verbose=True,
        llm=llm,
    )

    writer = Agent(
        role="Redattore tecnico",
        goal="Produrre una risposta articolata, citando le parti rilevanti del documento.",
        backstory=(
            "Hai il compito di trasformare analisi tecniche in testo chiaro e fluido per il"
            " lettore finale, mantenendo riferimenti alle fonti."
        ),
        allow_delegation=False,
        verbose=True,
        llm=llm,
    )

    research_task = Task(
        description=(
            "Analizza gli estratti recuperati dal documento 'documento.txt' per rispondere"
            f" alla domanda dell'utente. Domanda: {question}\n\n"
            "Per ogni estratto identifica:\n"
            "- fatti o numeri rilevanti\n"
            "- eventuali citazioni utili\n"
            "- il motivo per cui il passaggio aiuta a rispondere\n\n"
            "Formato richiesto: elenco numerato con voce, riassunto e riferimento"
            " (es. [Estratto 1])." 
            f"\n\nEstratti disponibili:\n{context_block}"
        ),
        expected_output=(
            "Elenco numerato di fatti chiave con riferimento all'estratto e breve spiegazione"
        ),
        agent=researcher,
    )

    synthesis_task = Task(
        description=(
            "Usa l'elenco dei fatti per costruire una bozza di risposta strutturata."
            f" Domanda a cui rispondere: {question}\n\n"
            "La bozza deve avere:\n"
            "- una breve introduzione\n"
            "- 2-3 paragrafi che collegano i fatti al quesito\n"
            "- punti chiave che mostrano come ogni estratto supporta la risposta\n"
            "Mantieni i riferimenti agli estratti con la sintassi [Estratto X]."
        ),
        expected_output=(
            "Testo strutturato (introduzione, corpo, punti chiave) con riferimenti alle fonti"
        ),
        agent=synthesizer,
        context=[research_task],
    )

    writing_task = Task(
        description=(
            "Trasforma la bozza nella risposta finale per l'utente. Utilizza un tono chiaro,"
            " professionale e in lingua italiana.\n\n"
            f"Domanda originale: {question}\n"
            "Requisiti:\n"
            "- rispondi direttamente alla domanda nelle prime due frasi\n"
            "- integra i dettagli dai punti del coordinatore\n"
            "- mantieni i riferimenti [Estratto X] dove hai preso le informazioni\n"
            "- chiudi con un suggerimento pratico o una frase conclusiva."
        ),
        expected_output="Risposta finale fluida e completa con riferimenti alle fonti",
        agent=writer,
        context=[synthesis_task],
    )

    crew = Crew(
        agents=[researcher, synthesizer, writer],
        tasks=[research_task, synthesis_task, writing_task],
        process=Process.sequential,
        verbose=True,
    )

    return RAGCrewRun(crew=crew, context=context_block, documents=docs)