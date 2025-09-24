# src/crew_builder.py

from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models.litellm import ChatLiteLLM
from crewai import Agent, Task, Crew, Process
import os

def create_blog_post_crew(topic: str):
    """
    Crea e configura una Crew per ricercare e scrivere un blog post su un dato argomento.
    """
    # Inizializziamo il modello LLM che tutti gli agenti useranno
    llm = ChatLiteLLM(
        model="gemini/gemini-1.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY")
    )

    # --- 1. Definizione degli Agenti ---

    # Agente 1: Il Ricercatore
    researcher = Agent(
        role="Ricercatore Tecnologico Senior",
        goal=f"Trovare le informazioni più rilevanti, accurate e recenti sull'argomento: {topic}",
        backstory=(
            "Sei un esperto ricercatore con anni di esperienza nell'analizzare documentazione tecnica e articoli scientifici. "
            "Sei abile nel distillare informazioni complesse in punti chiave facili da capire."
        ),
        verbose=True,  # Mostra i "pensieri" dell'agente mentre lavora
        allow_delegation=False,
        llm=llm
    )

    # Agente 2: Lo Scrittore
    writer = Agent(
        role="Scrittore Tecnico Esperto",
        goal=f"Scrivere un blog post informativo e coinvolgente sull'argomento: {topic}, basandosi sui punti forniti dal ricercatore.",
        backstory=(
            "Sei uno scrittore rinomato, famoso per la tua capacità di spiegare argomenti complessi in modo chiaro e conciso. "
            "Trasformi elenchi puntati di fatti in narrazioni fluide e interessanti."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    # --- 2. Definizione dei Task ---

    # Task 1: Ricerca
    research_task = Task(
        description=(
            f"Analizza l'argomento '{topic}'. "
            "Identifica i 5 concetti chiave, le loro definizioni e un esempio pratico per ciascuno. "
            "Il tuo output finale deve essere un elenco puntato ben strutturato che lo scrittore possa usare facilmente."
        ),
        expected_output="Un elenco puntato contenente 5 punti chiave, ognuno con una definizione e un esempio.",
        agent=researcher
    )

    # Task 2: Scrittura
    # Nota il parametro 'context'. Dice a questo task di usare l'output del 'research_task'.
    writing_task = Task(
        description=(
            "Usa i punti chiave forniti dal ricercatore per scrivere un blog post di circa 300 parole. "
            "Il post deve avere un'introduzione, uno sviluppo che spiega i punti chiave, e una breve conclusione. "
            "Lo stile deve essere professionale ma accessibile."
        ),
        expected_output=f"Un articolo di blog ben formattato di circa 300 parole sull'argomento: {topic}.",
        agent=writer,
        context=[research_task] # <-- LA MAGIA DELLA COLLABORAZIONE!
    )

    # --- 3. Creazione della Crew ---

    blog_crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        process=Process.sequential,  # I task verranno eseguiti in ordine
        verbose=True # Mostra log dettagliati del processo della crew
    )
    
    return blog_crew