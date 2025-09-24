"""
Modulo principale dell'applicazione RAG.
Carica le variabili d'ambiente, inizializza la catena RAG e gestisce l'interfaccia utente.
"""
import os
import sys
from dotenv import load_dotenv
from pathlib import Path

# Aggiungi il percorso della directory principale al path per l'importazione
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_chain import create_rag_chain
from src.rag_graph import create_rag_graph

# Aggiungiamo l'import per la nostra nuova funzione
from src.crew_builder import create_blog_post_crew

def main():
    load_dotenv()
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("Errore: La chiave API di Google non è stata configurata.")
        return

    print("\nBenvenuto al sistema di creazione di contenuti con CrewAI!")
    print("Digita 'esci' per terminare il programma.\n")
    
    while True:
        # Modifichiamo il prompt per l'utente
        topic = input("\nSu quale argomento vuoi che la crew scriva un post? ")
        
        if topic.lower() in ["esci", "exit", "quit", "q"]:
            print("Arrivederci!")
            break
        
        if not topic.strip():
            continue
        
        try:
            print("\n--- La Crew sta iniziando a lavorare... ---")
            
            # 1. Crea la crew con l'argomento fornito
            blog_crew = create_blog_post_crew(topic)
            
            # 2. Avvia il lavoro della crew. Il metodo si chiama kickoff()
            result = blog_crew.kickoff()
            
            print("\n\n--- Lavoro Terminato! Ecco il risultato finale: ---")
            print(result)

        except Exception as e:
            import traceback
            print(f"\nSi è verificato un errore durante il lavoro della crew:")
            traceback.print_exc()

if __name__ == "__main__":
    main()