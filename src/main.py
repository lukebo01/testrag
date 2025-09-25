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

from src.crew_builder import create_rag_crew

def main():
    load_dotenv()
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("Errore: La chiave API di Google non è stata configurata.")
        return

    print("\nBenvenuto al sistema RAG basato su CrewAI!")
    print("Formula domande sul contenuto di 'documento.txt'. Digita 'esci' per terminare.\n")
    
    while True:
        question = input("\nQual è la tua domanda sul documento? ")
        
        if question.lower() in ["esci", "exit", "quit", "q"]:
            print("Arrivederci!")
            break
        
        if not question.strip():
            continue
        
        try:
            print("\n--- Recupero del contesto dal documento... ---")
            rag_run = create_rag_crew(question)

            if rag_run.context:
                print("\nEstratti principali utilizzati dalla crew:\n")
                print(rag_run.context)

            print("\n--- La crew sta elaborando la risposta... ---")
            result = rag_run.crew.kickoff()

            print("\n\n--- Risposta finale ---")
            print(result)

        except FileNotFoundError as err:
            print(f"\nErrore: {err}")
            break
        except Exception:
            import traceback
            print(f"\nSi è verificato un errore durante il lavoro della crew:")
            traceback.print_exc()

if __name__ == "__main__":
    main()