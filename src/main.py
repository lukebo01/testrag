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


def main():
    """
    Funzione principale dell'applicazione.
    Carica le variabili d'ambiente, inizializza la catena RAG e gestisce l'interfaccia CLI.
    """
    # Carica le variabili d'ambiente
    load_dotenv()
    
    # Verifica che la chiave API sia presente
    if not os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") == "inserisci_la_tua_chiave_qui":
        print("Errore: La chiave API di Google non è stata configurata.")
        print("Per favore, aggiorna il file .env con la tua chiave API.")
        return
    
    # Definisci il percorso al documento
    base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    document_path = base_dir / "data" / "documento.txt"
    
    if not document_path.exists():
        print(f"Errore: Il file {document_path} non esiste.")
        return
    
    print("Inizializzazione della catena RAG in corso...")
    
    try:
        # Crea la catena RAG
        rag_chain = create_rag_chain(str(document_path))
        
        print("\nBenvenuto al sistema RAG semplice!")
        print("Puoi fare domande sul documento caricato.")
        print("Digita 'esci' per terminare il programma.\n")
        
        # Loop principale dell'interfaccia CLI
        while True:
            user_input = input("\nLa tua domanda: ")
            
            if user_input.lower() in ["esci", "exit", "quit", "q"]:
                print("Arrivederci!")
                break
            
            if not user_input.strip():
                print("Per favore, inserisci una domanda.")
                continue
            
            # Esegui la catena con la domanda dell'utente
            try:
                response = rag_chain.invoke({"question": user_input})
                print("\nRisposta:")
                print(response)
            except Exception as e:
                print(f"Si è verificato un errore durante l'elaborazione della domanda: {e}")
                
    except Exception as e:
        print(f"Errore nell'inizializzazione della catena RAG: {e}")


if __name__ == "__main__":
    main()