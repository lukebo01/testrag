# Progetto testRAG

## Descrizione
Questo progetto ha l'obiettivo di esplorare e testare tecniche di Retrieval-Augmented Generation (RAG) per migliorare la qualità delle risposte generate da modelli di intelligenza artificiale tramite l'integrazione di fonti di conoscenza esterne.

## Struttura del progetto
- **/data**: contiene i dataset utilizzati per il retrieval.
- **/src**: codice sorgente per il retrieval, la generazione e l'integrazione dei modelli.
- **/notebooks**: notebook Jupyter per esperimenti e analisi.
- **/results**: risultati degli esperimenti.

## Requisiti
- Python 3.10+
- PyTorch
- Transformers (HuggingFace)
- FAISS
- Jupyter

## Installazione
```bash
git clone https://github.com/lukebo01/testRAG.git
cd testRAG
pip install -r requirements.txt
```

## Utilizzo
1. Prepara i dati nella cartella `/data`.
2. Esegui lo script:
    ```bash
    python src/main.py
    ```

## Contributi
Sono benvenuti suggerimenti, segnalazioni di bug e pull request.

## Licenza
Questo progetto è distribuito sotto licenza MIT.