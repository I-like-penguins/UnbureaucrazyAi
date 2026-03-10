from io import StringIO

import lancedb
import logging
from datetime import datetime, timedelta
import os
import pandas as pd
import numpy as np
from lancedb import Table
import ollama
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

SENTENCE_MODEL = "nomic-embed-text:latest"
LAW_TABLE = "law_table"
DOC_TABLE = "doc_table"
THOUGHT_TABLE = "thought_table"
SEARCH_TABLE = "search_table"

OVERLAP_SIZE = 300

class BrainDB:
    __TABLES = [LAW_TABLE, DOC_TABLE]

    logging.basicConfig(
        filename='law_import_debug.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    def __setup_custom_model(self,base_model=SENTENCE_MODEL, new_name=SENTENCE_MODEL + "_8k"):
        # Modelfile-Inhalt definieren
        modelfile_content = f'''
        FROM {base_model}
        PARAMETER num_ctx 8192
        '''
        # Modell in Ollama registrieren
        ollama.create(model=new_name, from_=base_model, parameters={'num_ctx': 8192})
        print(f"Modell '{new_name}' mit 8192 Kontext-Limit wurde erstellt.")

    def __init__(self, db_path="./brain/memory.db", restart_table=False, update_interval=90):
        self.__db = lancedb.connect(db_path)
        self.__setup_custom_model()
        #self.__model = SentenceTransformer(SENTENCE_MODEL)
        self.__table_name = ""
        self.__Update=False
        if LAW_TABLE in self.__db.table_names():
            if restart_table:
                self.__db.drop_table(LAW_TABLE)
                self.__Update=True
            else:
                table_path = os.path.join(db_path, f"{LAW_TABLE}.lance")
                last_modified = datetime.fromtimestamp(os.path.getmtime(table_path))
                if datetime.now() > last_modified + timedelta(days=update_interval):
                    self.__db.drop_table(LAW_TABLE)
                    self.__Update=True
                print(f"Using existing database at {db_path}.")
        else:
            self.__Update=True
            table_path = os.path.join(db_path, f"{LAW_TABLE}.lance")
            print(f"Creating new database at {table_path}.")

    def __get_ollama_embedding(self, text: str):
        if not text or not text.strip():
            return None

        response = ollama.embed(
            model=SENTENCE_MODEL + "_8k",
            input=text
        )
        return response.embeddings[0]

    def add_law(self, paragraphs):
        """Paragraphs: List of dicts {'id': 'SGB_1 §1', 'content': 'text'}"""
        if not self.__Update:
            return

        safety_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=OVERLAP_SIZE,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        data = []
        logging.info(f"--- Starte Import von {len(paragraphs)} Paragraphen ---")

        print(f"Vectorizing {len(paragraphs)} paragraphs (in chunks)...")
        for p in paragraphs:
            content = "".join(char for char in p['content'] if char.isprintable())      # clean text of useless characters
            logging.info(f"Verarbeite {p["id"]} (Gesamtlänge: {len(content)} Zeichen)")

            id_embedding = np.array(self.__get_ollama_embedding(p['id']), dtype=np.float32).flatten().tolist()
            chunks = safety_splitter.split_text(content) if len(content) > 12000 else [content]
            for i, chunk in enumerate(chunks):
                logging.info(f"  -> Chunk {i}: {len(chunk)} Zeichen")
                try:
                    raw_embedding = self.__get_ollama_embedding(chunk)
                    if raw_embedding is None:
                        continue
                    text_embedding = np.array(raw_embedding, dtype=np.float32).flatten().tolist()
                    data.append({
                        'id': p['id'],
                        'text': chunk,
                        'text_vector': text_embedding,
                        'id_vector': id_embedding,
                        'chunk_index': i
                    })
                    print(f"Vectorized paragraph {p['id']}: '{content}'")
                except Exception as e:
                    print(f"Error vectorizing paragraph {p['id']} Teil {i}: {e}")
                    logging.error(f"!!! FEHLER in {p['id']} (Chunk {i}): {str(e)}")
                    logging.error(f"Problematischer Textauszug: {chunk[:200]}...")
                    continue
        # create table or add data to existing table
        if LAW_TABLE in self.__db.table_names():
            table = self.__db.open_table(LAW_TABLE)
            table.add(data)
        else:
            self.__db.create_table(LAW_TABLE, data)
        print(f"Database updated.")

    def add_document(self, document_path: str):         # TODO: Maybe do it later: Adding entire documents to the database for linking
        """Adds a page's text to the database."""
        print(f"Adding page {document_path} as text to database...")
        with open(document_path, "r", encoding="utf-8") as f:
            text = f.read()
        embedding = self.__get_ollama_embedding(text).tolist()
        data = [{'vector': embedding, 'text': text}]
        if DOC_TABLE in self.__db.table_names():
            table = self.__db.open_table(DOC_TABLE)
            table.add(data)
            print(f"Document {str(document_path)} added to database.")
            return True
        else:
            self.__db.create_table(DOC_TABLE, data)
            print(f"Page {document_path} added to database.")
            return True

    def add_page_json(self, document_path: str, json_str: str, page_id: int):
        """Adds a page's json to the database."""
        print(f"Adding page {page_id} as JSON to database...")
        data = []
        try:
            page_data = pd.read_json(StringIO(json_str))
            embedding = self.__get_ollama_embedding(f"Pfad:{document_path}|page:{page_id}|Argument:{page_data['Argument']}").tolist()
            data.append({'vector': embedding,
                         'page': page_id,
                         'argument': page_data['Argument'] if not None else "",
                         'laws': page_data['Gesetze'] if not None else [],
                         'law_areas': page_data['Rechtsgebiet'] if not None else [],
                         'links': page_data['Verweise'] if not None else [],
                         'deadlines': page_data['Fristen'] if not None else [],
                         'json': json_str
                         })
        except ValueError as e:
            print(f"Error reading JSON: {e}")
            embedding = self.__get_ollama_embedding(f"Pfad:{document_path}|page:{page_id}|{json_str}").tolist()
            data.append({'vector': embedding,
                         'page': page_id,
                         'argument': "",
                         'laws': [],
                         'law_areas': [],
                         'links': [],
                         'deadlines': [],
                         'json': json_str})
        if DOC_TABLE in self.__db.table_names():
            table = self.__db.open_table(DOC_TABLE)
            table.add(data)
        else:
            self.__db.create_table(DOC_TABLE, data)

    def search_law_id(self, law_id, limit=1):
        """Searches the database for a law with the given id."""
        if law_id == "" or law_id is None:
            return "Keine ID angegeben."
        context_list = []
        query_vector_raw = self.__get_ollama_embedding(law_id)
        query_vector = np.array(query_vector_raw, dtype=np.float32).flatten()
        if LAW_TABLE in self.__db.table_names():
            table = self.__db.open_table(LAW_TABLE)
            results = table.search(query_vector, vector_column_name="id_vector").limit(limit).to_pandas()
            if results.empty:
                return "Keine relevanten Gesetze gefunden."
            for _, result in results.iterrows():

                full_text = self.get_full_paragraph(result['id'])
                formatted_text = f"id: {law_id} text: {full_text}"
                context_list.append(formatted_text)
            return "\n\n" + 30 * "-" + "\n\n".join(context_list) + "\n\n" + 30 * "-"
        return "Law not found."

    def get_full_paragraph(self, paragraph_id):
        table = self.__db.open_table(LAW_TABLE)

        chunks = table.search() \
            .where(f"id = '{paragraph_id}'") \
            .to_list()

        if not chunks:
            print(f"Keine Daten für {paragraph_id} gefunden.")
            return ""

        chunks.sort(key=lambda x: x.get('chunk_index', 0))

        full_text = " ".join([c['text'] for c in chunks])
        return full_text

    def search(self, query, table_name = "", limit=5):
        """Searches databases that match the query. Eather searches a specific table or every table in the database."""
        query_vector_raw = self.__get_ollama_embedding(query)
        query_vector = np.array(query_vector_raw, dtype=np.float32).flatten()
        results = pd.DataFrame()
        target_id_list = []
        if table_name != "":
            if table_name in self.__db.table_names():
                table = self.__db.open_table(table_name)
                if table_name == LAW_TABLE:
                    results = table.search(query_vector, vector_column_name="text_vector").limit(limit).to_pandas()
                    print("DEBUG")
                    print(results)
                    if results.empty:
                        return "Keine relevanten Gesetze gefunden."
                    for _, result in results.iterrows():
                        target_id_list.append(result['id'])
                else:
                    results = table.search(query_vector).limit(limit).to_pandas()
            else:
                return "Table not found."
        else:   # TODO: search every table if table-name empty
            return "Table needs to be specified."

        context_list = []
        if table_name == LAW_TABLE:
            ids= set(target_id_list)
            for p_id in ids:
                full_text = self.get_full_paragraph(p_id)
                formatted_text = f"id: {p_id} text: {full_text}"
                context_list.append(formatted_text)
            return "\n\n" + 30*"-" +"\n\n".join(context_list) + "\n\n" + 30*"-"
        elif table_name == DOC_TABLE:
            for _, row in results.iterrows():
                if row.get('argument') != "":
                    formatted_text = f"page: {row.get('page')} laws: {row.get('laws')} argument: {row.get('argument')}"
                else:
                    formatted_text = f"page: {row.get('page')} json: {row.get('json')}"
                context_list.append(formatted_text)

            return "\n\n" + 30 * "-" + "\n\n".join(context_list) + "\n\n" + 30 * "-"
        else:
            return "No table found."
