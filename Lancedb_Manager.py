from io import StringIO

import lancedb
from datetime import datetime, timedelta
import os
import pandas as pd
from lancedb import Table
import ollama
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

SENTENCE_MODEL = "nomic-embed-text:latest"
LAW_TABLE = "law_table"
DOC_TABLE = "doc_table"
THOUGHT_TABLE = "thought_table"
SEARCH_TABLE = "search_table"

class BrainDB:
    __TABLES = [LAW_TABLE, DOC_TABLE]

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
        response = ollama.embed(
            model=SENTENCE_MODEL + "_8k",
            input=text
        )
        return response.embeddings

    def add_law(self, paragraphs):
        """Paragraphs: List of dicts {'id': 'SGB_1 §1', 'content': 'text'}"""
        if not self.__Update:
            return
        data = []
        print(f"Vectorizing {len(paragraphs)} paragraphs...")
        for p in paragraphs:
            text_embedding = self.__get_ollama_embedding(p['content'])
            id_embedding = self.__get_ollama_embedding(p['id'])
            data.append({
                'text_vector': text_embedding,
                'id_vector': id_embedding,
                'text': p['content'],
                'id': p['id']
            })

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
        if LAW_TABLE in self.__db.table_names():
            table = self.__db.open_table(LAW_TABLE)
            query_vector = self.__get_ollama_embedding(law_id).tolist()
            results = table.search(query_vector, vector_column_name='id_vector').limit(limit).to_pandas()
            context_list = []
            for _, row in results.iterrows():
                full_text = str(row['text']).strip()
                formatted_text = f"id: {law_id} text: {full_text}"
                context_list.append(formatted_text)
            return "\n\n" + 30 * "-" + "\n\n".join(context_list) + "\n\n" + 30 * "-"
        return "Law not found."

    def search(self, query, table_name = "", limit=5):
        """Searches databases that match the query. Eather searches a specific table or every table in the database."""
        query_vector = self.__get_ollama_embedding(query).tolist()
        results = pd.DataFrame()
        if table_name != "":
            if table_name in self.__db.table_names():
                table = self.__db.open_table(table_name)
                if table_name == LAW_TABLE:
                    results = table.search(query_vector, vector_column_name="text_vector").limit(limit).to_pandas()
                else:
                    results = table.search(query_vector).limit(limit).to_pandas()
            else:
                return "Table not found."
        else:   # TODO: search every table if table-name empty
            return "Table needs to be specified."

        context_list = []
        if table_name == LAW_TABLE:
            results = results.drop_duplicates(subset=['text'])
            for _, row in results.iterrows():
                full_text = str(row['text']).strip()
                law_id = row.get('id')
                formatted_text = f"id: {law_id} text: {full_text}"
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
