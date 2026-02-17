import lancedb
from datetime import datetime, timedelta
import os
import pandas as pd
from lancedb import Table
from sentence_transformers import SentenceTransformer

SENTENCE_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
LAW_TABLE = "law_table"
PAGE_TABLE = "page_table"
THOUGHT_TABLE = "thought_table"
SEARCH_TABLE = "search_table"

class BrainDB:
    __TABLES = [LAW_TABLE, PAGE_TABLE, THOUGHT_TABLE, SEARCH_TABLE]

    def __init__(self, db_path="./brain/memory.db", restart_table=False, update_interval=90):
        self.__db = lancedb.connect(db_path)
        self.__model = SentenceTransformer(SENTENCE_MODEL)
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

    def add_law(self, paragraphs):
        """Paragraphs: List of dicts {'id': 'SGB_1 §1', 'content': 'text'}"""
        if not self.__Update:
            return
        data = []
        print(f"Vectorizing {len(paragraphs)} paragraphs...")
        for p in paragraphs:
            embedding = self.__model.encode(p['content']).tolist()
            data.append({
                'vector': embedding,
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

    def add_page_json(self, document_path: str, json_str: str, page_id: int):
        """Adds a page's json to the database."""
        print(f"Adding page {page_id} as JSON to database...")
        data = []
        page_data = pd.read_json(json_str)
        embedding = self.__model.encode(f"Pfad:{document_path}|page:{page_id}|Argument:{page_data['Inhalt']['Argument']}").tolist()
        data.append({'vector': embedding,
                     'page': page_id,
                     'argument': page_data['Inhalt']['Argument'],
                     'laws': page_data['Inhalt']['Gesetze'],
                     'law_areas': page_data['Inhalt']['Rechtsgebiet'],
                     'links': page_data['Inhalt']['Verweise'],
                     'deadlines': page_data['Inhalt']['Fristen'],
                     'json': json_str
                     })
        if PAGE_TABLE in self.__db.table_names():
            table = self.__db.open_table(PAGE_TABLE)
            table.add(data)
        else:
            self.__db.create_table(PAGE_TABLE, data)

    def search(self, query, table_name = "", limit=5):
        """Searches databases that match the query. Eather searches a specific table or every table in the database."""
        query_vector = self.__model.encode(query).tolist()
        results = pd.DataFrame()
        if table_name != "":
            if table_name in self.__db.table_names():
                table = self.__db.open_table(table_name)
                results = table.search(query_vector).limit(limit).to_pandas()

            else:
                return "Table not found."
        else:   # search every table if table-name empty
            for table_name in self.__db.table_names():
                table = self.__db.open_table(table_name)
                results = table.search(query_vector).limit(limit).to_pandas()
                results = results.drop_duplicates(subset=['text'])

        context_list = []
        if table_name == LAW_TABLE:
            results = results.drop_duplicates(subset=['text'])
            for _, row in results.iterrows():
                full_text = str(row['text']).strip()
                law_id = row.get('id')
                formatted_text = f"id: {law_id} text: {full_text}"
                context_list.append(formatted_text)
            return "\n\n" + 30*"-" +"\n\n".join(context_list) + "\n\n" + 30*"-"
        elif table_name == PAGE_TABLE:
            results = results.drop_duplicates(subset=['argument', 'laws', 'page'])
            for _, row in results.iterrows():
                formatted_text = f"page: {row.get('page')} laws: {row.get('laws')} argument: {row.get('argument')}"
                context_list.append(row.get(formatted_text))
            return "\n\n" + 30 * "-" + "\n\n".join(context_list) + "\n\n" + 30 * "-"
        else:
            return "No table found."
