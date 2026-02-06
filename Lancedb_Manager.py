import lancedb
from datetime import datetime, timedelta
import os
import pandas as pd
from sentence_transformers import SentenceTransformer

SENTENCE_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

class LawDB:
    def __init__(self, db_path="./laws/laws.db", restart_table=False, update_interval=90):
        self.__db = lancedb.connect(db_path)
        self.__model = SentenceTransformer(SENTENCE_MODEL)
        self.__table_name = "law_table"
        self.__Update=False
        if self.__table_name in self.__db.table_names():
            if restart_table:
                self.__db.drop_table(self.__table_name)
                self.__Update=True
            else:
                table_path = os.path.join(db_path, f"{self.__table_name}.lance")
                last_modified = datetime.fromtimestamp(os.path.getmtime(table_path))
                if datetime.now() > last_modified + timedelta(days=update_interval):
                    self.__db.drop_table(self.__table_name)
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
        if self.__table_name in self.__db.table_names():
            table = self.__db.open_table(self.__table_name)
            table.add(data)
        else:
            self.__db.create_table(self.__table_name, data)
        print(f"Database updated.")

    def search(self, query, limit=5):
        """Searches database for law documents that match the query."""
        table = self.__db.open_table(self.__table_name)
        query_vector = self.__model.encode(query).tolist()

        results = table.search(query_vector).limit(limit*3).to_pandas()
        results = results.drop_duplicates(subset=['text'])
        context_list = []
        for _, row in results.iterrows():
            full_text = str(row['text']).strip()
            law_id = row.get('id')
            formatted_text = f"id: {law_id} text: {full_text}"
            context_list.append(formatted_text)
        return "\n\n" + 30*"-" +"\n\n".join(context_list) + "\n\n" + 30*"-"
