import lancedb
import pandas as pd
from sentence_transformers import SentenceTransformer

SENTENCE_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

class LawDB:
    def __init__(self, db_path="./laws/laws.db"):
        self.__db = lancedb.connect(db_path)
        self.__model = SentenceTransformer(SENTENCE_MODEL)
        self.__table_name = "law_table"

    def add_law(self, paragraphs):
        """Paragraphs: List of dicts {'id': 'SGB_1 §1', 'content': 'text'}"""
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

        results = table.search(query_vector).limit(limit).to_pandas()
        context_list = []
        for _, row in results.iterrows():
            full_text = str(row['text']).strip()
            law_id = row.get('id')
            formatted_text = f"id: {law_id} text: {full_text}"
            context_list.append(formatted_text)
        return "\n\n" + 30*"-" +"\n\n".join(context_list) + "\n\n" + 30*"-"

    def get_query_str_text(self, query, limit=5):
        tmp_query = self.search(query, limit)
        return f"id: {tmp_query['id']} text: {(tmp_query['text'])}"
