from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict
import os

class VectorStore:
    def __init__(self, config: Dict):
        self.embeddings = OpenAIEmbeddings(
            model=config["vector_stores"]["legal"]["embedding_model"],
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.stores = {}
        self.config = config
        
    def create_index(self, name: str, docs_path: str):
        # 向量索引
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["retrieval"]["chunk_size"],
            chunk_overlap=self.config["retrieval"]["chunk_overlap"]
        )
        local_dir = os.path.join(docs_path, "faiss")
        # if exit, not load
        if os.path.exists(local_dir):
            store = FAISS.load_local(folder_path=local_dir, embeddings=self.embeddings, index_name=name, allow_dangerous_deserialization=True)
            self.stores[name] = store
            return
        documents = []
        for file in os.listdir(docs_path):
            with open(os.path.join(docs_path, file), 'r') as f:
                doc_chunks = text_splitter.split_text(f.read())
                for chunk in doc_chunks:
                    documents.append(
                        Document(
                            page_content=chunk,
                            metadata={"source": file, "index_name": name}
                        )
                    )

        store = FAISS.from_documents(documents, self.embeddings)
        os.makedirs(local_dir, exist_ok=True)
        self.stores[name] = store
        store.save_local(folder_path=local_dir, index_name=name)
        
    def search(self, name: str, query: str, k: int = None) -> List[str]:
        if k is None:
            k = self.config["retrieval"]["top_k"]
        
        store = self.stores.get(name)
        if not store:
            raise KeyError(f"Index {name} not found")
        docs = store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs] 
