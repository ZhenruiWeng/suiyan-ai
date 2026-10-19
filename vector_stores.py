import config_data as config
from langchain_chroma import Chroma


class VectorStoreService(object):
    def __init__(self, embedding):

        self.embedding = embedding

        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )

    # 返回向量检索器，方便加入chain
    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k":config.similarity_threshold})





