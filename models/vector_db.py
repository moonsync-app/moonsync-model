from dataclasses import dataclass

from typing import List

from pinecone import Pinecone

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore


@dataclass
class VectorDB:
    """
    Vector database used in the application.
    """

    api_key: str

    def __post_init__(self):
        self.pc = Pinecone(api_key=self.api_key)

    def get_vector_indexes(self, indexes: List[str]) -> List[VectorStoreIndex]:

        pc_indexes = [self.pc.Index(name=index) for index in indexes]

        vector_indexes = [
            VectorStoreIndex.from_vector_store(PineconeVectorStore(pc_index))
            for pc_index in pc_indexes
        ]

        return vector_indexes
