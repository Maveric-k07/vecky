from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from sqlalchemy import Column, create_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import Field, SQLModel, select
from pgvector.sqlalchemy import Vector
import os
from uuid import UUID, uuid4
import time


def measure_time(function):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = function(*args, **kwargs)
        execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        return result, execution_time

    return wrapper


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def encode(self, text: Union[str, List[str]]) -> List[float]:
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass


class SentenceTransformerModel(BaseEmbeddingModel):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode(self, text: Union[str, List[str]]) -> List[float]:
        return self.model.encode(text).tolist()

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name

    def encode(self, text: Union[str, List[str]]) -> List[float]:
        if isinstance(text, str):
            text = [text]
        response = self.client.embeddings.create(input=text, model=self.model_name)
        return response.data[0].embedding

    @property
    def dimension(self) -> int:
        return 1536  # OpenAI's embedding dimension


class EmbeddingModelType(Enum):
    SENTENCE_TRANSFORMER = "sentence_transformer"
    OPENAI = "openai"


class SimilarityMetric(Enum):
    EUCLIDEAN = 1
    COSINE = 2


class Document(SQLModel, table=True):
    __tablename__ = "vecky-documents"

    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    collection_name: str = Field(default="vecky", index=True)
    content: str
    embedding: List[float] = Field(
        sa_column=Column(Vector(1536))
    )  # Use the largest dimension


def ensure_embedding_dimension(embedding, target_dim=1536):
    if len(embedding) > target_dim:
        return embedding[:target_dim]
    elif len(embedding) < target_dim:
        return embedding + [0.0] * (target_dim - len(embedding))
    return embedding


class VectorDatabase:
    DEFAULT_COLLECTION = "vecky"

    def __init__(
        self,
        db_url: str,
        embedding_model: BaseEmbeddingModel,
        similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
    ):
        self.engine = create_engine(db_url)
        SQLModel.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.embedding_model = embedding_model
        self.similarity_metric = similarity_metric
        self._set_similarity_function()

    def _set_similarity_function(self):
        if self.similarity_metric == SimilarityMetric.EUCLIDEAN:
            self._sim_func = lambda x, y: x.l2_distance(y)
        elif self.similarity_metric == SimilarityMetric.COSINE:
            self._sim_func = lambda x, y: 1 - x.cosine_distance(y)
        else:
            raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")

    @classmethod
    def from_documents(
        cls,
        db_url: str,
        docs: List[str],
        embedding_model_type: EmbeddingModelType,
        model_name: str,
        collection_name: Optional[str] = None,
        similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
    ):
        if embedding_model_type == EmbeddingModelType.SENTENCE_TRANSFORMER:
            embedding_model = SentenceTransformerModel(model_name)
        elif embedding_model_type == EmbeddingModelType.OPENAI:
            embedding_model = OpenAIEmbeddingModel(model_name)
        else:
            raise ValueError(
                f"Unsupported embedding model type: {embedding_model_type}"
            )

        instance = cls(db_url, embedding_model, similarity_metric)
        instance.add_documents(docs, collection_name)
        return instance

    def add_documents(self, docs: List[str], collection_name: Optional[str] = None):
        collection_name = collection_name or self.DEFAULT_COLLECTION
        session = self.Session()
        try:
            for doc in docs:
                embedding = self.embedding_model.encode(doc)
                embedding = ensure_embedding_dimension(embedding=embedding)
                db_doc = Document(
                    content=doc,
                    embedding=embedding,
                    collection_name=collection_name,
                )
                db_doc = Document.model_validate(db_doc)
                session.add(db_doc)
            session.commit()
        finally:
            session.close()

    @measure_time
    def search(self, query: str, collection_name: Optional[str] = None, k: int = 5):
        query_embedding = self.embedding_model.encode(query)
        query_embedding = ensure_embedding_dimension(query_embedding)

        session = self.Session()
        try:
            similarity_function = self._sim_func(
                Document.embedding, query_embedding
            ).label("similarity")
            stmt = select(
                Document.content,
                similarity_function,
            ).order_by(similarity_function)

            if collection_name:
                stmt = stmt.filter(Document.collection_name == collection_name)

            stmt = stmt.limit(k)
            results = session.execute(stmt).all()
            return [r.content for r in results], [r.similarity for r in results]
        finally:
            session.close()

    def __repr__(self):
        session = self.Session()
        total_docs = session.query(Document).count()
        collections = session.query(Document.collection_name.distinct()).count()
        return (
            f"VectorDatabase(total_documents={total_docs}, "
            f"num_collections={collections}, "
            f"embedding_model={self.embedding_model.__class__.__name__}, "
            f"similarity_metric={self.similarity_metric.name})"
        )

    def list_collections(self):
        session = self.Session()
        try:
            collections = session.query(Document.collection_name.distinct()).all()
            return [c[0] for c in collections]
        finally:
            session.close()

    def get_collection_size(self, collection_name=None):
        collection_name = collection_name or self.DEFAULT_COLLECTION
        session = self.Session()
        try:
            return (
                session.query(Document)
                .filter(Document.collection_name == collection_name)
                .count()
            )
        finally:
            session.close()


if __name__ == "__main__":
    print("Loading embedding model...")

    sentence_transformer_model = "sentence-transformers/all-MiniLM-L12-v2"
    openai_embedding_model = "text-embedding-3-small"
    db_url = "postgresql://postgres:password@localhost:5433/vector_db"

    document = """She found the old lockbox buried under the floorboards, its metal surface tarnished with age. With trembling hands, she twisted the rusty latch and gasped at its contents - a stack of letters tied with faded ribbon, each bearing her grandmother's elegant script from decades past.
The rain drummed an insistent melody on the rooftop as Tim stared out the window. He longed for adventure, for escape from this monotonous life. A flash of movement caught his eye - a folded paper boat drifting down the street, bobbing over puddles with reckless abandon.
In the stillness of the autumn forest, a single crimson leaf drifted lazily downward, twirling and dancing through gilded shafts of sunlight. Lila watched it in silent reverie until it finally came to rest atop her upturned palm - a transient beauty now preserved in memory.
The circus had left town weeks ago, but Timmy swore he could still hear the faint calling of the ringmaster's voice. He followed it into an overgrown field where the tattered remains of a striped tent lay haunting the weeds, casting serpentine shadows in the moonlight.
Emma pressed her palm against the chilled windowpane, mesmerized by the fragile, lacy patterns that the frost had etched across the glass. She yearned to reach through, to trace those intricate whorls and skim her fingertips through the frozen fractals glittering in the winter dawn.
Beneath the inky cloak of midnight, the cove's usually placid waters had turned tumultuous and thick with brutish undercurrents. Yet the old fisherman calmly launched his skiff to wrestle with the gaping maw of the tempest, as if summoned by some primordial calling of the sea.
The attic was a forgotten sanctuary where sunbeams danced through clouds of golden dust. Among the clutter, Alice discovered an intricate jewel-encrusted dagger - surely just an antique stage prop. But when she grasped its ornate hilt, her mind was gripped by visions of torchlit battles.
No one could quite remember when the old railway handcar had been abandoned beside the rusting tracks. But every full moon, the neighborhood kids gathered to take turns pumping the stubborn lever, half-hoping the stubborn thing might finally roll to life with heavy metallic creaks.
Though Jim's body had failed him long ago, he blessed the twilight years that had sharpened his mind's eye. From his worn rocking chair, he constantly reimagined the dance of the clouds into fantastic beasts and sailing ships bound for lands unseen by weathered mariners.
The ramshackle cottage sat placid amid the gradual encroachment of the marsh thickets. Lillian would never glimpse the rambling rose vines that had wound the crumbling chimney, but she could smell their haunting sweet perfume drifting through the skeins of mist swirling beneath the moon's watchful gaze."""

    collection_name = "park-stories-3"

    docs = document.split("\n")
    print(f"Creating vector database for {len(docs)} documents...")

    vdb = VectorDatabase.from_documents(
        db_url=db_url,
        docs=docs,
        embedding_model_type=EmbeddingModelType.OPENAI,
        model_name=openai_embedding_model,
        collection_name=collection_name,
        similarity_metric=SimilarityMetric.COSINE,
    )

    print(vdb)

    query = "What did Emma do in this story?"
    (results, scores), search_time = vdb.search(query, k=2)

    print(f"\nMost similar documents: {results}")
    print(f"Similarity scores: {scores}")
    print(f"Search time: {search_time:.2f} ms")

    # Adding new documents
    new_docs = [
        "Emma enjoyed her day at the park.",
        "The park had many fun activities.",
    ]
    vdb.add_documents(new_docs, collection_name=collection_name)
    print(f"\nAdded {len(new_docs)} new documents. Total documents: {vdb}")

    (results, scores), search_time = vdb.search(query, k=3)

    print("\nUpdated search results:")
    print(f"Most similar documents: {results}")
    print(f"Similarity scores: {scores}")
    print(f"Search time: {search_time:.2f} ms")
