import numpy as np
from sqlalchemy import create_engine, Column, func, select
from sqlmodel import Field, SQLModel
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector
from typing import List
from sentence_transformers import SentenceTransformer
import time
from enum import Enum
from uuid import UUID, uuid4

Base = declarative_base()


class SimilarityMetric(Enum):
    EUCLIDEAN = 1
    COSINE = 2


def measure_time(function):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = function(*args, **kwargs)
        execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        return result, execution_time

    return wrapper


class Document(SQLModel, table=True):
    __tablename__ = "vecky-documents"

    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    content: str
    embedding: List[float] = Field(default=None, sa_column=Column(Vector(384)))


class VectorDatabase:
    def __init__(
        self, db_url, embedding_model, similarity_metric=SimilarityMetric.COSINE
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
        cls, db_url, docs, embedding_model, similarity_metric=SimilarityMetric.COSINE
    ):
        instance = cls(db_url, embedding_model, similarity_metric)
        instance.add_documents(docs)
        return instance

    def add_documents(self, docs):
        session = self.Session()
        try:
            for doc in docs:
                embedding = self.embedding_model.encode(doc)
                embedding_list = embedding.tolist()
                db_doc = Document(content=doc, embedding=embedding_list)
                session.add(db_doc)
            session.commit()
        finally:
            session.close()

    def create_embeddings(self):
        # This method is now a no-op, as embeddings are created when adding documents
        pass

    def set_similarity_metric(self, metric):
        self.similarity_metric = metric
        self._set_similarity_function()

    @measure_time
    def search(self, query: str, k: int = 5):
        query_embedding = self.embedding_model.encode(query).tolist()
        session = self.Session()
        try:
            stmt = (
                select(
                    Document.content,
                    self._sim_func(Document.embedding, query_embedding).label(
                        "similarity"
                    ),
                )
                .order_by(self._sim_func(Document.embedding, query_embedding))
                .limit(k)
            )
            results = session.execute(stmt).all()
            return [r.content for r in results], [r.similarity for r in results]
        finally:
            session.close()

    def __repr__(self):
        session = self.Session()
        try:
            doc_count = session.query(Document).count()
            return (
                f"VectorDatabase(num_documents={doc_count}, "
                f"embedding_model={self.embedding_model.__class__.__name__}, "
                f"similarity_metric={self.similarity_metric.name})"
            )
        finally:
            session.close()


if __name__ == "__main__":
    print("Loading embedding model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

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

    docs = document.split("\n")
    print(f"Creating vector database for {len(docs)} documents...")
    vdb = VectorDatabase.from_documents(db_url, docs, model, SimilarityMetric.COSINE)
    print(vdb)

    query = "What did Emma do in this story?"
    results, scores = vdb.search(query, k=2)
    search_time = (
        scores  # The measure_time decorator returns execution time as the second item
    )

    print(f"\nMost similar documents: {results}")
    print(f"Similarity scores: {scores}")
    print(f"Search time: {search_time:.2f} ms")

    # Adding new documents
    new_docs = [
        "Emma enjoyed her day at the park.",
        "The park had many fun activities.",
    ]
    vdb.add_documents(new_docs)
    print(f"\nAdded {len(new_docs)} new documents. Total documents: {vdb}")

    # Changing similarity metric
    vdb.set_similarity_metric(SimilarityMetric.EUCLIDEAN)
    print(f"Changed similarity metric to {vdb.similarity_metric.name}")

    results, scores = vdb.search(query, k=3)
    search_time = scores

    print("\nUpdated search results:")
    print(f"Most similar documents: {results}")
    print(f"Similarity scores: {scores}")
    print(f"Search time: {search_time:.2f} ms")
