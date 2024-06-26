import time
from enum import Enum
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer


class SimilarityMetric(Enum):
    EUCLIDEAN = 1
    COSINE = 2


def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        return result, execution_time

    return wrapper


class VectorDatabase:
    def __init__(
        self,
        documents: List[str] = None,
        embedding_model: SentenceTransformer = None,
        similarity_metric: SimilarityMetric = SimilarityMetric.EUCLIDEAN,
    ):
        self.documents = np.array(documents) if documents else None
        self.embedding_model = embedding_model
        self.similarity_metric = similarity_metric
        self.embeddings = None
        self._set_similarity_function()

    def _set_similarity_function(self):
        if self.similarity_metric == SimilarityMetric.EUCLIDEAN:
            self._similarity_func = self._euclidean_distance
        elif self.similarity_metric == SimilarityMetric.COSINE:
            self._similarity_func = self._cosine_similarity
        else:
            raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")

    @staticmethod
    def _euclidean_distance(a, b):
        distances = np.linalg.norm(a - b, axis=1)
        return 1 / (1 + distances)  # Normalize to 0-1 range, where 1 is most similar

    @staticmethod
    def _cosine_similarity(a, b):
        norm_a = np.linalg.norm(a, axis=1, keepdims=True)
        norm_b = np.linalg.norm(b)
        cosine_sim = np.dot(a / norm_a, b / norm_b).flatten()
        return (cosine_sim + 1) / 2

    @classmethod
    def from_documents(
        cls,
        documents: List[str],
        embedding_model: SentenceTransformer,
        similarity_metric: SimilarityMetric = SimilarityMetric.EUCLIDEAN,
    ) -> "VectorDatabase":
        instance = cls(documents, embedding_model, similarity_metric)
        instance.create_embeddings()
        return instance

    def create_embeddings(self):
        if self.documents is None or self.embedding_model is None:
            raise ValueError(
                "Documents and embedding model must be set before creating embeddings"
            )
        self.embeddings = self.embedding_model.encode(self.documents)

    def set_similarity_metric(self, metric: SimilarityMetric):
        self.similarity_metric = metric
        self._set_similarity_function()

    @measure_time
    def search(self, query: str, k: int = 5) -> Tuple[List[str], np.ndarray, float]:
        if self.embeddings is None:
            raise ValueError(
                "Embeddings have not been created. Call create_embeddings() first."
            )

        query_embedding = self.embedding_model.encode([query])[0]
        similarities = self._similarity_func(self.embeddings, query_embedding)

        top_indices = np.argsort(similarities)[:k]
        top_documents = self.documents[top_indices].tolist()
        top_scores = similarities[top_indices]

        return top_documents, top_scores

    def add_documents(self, new_documents: List[str]):
        if self.documents is None:
            self.documents = np.array(new_documents)
        else:
            self.documents = np.concatenate((self.documents, np.array(new_documents)))

        new_embeddings = self.embedding_model.encode(new_documents)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack((self.embeddings, new_embeddings))

    def __repr__(self):
        return (
            f"VectorDatabase(num_documents={len(self.documents) if self.documents is not None else 0}, "
            f"embedding_model={self.embedding_model.__class__.__name__}, "
            f"similarity_metric={self.similarity_metric.name})"
        )


if __name__ == "__main__":
    print("Loading embedding model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

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
    vdb = VectorDatabase.from_documents(docs, model, SimilarityMetric.COSINE)
    print(repr(vdb))

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
    vdb.add_documents(new_docs)
    print(
        f"\nAdded {len(new_docs)} new documents. Total documents: {len(vdb.documents)}"
    )

    # Changing similarity metric
    vdb.set_similarity_metric(SimilarityMetric.EUCLIDEAN)
    print(f"Changed similarity metric to {vdb.similarity_metric.name}")

    (results, scores), search_time = vdb.search(query, k=2)

    print("\nUpdated search results:")
    print(f"Most similar documents: {results}")
    print(f"Similarity scores: {scores}")
    print(f"Search time: {search_time:.2f} ms")
