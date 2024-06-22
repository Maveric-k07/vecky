import time
from enum import Enum
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class SimilarityMetric(Enum):
    EUCLIDEAN = 1
    COSINE = 2


class VectorDatabase:
    def __init__(
        self,
        documents: List[str],
        embedding_model: SentenceTransformer,
        similarity_metric: SimilarityMetric,
    ):
        self.documents = np.array(documents)
        self.embedding_model = embedding_model
        self.similarity_metric = similarity_metric
        self.embeddings = None

    def create_embeddings(self):
        self.embeddings = self.embedding_model.encode(self.documents)

    def calculate_similarity(self, query_embedding):
        if self.similarity_metric == SimilarityMetric.EUCLIDEAN:
            return np.linalg.norm(self.embeddings - query_embedding, axis=1)
        elif self.similarity_metric == SimilarityMetric.COSINE:
            norm_embeddings = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norm_query = np.linalg.norm(query_embedding)
            return (
                1
                - np.dot(
                    self.embeddings / norm_embeddings, query_embedding / norm_query
                ).flatten()
            )

    def search(self, query: str, k: int = 5):
        start_time = time.time()

        query_embedding = self.embedding_model.encode([query])[0]
        similarities = self.calculate_similarity(query_embedding)

        top_indices = np.argsort(similarities)[:k]
        top_documents = self.documents[top_indices]
        top_scores = similarities[top_indices]

        execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        return top_documents, top_scores, execution_time


# Usage example
if __name__ == "__main__":

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

    print("Loading embedding model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

    # Example documents (replace with your actual documents)
    docs = document.split("\n")

    print(f"Creating vector database for {len(docs)} docs...")
    vdb = VectorDatabase(docs, model, SimilarityMetric.COSINE)
    vdb.create_embeddings()

    query = "What did Emma do in this story?"
    results, scores, exec_time = vdb.search(query, k=2)

    print(f"\nMost similar documents: {results}")
    print(f"Similarity scores: {scores}")
    print(f"Search time: {exec_time:.2f} ms")
