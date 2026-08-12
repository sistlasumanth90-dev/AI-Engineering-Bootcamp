from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
sentence = "The Puppy is playing in the garden"
embedding = model.encode(sentence)
print("Embedding:", embedding)
print("Shape:", embedding.shape)

sentence_a = "The puppy is playing in the garden"
sentence_b = "A young dog is playing in the garded"
sentence_c = "SQL Server supports database indexes"

embedding_a = model.encode(sentence_a)
embedding_b = model.encode(sentence_b)
embedding_c = model.encode(sentence_c)

import numpy as np
def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a,vector_b)
    magnitude_a = np.linalg.norm(vector_a)
    maginitude_b = np.linalg.norm(vector_b)
    return dot_product/(magnitude_a * maginitude_b)

similarity_ab = cosine_similarity(embedding_a, embedding_b)
similarity_ac = cosine_similarity(embedding_a, embedding_c)

print("Puppy Sentence vs Doggy Sentence:", similarity_ab)
print("Puppy Sentence vs SQL Sentence:", similarity_ac)

documents = [
    "Dogs are loyal animals and make great pets.",
    "Puppies need regular exercise and healthy food.",
    "SQL indexes can improve database query performance.",
    "Python is widely used for machine learning.",
    "Dubai has very hot summers.",
    "Database indexing can make queries run faster."
]

document_embeddings = model.encode(documents)
print("Shape:", document_embeddings.shape)

query = "How can I make my database queries faster?"

query_embedding = model.encode(query)

results = {}

for i, document_embeddings in enumerate(document_embeddings):
    similarity = cosine_similarity(query_embedding, document_embeddings)
    results[i] = similarity

ranked_results = sorted(results.items(), key=lambda item: item[1],reverse=True)
top_3 = ranked_results[:3]

print("\nQuery:", query)
print("n\Top 3 Results:")

for index, score in top_3:
    print(documents[index],">",score)
