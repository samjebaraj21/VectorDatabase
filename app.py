from vector_store import VectorStore
import numpy as np

# Create a VectorStore instance
vector_store = VectorStore()

sentence_vectors = {}

# Define sentences
sentences = [
    "I eat mango",
    "mango is my favorite fruit",
    "mangoes, apples, and oranges are the best fruits objectively.",
    "fruits are good for your health.",
]

# Tokenization and vocabulary building
vocabulary = set()

for sentence in sentences:
    tokens = sentence.lower().split()
    vocabulary.update(tokens)

# Assign unique indices to words in the vocab
word_to_index = {word:i for i, word in enumerate(vocabulary)}

# Vectorization
for sentence in sentences:
    vector = np.zeros(len(vocabulary))
    for token in sentence.lower().split():
        vector[word_to_index[token]] += 1

    sentence_vectors[sentence] = vector

# Add vectors to the vector database
for sentence, vector in sentence_vectors.items():
    vector_store.add_vector(sentence, vector)

# Search for the similarity
query_sentence = "Mango is the best fruit"
query_vector = np.zeros(len(vocabulary))
query_tokens = query_sentence.lower().split()
for token in query_tokens:
    query_vector[word_to_index[token]] += 1

similar_sentences = vector_store.find_similar_vectors(query_vector, num_results=2)

print("Query sentence:", query_sentence)
print("Similar sentence:")
for sentence, similarity in similar_sentences:
    print(f"{sentence}: Similarity = {similarity:.4f}")