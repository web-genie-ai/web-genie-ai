from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

def score(sentences1, sentences2):
    embeddings1 = model.encode(sentences1)
    embeddings2 = model.encode(sentences2)
    similarities = cosine_similarity(embeddings1, embeddings2)
    # Scale similarities to be between 0 and 1
    scores = [(float(similarity[0]) + 1) / 2 for similarity in similarities]
    return scores

if __name__ == "__main__":
    # Define a list of sentence pairs
    sentence_pairs = [
        ("The cat is on the mat.", "A cat sits on a rug."),
        ("I am going to the store.", "I will head to the shop."),
        ("The weather is great today.", "It is sunny outside."),
        ("She loves playing the piano.", "She enjoys playing the guitar.")
    ]
    sentences1 = [pair[0] for pair in sentence_pairs]
    sentences2 = [pair[1] for pair in sentence_pairs]
    # Call the function
    print(score(sentences1, sentences2))
