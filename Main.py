import requests
from config import HF_API_KEY

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL}"

headers = {"Authorization": f"Bearer {HF_API_KEY}"}

def get_embeddings(sentence):
    response=requests.post(API_URL,headers=headers,json={"inputs":sentence})
    return response.json()
def cosine_similarity(vec1, vec2):
    import numpy as np
    v1, v2 = np.array(vec1), np.array(vec2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
def check_similarity(sentence1,sentence2):
    emb1=get_embeddings(sentence1)[0]
    emb2=get_embeddings(sentence2)[0]
    score=cosine_similarity(emb1,emb2)
    print(f"\n Similarity between:\n 1) {sentence1}\n 2) {sentence2}")
    print(f"Score : {round(score*100, 2)}%")

if __name__=="__main__":
    print("Sentence similarity checker")
    print("type 2 sentences to compare, or 'exit' to quit.")
    while True:
        s1 = input("Sentence 1: ")
        if s1.lower() =="exit": break
        s2 = input("Sentence 2:")
        if s2.lower() =="exit": break
        check_similarity(s1, s2)
        print("\n--- Try again or type 'exit' ---\n")