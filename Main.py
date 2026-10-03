import requests
from config import HF_API_KEY


MODEL = "sentence-transformers/all-MiniLM-L6-v2"
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL}"

headers = {"Authorization": f"Bearer {HF_API_KEY}"}

def get_similarity_score(source_sentence, target_sentence):
    response = requests.post(
        API_URL,
        headers=headers,
        json={"inputs": {"source_sentence": source_sentence, "sentences": [target_sentence]}}
    )
    response.raise_for_status()
    data = response.json()
    
    if isinstance(data, list) and len(data) > 0:
        return data[0]
    else:
        raise ValueError(f"Unexpected response format: {data}")

def check_similarity(sentence1, sentence2):
    score = get_similarity_score(sentence1, sentence2)
    print(f"\n🔎 Similarity between:\n  1) {sentence1}\n  2) {sentence2}")
    print(f"➡️ Score: {round(score * 100, 2)}%")

if __name__ == "__main__":
    print("✨ Sentence Similarity Checker ✨")
    print("Type two sentences to compare, or 'exit' to quit.\n")
    while True:
        s1 = input("Sentence 1: ")
        if s1.lower() == "exit":
            break
        s2 = input("Sentence 2: ")
        if s2.lower() == "exit":
            break
        check_similarity(s1, s2)
        print("\n--- Try again or type 'exit' ---\n")
