from uuid import uuid4
from fastapi.testclient import TestClient
from dotenv import load_dotenv
import numpy as np

from app.main import app
from utils.cohere_client import CohereClient

load_dotenv()
client = TestClient(app)

PHRASE_GROUPS = {
    "greetings": [
        "Hello, how are you?",
        "Hi there, how's it going?",
        "Hey, what's up?",
        "Good morning, how are you doing?",
        "Greetings, how do you do?"
    ],
    "weather": [
        "It's a beautiful sunny day today",
        "The weather is great outside",
        "Such lovely weather we're having",
        "Perfect weather for a picnic",
        "The sun is shining brightly today"
    ],
    "food": [
        "I'm hungry, let's get something to eat",
        "Should we order some food?",
        "Let's grab a bite to eat",
        "Time for dinner, what should we have?",
        "I could use a snack right now"
    ]
}

def create_library_with_embeddings():
    cohere = CohereClient()
    

    resp = client.post("/libraries", json={
        "name": "Semantic Search Test",
        "metadata": {"description": "Test library for semantic search comparison"}
    })
    assert resp.status_code == 200
    lib_id = resp.json()["id"]
    
    results = {}
    for group_name, phrases in PHRASE_GROUPS.items():
        doc_id = str(uuid4())
        doc_resp = client.post(
            f"/libraries/{lib_id}/documents",
            json={
                "id": doc_id,
                "title": f"{group_name.title()} Examples",
                "metadata": {"category": group_name}
            }
        )
        assert doc_resp.status_code == 200
        
        embeddings = cohere.get_embeddings(phrases)
        
        chunk_ids = []
        for phrase, embedding in zip(phrases, embeddings):
            chunk_resp = client.post(
                f"/libraries/{lib_id}/chunks",
                json={
                    "doc_id": doc_id,
                    "text": phrase,
                    "embedding": embedding,
                    "metadata": {"group": group_name}
                }
            )
            assert chunk_resp.status_code == 200
            chunk_ids.append(chunk_resp.json()["id"])
        
        results[group_name] = {
            "doc_id": doc_id,
            "chunk_ids": chunk_ids,
            "phrases": phrases,
            "embeddings": embeddings
        }
    
    return lib_id, results


def test_semantic_search_comparison():
    """compare semantic search results across algorithms"""
    lib_id, data = create_library_with_embeddings()
    cohere = CohereClient()
    
    test_queries = {
        "greetings": [
            "How are you doing today?",
            "What's happening?",
            "How do you fare?"
        ],
        "weather": [
            "Is it nice outside?",
            "How's the weather looking?",
            "Perfect day for outdoor activities"
        ],
        "food": [
            "Where should we eat?",
            "I need some food",
            "What's for dinner?"
        ]
    }
    
    algorithms = ["kd", "ball", "linear"]
    results = {}
    
    for algo in algorithms:
        algo_results = {}
        print(f"\nTesting {algo.upper()} Tree Search:")
        print("-" * 50)
        
        for category, queries in test_queries.items():
            category_matches = 0
            total_queries = len(queries)
            
            for query in queries:
                query_embedding = cohere.get_embedding(query)
                
                response = client.post(
                    f"/libraries/{lib_id}/search",
                    json={
                        "embedding": query_embedding,
                        "k": 3,
                        "algorithm": algo
                    }
                )
                assert response.status_code == 200
                
                search_results = response.json()["results"]
                top_categories = [
                    r["chunk"]["metadata"]["group"] 
                    for r in search_results
                ]
                
                if category in top_categories:
                    category_matches += 1
                
                print(f"\nQuery: {query}")
                print(f"Expected Category: {category}")
                print("Top 3 Results:")
                for i, result in enumerate(search_results, 1):
                    print(f"{i}. {result['chunk']['text']} "
                          f"(Category: {result['chunk']['metadata']['group']}, "
                          f"Distance: {result['distance']:.4f})")
            
            accuracy = category_matches / total_queries
            algo_results[category] = accuracy
            print(f"\n{category.title()} Accuracy: {accuracy:.2%}")
        
        results[algo] = algo_results
    
    print("\nAlgorithm Performance Comparison:")
    print("-" * 50)
    for algo, accuracies in results.items():
        avg_accuracy = sum(accuracies.values()) / len(accuracies)
        print(f"{algo.upper():6} Average Accuracy: {avg_accuracy:.2%}")
        for category, accuracy in accuracies.items():
            print(f"  - {category.title():10}: {accuracy:.2%}")
    
    client.delete(f"/libraries/{lib_id}")


if __name__ == "__main__":
    test_semantic_search_comparison() 