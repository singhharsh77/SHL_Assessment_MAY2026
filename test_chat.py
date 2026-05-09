import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_health():
    res = requests.get(f"{BASE_URL}/health")
    print("Health check:", res.status_code, res.json())

def test_chat():
    messages = [
        {"role": "user", "content": "I am hiring a Java developer who works with stakeholders"}
    ]
    
    print("\n--- Turn 1 ---")
    start = time.time()
    res = requests.post(f"{BASE_URL}/chat", json={"messages": messages})
    print(f"Time: {time.time() - start:.2f}s")
    data = res.json()
    print(json.dumps(data, indent=2))
    
    messages.append({"role": "assistant", "content": data["reply"]})
    messages.append({"role": "user", "content": "Mid-level, around 4 years"})

    print("\n--- Turn 2 ---")
    start = time.time()
    res = requests.post(f"{BASE_URL}/chat", json={"messages": messages})
    print(f"Time: {time.time() - start:.2f}s")
    data = res.json()
    print(json.dumps(data, indent=2))

if __name__ == "__main__":
    test_health()
    test_chat()
