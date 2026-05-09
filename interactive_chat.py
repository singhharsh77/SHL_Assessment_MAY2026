import requests
import json

BASE_URL = "http://localhost:8000"

def interactive_session():
    print("=========================================")
    print("  SHL Agent Interactive Test Shell")
    print("  Type 'quit' or 'exit' to stop.")
    print("=========================================\n")
    
    messages = []
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        messages.append({"role": "user", "content": user_input})
        
        try:
            res = requests.post(f"{BASE_URL}/chat", json={"messages": messages})
            res.raise_for_status()
            data = res.json()
            
            # Print Agent Reply
            print(f"\nAgent: {data['reply']}")
            
            # Print Recommendations if any
            if data['recommendations']:
                print("\n[ Recommendations Provided ]")
                for i, r in enumerate(data['recommendations'], 1):
                    print(f"  {i}. {r['name']} ({r['test_type']}) - {r['url']}")
            
            print("\n-----------------------------------------")
            
            # Append agent reply to history so context is maintained
            messages.append({"role": "assistant", "content": data['reply']})
            
            if data.get('end_of_conversation'):
                print("\n[ Agent marked conversation as ENDED ]")
                
        except Exception as e:
            print(f"\n[!] Error connecting to server: {e}")

if __name__ == "__main__":
    interactive_session()
