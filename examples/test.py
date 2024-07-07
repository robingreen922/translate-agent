import requests

url = 'https://translate-agent-61d887725b78.herokuapp.com/translate'
# url = 'http://127.0.0.1:5000/translate'
data = {
    "source_lang": "English",
    "target_lang": "Chinese",
    "source_text": "Hello, how are you?",
    "country": "China"
}

response = requests.post(url, json=data)
print(response.json())