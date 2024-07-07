import requests

url = 'https://translate-agent-61d887725b78.herokuapp.com/translate'
data = {
    "source_lang": "English",
    "target_lang": "Chinese",
    "source_text": "Hello, how are you?",
    "country": "China"
}

try:
    response = requests.post(url, json=data, timeout=60)  # 设置超时时间为60秒
    print(response.json())
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")
