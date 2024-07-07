from translation_agent.utils import translate  # 使用绝对导入

from flask import Flask, request, jsonify
import os

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "Translation service is running.", 200

@app.route('/translate', methods=['POST'])
def translate_text():
    data = request.json
    source_lang = data.get('source_lang')
    target_lang = data.get('target_lang')
    source_text = data.get('source_text')
    country = data.get('country', '')

    if not all([source_lang, target_lang, source_text]):
        return jsonify({"error": "Missing required parameters"}), 400

    try:
        translated_text = translate(source_lang, target_lang, source_text, country)
        return jsonify({"translated_text": translated_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
