from flask import Flask, request, jsonify, send_from_directory
from FlaskInference import translate_english_to_klingon

app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json(force=True)
    english_sentence = data.get('english_sentence')
    if not english_sentence:
        return jsonify({'error': 'No sentence provided'}), 400
    
    klingon_sentence = translate_english_to_klingon(english_sentence)
    return jsonify({
        'english_sentence': english_sentence,
        'klingon_translation': klingon_sentence
    })

if __name__ == '__main__':
    app.run(debug=True)
