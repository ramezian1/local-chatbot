from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
from chatbot import Bot

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
CORS(app)

# Lazy loading: bot is initialized on first use
bot_instance = None

def get_bot():
    global bot_instance
    if bot_instance is None:
        bot_instance = Bot(name="Robo", top_k=3, min_score=0.05, use_color=False)
    return bot_instance

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        if not message:
            return jsonify({'error': 'Empty message'}), 400
        
        bot = get_bot()
        response = bot.respond(message)
        return jsonify({'response': response, 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'bot_name': 'Robo'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
