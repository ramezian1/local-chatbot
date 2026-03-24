import os

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from chatbot import Bot

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
CORS(app)

bot = Bot(name="Bobo", top_k=3, min_score=0.05, use_color=False)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(silent=True) or {}
        message = str(data.get("message", "")).strip()
        history = data.get("history", [])
        if not message:
            return jsonify({"error": "Empty message"}), 400
        if not isinstance(history, list):
            history = []
        response = bot.respond(message, history=history)
        return jsonify({"response": response, "status": "success"})
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "healthy",
            "bot_name": bot.name,
            "llm_enabled": bot.is_llm_enabled(),
            "llm_model": bot.llm.model if bot.is_llm_enabled() else None,
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
