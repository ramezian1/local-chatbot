import inspect
import os

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from chatbot import Bot

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
CORS(app)

bot = Bot(name="Bobo", top_k=3, min_score=0.05, use_color=False)


def _bot_accepts_history(bot_instance) -> bool:
    respond = getattr(bot_instance, "respond", None)
    if not callable(respond):
        return False
    try:
        signature = inspect.signature(respond)
    except (TypeError, ValueError):
        return False
    return "history" in signature.parameters


def _bot_response(message: str, history):
    if _bot_accepts_history(bot):
        return bot.respond(message, history=history)
    return bot.respond(message)


def _bot_health_payload():
    llm_enabled = False
    llm_model = None

    is_llm_enabled = getattr(bot, "is_llm_enabled", None)
    if callable(is_llm_enabled):
        llm_enabled = bool(is_llm_enabled())
    else:
        llm = getattr(bot, "llm", None)
        is_configured = getattr(llm, "is_configured", None)
        if callable(is_configured):
            llm_enabled = bool(is_configured())
        elif llm is not None:
            llm_enabled = bool(getattr(llm, "api_key", ""))

    if llm_enabled:
        llm_model = getattr(getattr(bot, "llm", None), "model", None)

    return {
        "status": "healthy",
        "bot_name": getattr(bot, "name", "Bobo"),
        "llm_enabled": llm_enabled,
        "llm_model": llm_model,
    }


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
        response = _bot_response(message, history)
        return jsonify({"response": response, "status": "success"})
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify(_bot_health_payload())


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
