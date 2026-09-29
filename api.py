"""
Minimal API for the Recommendation Agent UI.

No existing frontend or API existed anywhere in the repo, so this is a
small, plain Flask app — deliberately not a microservice, not a
database-backed service, just two endpoints reading the same CSVs the
agents already use.

Endpoints:
    GET  /api/customers                     -> list of customers for the demo picker
    GET  /api/dashboard/<customer_id>       -> full dashboard (all agents)
    POST /api/ask   {customer_id, question} -> routed single-question answer
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify, request, send_from_directory

from data_access import load_customers
from coordinator.orchestrator import load_dashboard, handle_question

app = Flask(__name__, static_folder=None)


@app.get("/")
def index():
    return send_from_directory(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend"),
        "index.html",
    )


@app.get("/api/customers")
def api_customers():
    df = load_customers()
    return jsonify(df[["customer_id", "name"]].to_dict(orient="records"))


@app.get("/api/dashboard/<customer_id>")
def api_dashboard(customer_id):
    result = load_dashboard(customer_id)
    if "error" in result:
        return jsonify(result), 404
    return jsonify(result)


@app.post("/api/ask")
def api_ask():
    body = request.get_json(force=True) or {}
    customer_id = body.get("customer_id")
    question = body.get("question", "")
    if not customer_id or not question:
        return jsonify({"error": "customer_id and question are both required"}), 400
    result = handle_question(customer_id, question)
    if "error" in result:
        return jsonify(result), 404
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
