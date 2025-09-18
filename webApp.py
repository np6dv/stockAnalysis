from flask import Flask, request, jsonify
from stock_analysis import analyze_stock, convert_to_builtin_type

app = Flask(__name__)

@app.route("/analyze", methods=["GET"])
def analyze():
    ticker = request.args.get("ticker")
    if not ticker:
        return jsonify({"error": "Missing ticker"}), 400
    try:
        result = analyze_stock(ticker)
        result = convert_to_builtin_type(result)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
