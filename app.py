import os
import logging
from flask import Flask, request, render_template, jsonify
from model import SentimentRecommendationModel

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

# Try to instantiate the model at startup. If it fails, keep the error to show on the UI.
model = None
model_load_error = None
try:
    model_root = os.environ.get('MODEL_ROOT')  # optional override
    if model_root:
        model = SentimentRecommendationModel(root_path=model_root)
    else:
        model = SentimentRecommendationModel()
    app.logger.info('Model loaded successfully')
except Exception as e:
    model_load_error = str(e)
    app.logger.exception('Failed to load model: %s', model_load_error)


@app.route('/')
def home():
    # Show load error if model failed to initialize
    return render_template('index.html', message_display=model_load_error)


@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    # Allow GET to render the form and POST to compute recommendations
    if request.method == 'GET':
        return render_template('index.html')

    if model is None:
        app.logger.error('Model not available: %s', model_load_error)
        return render_template('index.html', message_display=f"Model not available: {model_load_error}")

    user_id = request.form.get('user_id') or request.form.get('username')
    if not user_id:
        return render_template('index.html', message_display="Please provide a user id.")

    try:
        recs = model.recommend_products(user_id, top_k=5)
    except ValueError as ve:
        app.logger.warning('Invalid user id provided: %s - %s', user_id, ve)
        return render_template('index.html', message_display=str(ve))
    except Exception as e:
        app.logger.exception('Error computing recommendations for user %s', user_id)
        # Show a friendly message to the user but keep details in logs
        return render_template('index.html', message_display=f"Error computing recommendations. See server logs for details.")

    if not recs:
        return render_template('index.html', message_display="No recommendations found for this user.")

    return render_template('index.html', recommendations=recs, user_id=user_id)


@app.errorhandler(405)
def method_not_allowed(e):
    app.logger.warning('405 Method Not Allowed: %s %s', request.method, request.path)
    return render_template('index.html', message_display='Method not allowed. Please use the form to submit.'), 405


@app.route('/api/recommend', methods=['GET'])
def api_recommend():
    if model is None:
        return jsonify({'error': f"Model not available: {model_load_error}"}), 500

    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'user_id parameter required'}), 400

    try:
        recs = model.recommend_products(user_id, top_k=5)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'user_id': user_id, 'recommendations': recs})


if __name__ == '__main__':
    app.run(debug=True)