import os
import pickle
import pandas as pd
import numpy as np


from pathlib import Path


class SentimentRecommendationModel:
    # keep a default (for local dev) but allow overrides via env var or constructor
    model_path = None  # will be set in __init__
    sentiment_model_fname = 'best_rf.pkl'
    tfidf_vectorizer_fname = 'tfidf_vectorizer.pkl'
    recommendation_model_fname = 'final_recommendation_model.pkl'
    cleaned_data_sentiment_fname = 'clean_data.pkl'

    def __init__(self, root_path: str = None):
        # Priority: explicit constructor root_path > MODEL_ROOT env var > repo-relative ./models
        env_root = os.environ.get('MODEL_ROOT')
        if root_path:
            self.model_path = root_path
        elif env_root:
            self.model_path = env_root
        else:
            # default to models directory next to this file
            self.model_path = str(Path(__file__).resolve().parent.joinpath('models'))

        def path_join(fname: str) -> str:
            return os.path.join(self.model_path, fname)

        # Load sentiment model
        try:
            with open(path_join(self.sentiment_model_fname), 'rb') as f:
                self.sentiment_model = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Error loading the sentiment model: {e}")

        # Load TF-IDF vectorizer
        try:
            with open(path_join(self.tfidf_vectorizer_fname), 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Error loading TF-IDF vectorizer: {e}")

        # Load recommendation model
        try:
            with open(path_join(self.recommendation_model_fname), 'rb') as f:
                self.user_final_rating = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Error loading recommendation model: {e}")

        # Load cleaned sentiment data
        try:
            with open(path_join(self.cleaned_data_sentiment_fname), 'rb') as f:
                self.cleaned_data_sentiment = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Error loading cleaned sentiment data: {e}")

        if not isinstance(self.cleaned_data_sentiment, pd.DataFrame):
            raise RuntimeError("Cleaned sentiment data is not a pandas DataFrame.")

        expected_columns = {'id', 'name', 'reviews_lemmatized'}
        if not expected_columns.issubset(self.cleaned_data_sentiment.columns):
            raise ValueError(
                f"Cleaned sentiment data is missing expected columns: "
                f"{expected_columns - set(self.cleaned_data_sentiment.columns)}"
            )

    def check_user_index_exists(self, user_id):
        idx = self.user_final_rating.index

        if user_id in idx:
            return user_id

        try:
            if isinstance(user_id, (int, np.integer)):
                user_id_str = str(user_id)
                if user_id_str in idx:
                    return user_id_str
        except Exception:
            pass

        str_map = {str(x): x for x in idx}
        if str(user_id) in str_map:
            return str_map[str(user_id)]

        raise ValueError(f"User ID {user_id} not found in recommendation model.")

    def recommend_products(self, user_id, top_k=5, top_n_sentiments=20):
        """
        Recommend products for a given user based on sentiment analysis
        and collaborative filtering.
        """
        try:
            user_key = self.check_user_index_exists(user_id)
        except KeyError:
            return None

        try:
            user_row = self.user_final_rating.loc[user_key]
        except Exception as e:
            raise RuntimeError(
                f"User Final Rating has unexpected structure due to sparse matrix: {e}"
            )

        if isinstance(user_row, pd.DataFrame):
            user_row = user_row.iloc[0]

        try:
            products = (
                user_row.sort_values(ascending=False)
                .head(top_n_sentiments)
                .index.tolist()
            )
        except Exception:
            vals = np.asarray(user_row)
            cols = list(user_row.index)
            order = np.argsort(-vals)[:top_n_sentiments]
            products = [cols[i] for i in order]

        if not products:
            return []

        df_top = self.cleaned_data_sentiment[
            self.cleaned_data_sentiment['id'].isin(products)
        ].copy()

        if df_top.empty:
            return []

        reviews = df_top['reviews_lemmatized'].astype(str).values

        try:
            X_tfidf = self.tfidf_vectorizer.transform(reviews)
        except Exception as e:
            raise RuntimeError(f"Error transforming reviews with TF-IDF vectorizer: {e}")

        try:
            predicted_sentiments = self.sentiment_model.predict(X_tfidf)
        except Exception as e:
            raise RuntimeError(f"Error predicting sentiments: {e}")

        df_top = df_top.assign(predicted_sentiment=predicted_sentiments)
        df_top['positive_sentiment'] = df_top['predicted_sentiment'].apply(
            lambda x: 1 if str(x).lower().startswith('pos') else 0
        )

        agg = df_top.groupby(['name']).agg(
            positive_sentiment_count=('positive_sentiment', 'sum'),
            total_sentiment_count=('predicted_sentiment', 'count')
        ).reset_index()

        agg['positive_sentiment_count'] = agg['positive_sentiment_count'].fillna(0)
        agg['total_sentiment_count'] = agg['total_sentiment_count'].replace(0, np.nan)
        agg['positive_sentiment_percent'] = (
            np.round(agg['positive_sentiment_count'] / agg['total_sentiment_count']) * 100
        )
        agg['positive_sentiment_percent'] = agg['positive_sentiment_percent'].fillna(0.0)

        topk_products = agg.sort_values(
            by='positive_sentiment_percent',
            ascending=False
        ).head(top_k)

        result = []
        for _, row in topk_products.iterrows():
            result.append({
                'name': row['name'],
                'positive_sentiment_percent': float(
                    np.round(row['positive_sentiment_percent'], 2)
                )
            })

        return result

    def predict_sentiment(self, reviews: str):
        """
        Predict sentiment for a single review.
        """
        if not reviews or not str(reviews).strip():
            return None

        try:
            X_tfidf = self.tfidf_vectorizer.transform([str(reviews)])
            predicted_sentiments = self.sentiment_model.predict(X_tfidf)
            return predicted_sentiments[0]
        except Exception as e:
            raise RuntimeError(f"Error predicting sentiments: {e}")
