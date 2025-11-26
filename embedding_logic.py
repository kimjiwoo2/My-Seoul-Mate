import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import os

# --- [거리 계산용 함수] 하버사인 공식 ---
def haversine_vectorized(lat1, lon1, lat2_array, lon2_array):
    """
    지구 구면 거리 계산 (Haversine Formula)
    """
    R = 6371.0
    
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2_array)
    lon2_rad = np.radians(lon2_array)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c

class EmbeddingPredictor:
    def __init__(self, model_name='paraphrase-multilingual-mpnet-base-v2', base_data_dir="data/embeddings"):
        self.embed_model = None
        self.regression_model = None
        self.ohe_columns = {}
        self.all_places = [] 
        
        # 데이터 저장소
        self.new_places_data = {}
        self.new_places_embeddings = {}
        self.core_global_embeddings = {}
        self.core_global_metadata = {} 
        
        self.base_data_dir = base_data_dir
        if not os.path.exists(self.base_data_dir):
            os.makedirs(self.base_data_dir)

        self._load_embed_model(model_name)

    def _load_embed_model(self, model_name):
        try:
            print(f"⏳ [Embedding] '{model_name}' 모델 로드 중...")
            self.embed_model = SentenceTransformer(model_name)
            print("✅ [Embedding] 모델 로드 성공.")
        except Exception as e:
            print(f"🚨 [Embedding] 모델 로드 실패: {e}")

    def load_regression_model(self, model_path):
        try:
            self.regression_model = joblib.load(model_path)
            print("✅ [Embedding] 회귀 모델 로드 성공.")
        except Exception as e:
            print(f"🚨 [Embedding] 회귀 모델 로드 실패: {e}")

    def setup_ohe_features(self, all_places, all_companions, all_purposes):
        self.all_places = list(all_places)
        self.ohe_columns['source'] = sorted(list(all_places))[1:]
        self.ohe_columns['companion'] = sorted(list(all_companions))[1:]
        self.ohe_columns['purpose'] = sorted(list(all_purposes))[1:]

    def _save_embeddings(self, type_key, embeddings):
        if not isinstance(embeddings, np.ndarray):
            embeddings_np = embeddings.cpu().numpy()
        else:
            embeddings_np = embeddings
        file_path = os.path.join(self.base_data_dir, f"{type_key}_embeddings.npy")
        np.save(file_path, embeddings_np)

    def _load_embeddings(self, type_key):
        file_path = os.path.join(self.base_data_dir, f"{type_key}_embeddings.npy")
        if os.path.exists(file_path):
            return torch.from_numpy(np.load(file_path))
        return None

    def load_and_register_data(self, type_key, df, name_matcher_func):
        # 전처리
        df['명칭'] = df['명칭'].fillna('').astype(str)
        df['개요'] = df['개요'].fillna('').astype(str)
        if '위도' not in df.columns: df['위도'] = 0.0
        if '경도' not in df.columns: df['경도'] = 0.0
        df['위도'] = pd.to_numeric(df['위도'], errors='coerce').fillna(0.0)
        df['경도'] = pd.to_numeric(df['경도'], errors='coerce').fillna(0.0)

        # 텍스트 결합
        if '중분류' in df.columns:
            df['중분류'] = df['중분류'].fillna('').astype(str)
            df['combined_text'] = df['명칭'] + " (" + df['중분류'] + ") " + df['개요']
        else:
            df['combined_text'] = df['명칭'] + " " + df['개요']
        
        # 임베딩 로드 or 생성
        embeddings = self._load_embeddings(type_key)
        if embeddings is None or embeddings.shape[0] != len(df):
            print(f"    (정보) '{type_key}' 임베딩 생성/갱신 중...")
            embeddings = self.embed_model.encode(df['combined_text'].tolist(), convert_to_tensor=True)
            self._save_embeddings(type_key, embeddings)
        
        self.new_places_data[type_key] = df
        self.new_places_embeddings[type_key] = embeddings
        
        # Core 데이터 등록
        match_count = 0
        for idx, row in df.iterrows():
            current_name = row['명칭']
            matched = name_matcher_func(current_name)
            if matched:
                self.core_global_embeddings[matched] = embeddings[idx]
                self.core_global_metadata[matched] = {
                    "real_name": current_name, "lat": row['위도'], "lon": row['경도']
                }
                match_count += 1
        print(f"✅ [Embedding] '{type_key}' 로드 완료. (Core 매칭: {match_count}건)")

    # [Pipeline 1] 쿼리 유사도 계산 (Query Similarity)
    def get_query_similarity(self, type_key, user_query):
        """
        Pipeline 1에서 모델 점수와 섞기 위해 사용 (단순 텍스트 유사도)
        """
        if type_key not in self.new_places_embeddings:
            return {}
            
        # 1. 쿼리 벡터화
        query_emb = self.embed_model.encode(user_query, convert_to_tensor=True)
        target_embeddings = self.new_places_embeddings[type_key]
        
        # 2. 코사인 유사도 계산
        cos_scores = util.cos_sim(query_emb, target_embeddings)[0].cpu().numpy()
        
        # 3. 점수 변환 (1~5점)
        sim_scores_scaled = 1.0 + ((cos_scores + 1.0) / 2.0) * 4.0
        
        result_dict = {}
        df = self.new_places_data[type_key]
        for idx, score in enumerate(sim_scores_scaled):
            result_dict[df.iloc[idx]['명칭']] = score
            
        return result_dict

    # [Pipeline 1] 회귀 예측 점수 (Regression)
    def get_regression_scores(self, user_query, user_companion, user_purpose):
        if self.regression_model is None: return {}
        try:
            query_embedding = self.embed_model.encode(user_query, convert_to_numpy=True)
        except: return {}

        predicted_scores = {}
        companion_ohe = [1.0 if col == user_companion else 0.0 for col in self.ohe_columns['companion']]
        purpose_ohe = [1.0 if col == user_purpose else 0.0 for col in self.ohe_columns['purpose']]

        for place in self.all_places:
            source_ohe = [1.0 if col == place else 0.0 for col in self.ohe_columns['source']]
            input_vector = np.hstack([query_embedding, source_ohe, companion_ohe, purpose_ohe]).reshape(1, -1)
            try:
                pred = self.regression_model.predict(input_vector)[0]
                predicted_scores[place] = np.clip(pred, 1.0, 5.0)
            except: predicted_scores[place] = 0.0
        
        return predicted_scores

    # [Pipeline 2] 컨텍스트 점수 계산 (Dist, Key, Sim)
    def calculate_all_context_scores(self, type_key, user_query, prev_place_name, location_coords, query_keywords):
        if type_key not in self.new_places_data:
            return pd.DataFrame()
        
        df = self.new_places_data[type_key]
        cand_embs = self.new_places_embeddings[type_key]
        
        n_samples = len(df)
        score_sim = np.zeros(n_samples)
        score_dist = np.zeros(n_samples)
        score_key = np.zeros(n_samples)
        
        # A. SimCon (직전 장소 유사도)
        if prev_place_name and prev_place_name in self.core_global_embeddings:
            prev_emb = self.core_global_embeddings[prev_place_name]
            cos_sim = util.cos_sim(prev_emb, cand_embs)[0].cpu().numpy()
            score_sim = 1.0 + ((cos_sim + 1.0) / 2.0) * 4.0
        
        # B. DistCon (거리 점수)
        r_lat, r_lon = None, None
        if location_coords: 
            r_lat, r_lon = location_coords
        elif prev_place_name and prev_place_name in self.core_global_metadata:
            meta = self.core_global_metadata[prev_place_name]
            r_lat, r_lon = meta.get('lat', 0.0), meta.get('lon', 0.0)
            
        if r_lat and r_lat != 0.0:
            dists_km = haversine_vectorized(r_lat, r_lon, df['위도'].values, df['경도'].values)
            LAMBDA_DIST = 1.5
            dist_weights = np.exp(-dists_km / LAMBDA_DIST)
            score_dist = 1.0 + (dist_weights * 4.0)

        # C. KeyCon (키워드 유사도)
        if query_keywords and isinstance(query_keywords, str) and len(query_keywords) > 1:
            key_emb = self.embed_model.encode(query_keywords, convert_to_tensor=True)
            key_sim = util.cos_sim(key_emb, cand_embs)[0].cpu().numpy()
            score_key = 1.0 + ((key_sim + 1.0) / 2.0) * 4.0

        return pd.DataFrame({
            'name': df['명칭'],
            'sim_score': score_sim,
            'dist_score': score_dist,
            'key_score': score_key
        })

class RestaurantRecommender:
    def __init__(self):
        self.full_df = None
        self.is_ready = False
        print("✅ [Logic B] 음식점 랜덤 추천기 초기화")

    def load_data_from_df(self, df):
        if df is None or df.empty: return
        self.full_df = df.reset_index(drop=True)
        self.is_ready = True
        print(f"✅ [Logic B] 음식점 데이터 로드 완료 ({len(df)}개)")

    def get_random_recommendations(self, category, n=5):
        if not self.is_ready: return []
        target_df = self.full_df[self.full_df['소분류'] == category]
        if target_df.empty: return []
        sample_n = min(len(target_df), n)
        return target_df.sample(n=sample_n).to_dict('records')