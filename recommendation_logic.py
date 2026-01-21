import pandas as pd
import numpy as np

# --- 이름 매칭 클래스 ---
class NameMatcher:
    @staticmethod
    def check_match(core_names, target_name):
        clean_target = str(target_name).strip()
        if "한강공원" in clean_target:
            if "눈썰매장" in clean_target: return None
            if "한강공원" in core_names: return "한강공원"
        if clean_target in core_names: return clean_target
        return None

# --- 베이지안 평점 ---
def calculate_bayesian_rating(N, R, m, C):
    if N == 0 or pd.isna(N) or pd.isna(R): return m
    return ((N / (C + N)) * R) + ((C / (C + N)) * m)

# --- [Pipeline 1-A] Core 장소 통계 점수 (Hierarchy Score) ---
def calculate_core_hierarchy_score(main_df, user_companion, user_purpose, user_tourism_type):
    C = 7
    RATING_COL = 'rating_adjusted_0.2'
    m_global = main_df[RATING_COL].mean()
    all_places = main_df['source'].unique()
    raw_results = {}

    for place in all_places:
        df_place = main_df[main_df['source'] == place]
        c_3 = len(df_place); R_3 = calculate_bayesian_rating(c_3, df_place[RATING_COL].mean(), m_global, C)
        df_L2 = df_place[df_place['purpose'] == user_purpose]
        c_2 = len(df_L2); R_2 = calculate_bayesian_rating(c_2, df_L2[RATING_COL].mean(), R_3, C)
        df_L1 = df_place[(df_place['purpose'] == user_purpose) & (df_place['companion'] == user_companion)]
        c_1 = len(df_L1); R_1 = calculate_bayesian_rating(c_1, df_L1[RATING_COL].mean(), R_2, C)

        if c_1 > 0: w = (0.4, 0.3, 0.3)
        elif c_2 > 0: w = (0.0, 0.7, 0.3)
        else: w = (0.0, 0.0, 1.0)
        base_score = (w[0] * R_1) + (w[1] * R_2) + (w[2] * R_3)
        
        try: place_topic = df_place['topic'].iloc[0] 
        except: place_topic = ""
        TOPIC_WEIGHT = 1.05 if user_tourism_type == '관광지' else 1.2
        if (user_tourism_type == '관광지' and str(place_topic).startswith('관광지')) or (place_topic == user_tourism_type):
            base_score *= TOPIC_WEIGHT
        raw_results[place] = base_score

    scores = pd.Series(raw_results)
    if scores.max() > scores.min():
        normalized = 4.0 * (scores - scores.min()) / (scores.max() - scores.min()) + 1.0
    else: normalized = np.clip(scores, 1.0, 5.0)
    return normalized.to_dict()

# --- [Pipeline 1 통합 함수] R_Model 계산 ---
def calculate_r_model(core_df, predictor, user_query_raw, user_companion, user_purpose, user_tourism_type, keyword_query):
    """
    Pipeline 1의 최종 결과물인 R_Model 점수를 계산하여 딕셔���리로 반환
    """
    # 1-1. Core Hierarchy Score
    hier_scores = calculate_core_hierarchy_score(core_df, user_companion, user_purpose, user_tourism_type)
    
    # 1-2. Regression Score
    reg_input = keyword_query if keyword_query else user_purpose
    reg_scores = predictor.get_regression_scores(reg_input, user_companion, user_purpose)
    
    # 1-3. Query Similarity (Core + New 모두 사용)
    query_sim_scores = predictor.get_query_similarity(user_tourism_type, user_query_raw)
    
    # 1-4. 전체 평균 평점 계산 (New 데이터의 Base Score용)
    global_mean_score = 3.0 # fallback
    if 'rating_adjusted_0.2' in core_df.columns:
        global_mean_score = core_df['rating_adjusted_0.2'].mean()
    
    core_places = list(hier_scores.keys())
    
    # R_Model 점수 저장소
    r_model_dict = {}
    
    # 모든 후보 장소(New + Core)에 대해 계산
    # (predictor.new_places_data에 있는 모든 장소 순회)
    all_candidates = predictor.new_places_data.get(user_tourism_type)
    if all_candidates is None: return {}, []
    
    for _, row in all_candidates.iterrows():
        name = row['명칭']
        matched_core = NameMatcher.check_match(core_places, name)
        
        # 쿼리 유사도 (공통)
        s_query = query_sim_scores.get(name, 3.0)
        
        if matched_core:
            # [Core Logic]
            # Step 1. S_Stat = 0.6 * Hierarchy + 0.4 * Regression
            s_h = hier_scores.get(matched_core, 3.0)
            s_r = reg_scores.get(matched_core, 3.0)
            s_stat = (0.6 * s_h) + (0.4 * s_r)
            
            # Step 2. R_Model = 0.6 * S_Stat + 0.4 * QuerySim
            r_model = (0.6 * s_stat) + (0.4 * s_query)
            
        else:
            # [New Logic]
            # Step 1. S_Stat = global_mean_score
            s_stat = global_mean_score
            
            r_model = (0.3 * s_stat) + (0.7 * s_query)
            
        r_model_dict[name] = r_model
        
    return r_model_dict, core_places

# --- [메인] 통합 추천 로직 ---
def get_combined_recommendations(
    core_df, predictor, user_query_raw, user_companion, user_purpose, user_tourism_type, 
    prev_place_name=None, location_coords=None, keyword_query=None,
    is_famous_intent=False # ★ 인자 추가
):
    print(f"\n--- 통합 추천: [{user_tourism_type}] ---")
    

    # Pipeline 1. 모델 예측 평점 (R_Model) 생성
    r_model_dict, core_places = calculate_r_model(
        core_df, predictor, user_query_raw, user_companion, user_purpose, user_tourism_type, keyword_query
    )
    
    # Pipeline 2. 컨텍스트 점수 계산 (Dist, Key, Sim)
    context_df = predictor.calculate_all_context_scores(
        type_key=user_tourism_type,
        user_query=user_query_raw,
        prev_place_name=prev_place_name,
        location_coords=location_coords,
        query_keywords=keyword_query
    )
    
    if context_df.empty: return pd.DataFrame()

    # 시나리오별 가중치(Gamma) 결정
    has_prev = bool(prev_place_name)
    has_loc  = bool(location_coords)
    has_key  = bool(keyword_query)
    
    # Default (S1)
    gamma = {'g1': 1.0, 'g2': 0.0, 'g3': 0.0, 'g4': 0.0}
    
    if not has_prev: # 최초
        if not has_loc and not has_key:   gamma = {'g1': 1.0, 'g2': 0.0, 'g3': 0.0, 'g4': 0.0}
        elif not has_loc and has_key:     gamma = {'g1': 0.6, 'g2': 0.0, 'g3': 0.0, 'g4': 0.4}
        elif has_loc and not has_key:     gamma = {'g1': 0.4, 'g2': 0.0, 'g3': 0.6, 'g4': 0.0}  # 노션상 g1 0.7, g3 0.3
        elif has_loc and has_key:         gamma = {'g1': 0.2, 'g2': 0.0, 'g3': 0.4, 'g4': 0.4}  # 노션상 g1 0.4, g3 0.1, g4 0.5
    else: # 연속
        if not has_loc and not has_key:   gamma = {'g1': 0.6, 'g2': 0.2, 'g3': 0.2, 'g4': 0.0}  # 의미적 & 거리적 유사도 동일한 비율로 반영
        elif not has_loc and has_key:     gamma = {'g1': 0.4, 'g2': 0.1, 'g3': 0.1, 'g4': 0.4}  # 패스
        elif has_loc and not has_key:     gamma = {'g1': 0.5, 'g2': 0.1, 'g3': 0.4, 'g4': 0.0}  # 패스
        elif has_loc and has_key:         gamma = {'g1': 0.3, 'g2': 0.1, 'g3': 0.3, 'g4': 0.3}  # 6번째와 한 끗 차이... -> 수정(추후 소수점 2째 자리까지 정밀히..)

    # =================================================================
    # ★ [NEW] 유명 관광지(Popularity) 부스팅 로직 (페르소나 2 대응)
    # =================================================================
    if is_famous_intent:
        print("🔥 [Intent] 유명/인기 장소 탐색 의도 감지 (By LLM) -> Core 부스팅 ON")
        
        # 1. 가중치 조정 (모델 점수 대폭 강화)
        # 모델 점수(g1)는 Core 데이터(유명 장소)가 높은 점수를 받도록 설계되어 있으므로 비중을 높임
        if has_loc:
            # 위치 조건이 있다면 거리 점수(g3)도 중요하므로 40% 유지
            gamma = {'g1': 0.6, 'g2': 0.0, 'g3': 0.4, 'g4': 0.0}
        else:
            # 위치 조건이 없다면 모델 점수(g1) 90% 몰빵 (유사도는 거의 무시)
            gamma = {'g1': 0.7, 'g2': 0.0, 'g3': 0.0, 'g4': 0.3}

    print(f"   [Scenario] Prev={has_prev}, Loc={has_loc}, Key={has_key}, Famous={is_famous_intent} -> Weights: {gamma}")

    # 최종 점수 계산 (Final Formula)
    final_results = []
    
    # 키워드 리스트 전처리 (쉼표나 공백으로 구분된 경우 대비)
    target_keywords = []
    if has_key and keyword_query:
        # "조용한, 한강공원" -> ["조용한", "한강공원"]
        target_keywords = [k.strip() for k in keyword_query.replace(",", " ").split() if len(k) >= 2]

    for _, row in context_df.iterrows():
        name = row['name']
        
        # Pipeline 2 Scores
        w_dist = row['dist_score']
        w_sim  = row['sim_score']
        w_key  = row['key_score']
        
        r_model = r_model_dict.get(name, 3.0)
        
        matched_core = NameMatcher.check_match(core_places, name)
        is_core = bool(matched_core)

        # 1. 기본 점수 계산 (가중치 합)
        base_score = (gamma['g1'] * r_model) + \
                     (gamma['g2'] * w_sim) + \
                     (gamma['g3'] * w_dist) + \
                     (gamma['g4'] * w_key)
        
        final_score = base_score

        # 2. 명시적 키워드 부스팅 (Ranking Promotion)
        if target_keywords:
            boost_factor = 1.0
            
            # Case A: 이름에 키워드가 직접 포함된 경우 (강력한 부스팅)
            # 예: "한강공원" -> "여의도 한강공원" (O)
            for kw in target_keywords:
                if kw in name: 
                    boost_factor += 0.3 # 30% 가산 (확실한 의도 반영)
                    break
            
            # Case B: 의미적 유사도가 매우 높은 경우 (의미 부스팅)
            # 예: "궁궐" -> "경복궁" (이름엔 없지만 w_key 점수가 높음)
            if w_key >= 4.2: # 5점 만점에 4.2 이상이면 매우 관련 높음
                boost_factor += 0.15 # 15% 가산
            
            final_score *= boost_factor
        # 3. ★ [NEW] Core 장소 하드 부스팅 (유명 키워드 감지 시)
        # 사용자가 "유명한 곳"을 원할 때, Core(유명) 장소에 강력한 가산점을 주어 상단 노출 보장
        if is_famous_intent and is_core:
            final_score += 2.0

            
        final_results.append({
            "name": matched_core if is_core else name,
            "real_name": name, 
            "score": final_score,
            "type": "Core" if is_core else "New"
        })
        
    results_df = pd.DataFrame(final_results)

    # [수정] Targeted Min-Max 정규화 (Target: 3.0 ~ 5.0)
    if not results_df.empty:
        max_score = results_df['score'].max()
        min_score = results_df['score'].min()
        
        # 최대 점수가 5.0을 초과하는 경우 (부스팅 발생 시)
        if max_score > 5.0:
            # 목표 점수 하한선과 상한선 설정
            TARGET_MIN = 3.0  # 1.0 -> 3.0으로 상향 (핵심 수정)
            TARGET_MAX = 5.0
            
            if max_score != min_score:
                # 공식: 3.0 + (비율 * 2.0)
                results_df['score'] = TARGET_MIN + \
                    (results_df['score'] - min_score) / (max_score - min_score) * (TARGET_MAX - TARGET_MIN)
            else:
                results_df['score'] = TARGET_MAX # 점수가 다 똑같으면 5.0
        else:
            # 5.0을 넘지 않더라도 최소 1.0보다는 크도록 안전장치
            results_df['score'] = results_df['score'].clip(lower=1.0)
    
    if not results_df.empty:
        # [★수정됨] 점수 순 정렬
        results_df = results_df.sort_values(by='score', ascending=False)
        
        # [★추가됨] Core 장소 중복 제거 (Diversity Logic)
        # 'name' 컬럼은 Core 매칭된 경우 통일된 이름(예: '한강공원')을 가집니다.
        # 따라서 'name'을 기준으로 중복을 제거하면, 
        # 여러 한강공원 지점 중 1등(거리/유사도 고려) 하나만 남습니다.
        results_df = results_df.drop_duplicates(subset=['name'], keep='first')

        # 상위 30개 추출
        results_df = results_df.head(30)
        
        origin_df = predictor.new_places_data.get(user_tourism_type)
        if origin_df is not None:
            results_df = pd.merge(results_df, origin_df[['명칭', '주소', '개요']], left_on='real_name', right_on='명칭', how='left')
            
    return results_df