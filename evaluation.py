import pandas as pd
import numpy as np
import ast
import math
import os
import time
from openai import OpenAI

# ==========================================
# [환경 설정] OpenAI API 키 
# ==========================================
os.environ["OPENAI_API_KEY"] = "키 입력"
client = OpenAI()

# ==========================================
# 평가지표 계산 로직 
# ==========================================
def get_hit_rate(row, k): 
    return 1 if row['target_id'] in row['rec_list'][:k] else 0

def get_mrr(row, k):
    try: return 1 / (row['rec_list'][:k].index(row['target_id']) + 1)
    except ValueError: return 0

def get_ndcg(row, k):
    try: return 1 / math.log2((row['rec_list'][:k].index(row['target_id']) + 1) + 1)
    except ValueError: return 0

def get_map(row, k): 
    return get_mrr(row, k)

# ==========================================
# 환각 평가 로직
# ==========================================
def check_hallucination(row, valid_db_ids, df_db, client):
    # PopRec처럼 system_response가 비어있는 경우
    if pd.isna(row['system_response']) or str(row['system_response']).strip().lower() == 'nan':
        return np.nan 

    sys_text = str(row['system_response'])

    # [1단계] DB 매칭 (가짜 ID 판별)
    for rec_id in row['rec_list']:
        if rec_id not in valid_db_ids:
            return 1 
    
    # [2단계] GPT-4o를 이용한 사실 검증 
    try:
        db_facts = df_db[df_db['target_id'].isin(row['rec_list'])]['target_keywords'].tolist()
        fact_context = " | ".join(map(str, db_facts))

        prompt = f"""
        당신은 추천 시스템의 답변이 사실인지 검증하는 엄격한 평가자입니다.
        시스템이 생성한 답변이 실제 데이터베이스의 사실과 일치하는지 확인하세요.

        [실제 DB 정보(Ground Truth)]
        {fact_context}

        [시스템의 답변]
        {sys_text}

        지시사항:
        1. 시스템의 답변에 DB 정보와 모순되거나, 존재하지 않는 허위 사실(Hallucination)이 포함되어 있다면 '1'을 출력하세요.
        2. 허위 사실이 없고 사실에 기반한 답변이라면 '0'을 출력하세요.
        3. 다른 설명 없이 오직 숫자 1 또는 0만 출력하세요.
        """

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful and strict factual verifier."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0, 
            max_tokens=5  
        )
        
        result = int(response.choices[0].message.content.strip())
        return result
        
    except Exception as e:
        print(f"API 호출 에러 (scenario: {row['scenario_id']}): {e}")
        return np.nan  


# ==========================================
# [메인 엔진] 통합 평가 및 자동 복구 파이프라인
# ==========================================
def run_evaluation_pipeline(log_csv_path, db_csv_path, final_output_path, k=5):
    print(f"\n[{log_csv_path}] 성능 평가 시작\n")
    
    # 데이터 로드 및 전처리
    df_log = pd.read_csv(log_csv_path)
    df_db = pd.read_csv(db_csv_path)
    valid_db_ids = set(df_db['target_id'].tolist()) 
    
    df_log['rec_list'] = df_log['rec_list'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # 정확도 평가
    df_log[f'hit_rate@{k}'] = df_log.apply(lambda row: get_hit_rate(row, k), axis=1)
    df_log[f'mrr@{k}'] = df_log.apply(lambda row: get_mrr(row, k), axis=1)
    df_log[f'ndcg@{k}'] = df_log.apply(lambda row: get_ndcg(row, k), axis=1)
    df_log[f'map@{k}'] = df_log.apply(lambda row: get_map(row, k), axis=1)
    
    # 1차 환각 검사
    print("⏳ GPT-4o 환각 검사 시작")
    df_log['is_hallucination'] = df_log.apply(lambda row: check_hallucination(row, valid_db_ids, df_db, client), axis=1)

    # 재시도 로직 
    missing_mask = df_log['system_response'].notna() & df_log['is_hallucination'].isna()
    missing_indices = df_log[missing_mask].index
    missing_count = len(missing_indices)
    
    if missing_count > 0:
        print(f"\nAPI 오류로 누락된 {missing_count}개의 데이터 재심사")
        for idx in missing_indices:
            row = df_log.loc[idx]
            try:
                result = check_hallucination(row, valid_db_ids, df_db, client)
                time.sleep(1.5) # 안전한 API 호출을 위한 대기 시간
                df_log.at[idx, 'is_hallucination'] = result 
                print(f"  └ Scenario {row['scenario_id']}: 복구 완료 ✅")
            except Exception as e:
                print(f"  └ Scenario {row['scenario_id']}: 최종 실패 ❌ ({e})")
    else:
        print("\n API 누락 없이 한 번에 모두 평가됨")

    # 최종 요약 산출
    summary_df = pd.DataFrame({
        'Metric': [f'Hit Rate@{k}', f'MRR@{k}', f'NDCG@{k}', f'MAP@{k}', 'Hallucination Rate'],
        'Average Score': [
            df_log[f'hit_rate@{k}'].mean(),
            df_log[f'mrr@{k}'].mean(),
            df_log[f'ndcg@{k}'].mean(),
            df_log[f'map@{k}'].mean(),
            df_log['is_hallucination'].mean() 
        ]
    })
    
    # 최종 파일 저장
    df_log.to_csv(final_output_path, index=False)
    print(f"\n 평가 완료 -> 최종 파일: [{final_output_path}]")
    print("-" * 40)
    print(summary_df)
    print("-" * 40)

    return df_log, summary_df

# ==========================================
# [실행부] 실제로 코드가 돌아가는 곳
# ==========================================
if __name__ == "__main__":
    MODEL_LOG_FILE = '(모델이름)_log.csv'   
    TOUR_DB_FILE = '(관광지DB).csv'          
    FINAL_EVAL_FILE = '(모델이름)_eval.csv' 
    
    df_final, df_summary = run_evaluation_pipeline(
        log_csv_path=MODEL_LOG_FILE, 
        db_csv_path=TOUR_DB_FILE, 
        final_output_path=FINAL_EVAL_FILE, 
        k=5
    )