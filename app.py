import gradio as gr
from openai import OpenAI
import os
import pandas as pd
import numpy as np
import unicodedata 
import json
from geopy import geocoders
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

print("--- 🚀 AI 여행 추천 봇 ---")
try:
    from recommendation_logic import get_combined_recommendations, NameMatcher
    from embedding_logic import EmbeddingPredictor, RestaurantRecommender
except ImportError:
    print("🚨 필수 Python 파일을 찾을 수 없습니다. (recommendation_logic.py 또는 embedding_logic.py)")
    exit()

OPENAI_API_KEY="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"

# 1-1. OpenAI API 로드
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
    if client.api_key is None:
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
    print("✅ [App] OpenAI API 키 로드 성공.")
except Exception as e:
    print(f"🚨 [App] OpenAI API 키 로드 중 오류: {e}")
    client = None

# 1-2. Logic A (관광지/쇼핑/문화) 설정
REGRESSION_MODEL_PATH = "RegressionRating_LGBM_Best.joblib"
predictor = EmbeddingPredictor(base_data_dir="data/embeddings") 
predictor.load_regression_model(REGRESSION_MODEL_PATH)

# [Core Data Load] Scoring_기말.xlsx
DATA_FILE_CORE = "data/Scoring_기말.xlsx"
UI_COMPANIONS = ["혼자", "애인", "가족", "친구", "단체", "기타"]
UI_PURPOSES = ["관광", "체험", "역사", "자연", "휴식", "쇼핑", "문화", "촬영지방문", "기타"]
main_df_A = None
ALL_PLACES_CORE = []

try:
    main_df_A = pd.read_excel(DATA_FILE_CORE)
    # 전처리 (유니코드 정규화 등)
    main_df_A['source'] = main_df_A['source'].str.replace('.xlsx', '', regex=False).str.strip()
    main_df_A['source'] = main_df_A['source'].apply(lambda x: unicodedata.normalize('NFC', str(x)))
    main_df_A['companion'] = main_df_A['companion'].astype(str).str.strip()
    main_df_A['purpose'] = main_df_A['purpose'].astype(str).str.strip()
    
    ALL_PLACES_CORE = main_df_A['source'].unique() 
    
    predictor.setup_ohe_features(ALL_PLACES_CORE, UI_COMPANIONS, UI_PURPOSES)
    print(f"✅ [Core] Scoring 데이터 로드 완료 (총 Target: {len(ALL_PLACES_CORE)}개).")
    
except Exception as e:
    print(f"🚨 [Core] 데이터 로드 실패: {e}")

# [New Data Load & Core Registration]
def core_matcher(target_name):
    return NameMatcher.check_match(ALL_PLACES_CORE, target_name)

files_map = {
    "관광지": "data/관광지_final.xlsx",
    "문화시설": "data/문화시설_final.xlsx",
    "쇼핑": "data/쇼핑_final.xlsx"
}

# 데이터 파일 로드
for type_key, file_path in files_map.items():
    try:
        df_new = pd.read_excel(file_path)
        predictor.load_and_register_data(type_key, df_new, core_matcher)
    except Exception as e:
        print(f"🚨 [Data] {type_key} 로드 실패: {e}")


# 1-3. Logic B (음식점) 데이터 로드
recommender = RestaurantRecommender()
try:
    df_food = pd.read_excel('data/음식점_final.xlsx')
    for col in ['소분류', '대표메뉴', '명칭', '주소', '개요']:
        if col in df_food.columns:
            df_food[col] = df_food[col].fillna('').astype(str).str.strip()
    recommender.load_data_from_df(df_food)
except Exception as e:
    print(f"🚨 [Food] 음식점 데이터 로드 실패: {e}")


# --- 2. NLU & Helper Functions ---

def extract_intent_keywords(user_text):
    """
    LLM을 사용하여 사용자 의도, 위치(거리계산용), 키워드(취향용)를 추출
    """
    if not client: return {"intent": "chat", "location": None, "keyword": None}
    
    system_prompt = """
    당신은 여행 추천 봇의 NLU(자연어 이해) 어시스턴트입니다.
    사용자의 입력을 분석하여 다음 정보를 JSON 형식으로 추출하세요.
    
    키(Keys):
    - "intent": "recommendation" (장소 추천 요청, 질문) 또는 "chat" (일상 대화, 인사)
    - "location_query": 거리 계산의 기준점이 될 구체적인 장소명 (예: "강남역", "서초구", "경복궁"). 없으면 null.
    - "keyword_query": 장소의 분위기, 테마, 특성 등 구체적인 취향 (예: "조용한", "야경이 예쁜", "공원"). 장소명 제외. 없으면 null.
    
    예시 1: "강남역 근처 조용한 카페 추천해줘"
    출력: {"intent": "recommendation", "location_query": "강남역", "keyword_query": "조용한 카페"}
    
    예시 2: "그냥 걷기 좋은 곳 있어?"
    출력: {"intent": "recommendation", "location_query": null, "keyword_query": "걷기 좋은"}

    반드시 유효한 JSON 형식만 반환하세요.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ],
            temperature=0.0
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"NLU Error: {e}")
        return {"intent": "chat", "location": None, "keyword": None}

def find_coordinates(location_name):
    """
    좌표 검색 로직: 1. DB 내부 검색 -> 2. 외부 Geocoding (Nominatim)
    """
    if not location_name: return None
    
    clean_loc = location_name.replace(" ", "").strip()
    print(f"🔎 [Debug] 좌표 검색 시작: '{location_name}'")
    
    # 1. DB 내부 검색
    for name, meta in predictor.core_global_metadata.items():
        clean_name = str(name).replace(" ", "")
        if clean_loc == clean_name:
            if meta.get('lat', 0.0) != 0.0:
                print(f"📍 [Internal DB] 내부 데이터 매칭 성공! -> {name}")
                return (meta['lat'], meta['lon'])
            
    # 2. 외부 Geocoding
    try:
        geolocator = Nominatim(user_agent="travel_bot_v7_final", timeout=5)
        location = geolocator.geocode(location_name)
        if location:
            print(f"✅ [External] 외부 검색 성공: {location.address}")
            return (location.latitude, location.longitude)
        else:
            print("❌ [External] 외부 지도에서도 찾을 수 없음.")
    except GeocoderTimedOut:
        print(f"⏳ [TimeOut] 외부 지도 검색 시간이 초과되었습니다.")
    except Exception as e:
        print(f"⚠️ [Error] Geocoding 오류: {e}")
        
    return None


# --- 3. 핸들러 함수 ---
def handle_chat_submit(message, history, state, tourism_type, companion, purpose, food_category_input):
    history.append({"role": "user", "content": message})
    
    # State 확인
    current_state = state.get("state", "IDLE")
    prev_place = state.get("prev_place", None) # 직전 추천 장소
    
    # 1. NLU 분석
    nlu_result = extract_intent_keywords(message)
    intent = nlu_result.get("intent", "chat")
    loc_query = nlu_result.get("location_query")
    key_query = nlu_result.get("keyword_query")
    
    print(f"💬 [User Input] {message}")
    print(f"🤖 [NLU Result] {nlu_result}")

    # 2. 추천 요청 처리
    if intent == "recommendation" or current_state == "AWAITING_QUERY":
        
        # 필수값 체크
        if not tourism_type:
            history.append({"role": "assistant", "content": "🚨 **관광 타입**을 먼저 선택해주세요!"})
            return history, "", state
            
        if tourism_type == "음식점":
            # [Logic B] 음식점 (랜덤 추천)
            if not food_category_input:
                history.append({"role": "assistant", "content": "🚨 **음식 소분류**를 선택해주세요!"})
                return history, "", state
            
            results = recommender.get_random_recommendations(food_category_input, n=5)
            if not results:
                bot_msg = "해당 카테고리의 데이터가 없습니다."
            else:
                bot_msg = f"🍽️ **{food_category_input}** 맛집 Top 5\n\n"
                for i, r in enumerate(results):
                    bot_msg += (
                        f"**{i+1}. {r['name']}**\n"
                        f" - 메뉴: {r['menu']}\n"
                        f" - 주소: {r['address']}\n\n"
                    )
            history.append({"role": "assistant", "content": bot_msg})
            return history, "", state

        else:
            # [Logic A] 관광지/쇼핑/문화 (Pipeline 1 & 2)
            if not companion or not purpose:
                history.append({"role": "assistant", "content": "🚨 **동반인**과 **목적**을 선택해주세요!"})
                return history, "", state
            
            # 좌표 변환 (거리 컨텍스트용)
            loc_coords = find_coordinates(loc_query)
            
            # 추천 로직 호출 (recommendation_logic.py)
            top5_df = get_combined_recommendations(
                main_df_A, 
                predictor, 
                user_query_raw=message, 
                user_companion=companion, 
                user_purpose=purpose, 
                user_tourism_type=tourism_type,
                prev_place_name=prev_place,  # 직전 장소 (Pipeline 2 SimCon)
                location_coords=loc_coords,  # 쿼리 위치 (Pipeline 2 DistCon)
                keyword_query=key_query      # 쿼리 키워드 (Pipeline 1 & 2)
            )
            
            if top5_df.empty:
                bot_msg = "추천할 장소를 찾지 못했습니다."
            else:
                top_place = top5_df.iloc[0]['real_name'] # 1등 장소 저장
                
                bot_msg = f"🔎 **{tourism_type}** 추천 결과 (Top 5)\n"
                if prev_place: bot_msg += f"🔗 직전 장소 **'{prev_place}'** 연계\n"
                if loc_query: 
                    if loc_coords: bot_msg += f"📍 **'{loc_query}'** 근처\n"
                    else: bot_msg += f"⚠️ **'{loc_query}'** 위치 확인 불가\n"
                if key_query: bot_msg += f"✨ 취향 **'{key_query}'** 반영\n"
                bot_msg += "\n"
                
                rank = 1
                for _, row in top5_df.iterrows():
                    bot_msg += (
                        f"**{rank}. {row['real_name']}** ({row['score']:.2f}점)\n" 
                        f" - 주소: {row.get('주소', '-')}\n"
                        f" - 설명: {row.get('개요', '')[:60]}...\n\n"
                    )
                    rank += 1
                
                # State 업데이트: 현재 1등 장소를 다음 턴의 '직전 장소'로 설정
                state["prev_place"] = top_place
            
            history.append({"role": "assistant", "content": bot_msg})
            state["state"] = "IDLE" # 상태 복귀
            return history, "", state

    else:
        # 단순 대화 (GPT)
        if client:
            try:
                sys_msg = "여행 추천 봇입니다. 짧고 친절하게 대답하세요."
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": sys_msg}] + history[-3:]
                )
                bot_msg = response.choices[0].message.content
                history.append({"role": "assistant", "content": bot_msg})
            except:
                history.append({"role": "assistant", "content": "오류 발생"})
        else:
             history.append({"role": "assistant", "content": "API 키가 없습니다."})
        return history, "", state


# --- 4. UI 구성 ---
UI_FOOD_CATEGORIES = ["한식", "양식", "일식", "중식", "카페", "이색음식점"]

with gr.Blocks(theme=gr.themes.Soft(), title="AI 여행 추천 봇") as demo:
    gr.Markdown("# ✈️ AI 여행 추천 봇")
    
    # State에 prev_place 추가
    chat_state = gr.State(value={"state": "IDLE", "prev_place": None})

    with gr.Row():
        with gr.Column(scale=1, min_width=250): 
            gr.Markdown("### 1. 여행 옵션")
            rb_tourism = gr.Radio(["관광지", "쇼핑", "문화시설", "음식점"], label="관광 타입")
            dd_food_category = gr.Dropdown(UI_FOOD_CATEGORIES, label="음식 소분류 (음식점용)")
            gr.Markdown("---")
            rb_companion = gr.Radio(UI_COMPANIONS, label="동반인")
            dd_purpose = gr.Dropdown(UI_PURPOSES, label="목적")
            gr.Markdown("<br>")
            btn_start = gr.Button("🚀 여행 계획 시작하기", variant="primary", size="lg")

        with gr.Column(scale=4): 
            gr.Markdown("### 2. AI 대화")
            chatbot = gr.Chatbot(elem_id="chat-window", type="messages", bubble_full_width=False, height=650)
            with gr.Row():
                txt_input = gr.Textbox(
                    show_label=False, 
                    placeholder="예: 강남역 근처 조용한 카페 추천해줘",
                    scale=8,
                    container=False
                )
                btn_submit = gr.Button("전송", scale=1, variant="secondary")

    # 시작 버튼 클릭 이벤트
    def handle_start_click(history, state, tourism_type, companion, purpose, food_cat):
        if not tourism_type:
            history.append({"role": "assistant", "content": "🚨 **[관광 타입]**을 먼저 선택해주세요!"})
            return history, state
            
        # 시작 시 State 초기화 (새로운 여행)
        new_state = {"state": "AWAITING_QUERY", "companion": companion, "purpose": purpose, "type": tourism_type, "prev_place": None}
        
        bot_msg = f"👋 **'{companion}'**분들과 **'{purpose}'** 여행을 시작합니다!\n\n원하시는 장소나 분위기를 말씀해주세요. (예: '경복궁 근처', '조용한 분위기')"
        history.append({"role": "assistant", "content": bot_msg})
        return history, new_state

    # 이벤트 리스너 연결
    btn_start.click(
        fn=handle_start_click,
        inputs=[chatbot, chat_state, rb_tourism, rb_companion, dd_purpose, dd_food_category],
        outputs=[chatbot, chat_state]
    )

    txt_input.submit(
        fn=handle_chat_submit,
        inputs=[txt_input, chatbot, chat_state, rb_tourism, rb_companion, dd_purpose, dd_food_category],
        outputs=[chatbot, txt_input, chat_state]
    )
    
    btn_submit.click(
        fn=handle_chat_submit,
        inputs=[txt_input, chatbot, chat_state, rb_tourism, rb_companion, dd_purpose, dd_food_category],
        outputs=[chatbot, txt_input, chat_state]
    )

if __name__ == "__main__":
    demo.launch()