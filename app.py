import gradio as gr
from openai import OpenAI
import os
import pandas as pd
import numpy as np
import unicodedata 
import json
import zipfile
import shutil
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from huggingface_hub import hf_hub_download
from huggingface_hub import hf_hub_download, HfApi
from huggingface_hub.utils import EntryNotFoundError
import base64
import random
import warnings
import csv
import datetime
import uuid
import shutil
shutil.rmtree("data/images", ignore_errors=True)


# 로직 파일 임포트
try:
    from embedding_logic import EmbeddingPredictor, haversine_vectorized
    from recommendation_logic import get_combined_recommendations, NameMatcher
except ImportError:
    print("🚨 필수 Python 파일을 찾을 수 없습니다.")
    exit()

# --- 1. 시스템 설정 및 API 키 ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = None
if OPENAI_API_KEY:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        print("✅ [App] OpenAI API 키 로드 성공.")
    except Exception as e:
        print(f"🚨 [App] OpenAI API 키 로드 중 오류: {e}")

# --- 2. [핵심 수정] 데이터 준비 (지능형 압축 해제) ---
# data/images 폴더가 없으면 다운로드 및 정리 시도
if not os.path.exists("data/images"):
    print("📦 [System] 이미지 데이터셋 다운로드 시작...")
    try:
        # 1. 다운로드
        zip_path = hf_hub_download(
            repo_id="KimJiwoo/My-Seoul-Images", # 데이터셋 ID
            filename="images.zip", 
            repo_type="dataset"
        )
        print("✅ [System] 다운로드 완료! 압축 해제 시작...")
        
        # 2. 압축 해제
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("data/")
            
        print("📂 [System] 폴더 구조 자동 보정 중...")
        
        # [Case 1] 'images_fixed'라는 이름으로 풀렸을 경우 -> 'images'로 변경
        if os.path.exists("data/images_fixed"):
            print("🔧 [Fix] 'images_fixed' 폴더 발견 -> 'images'로 변경")
            if os.path.exists("data/images"): shutil.rmtree("data/images")
            shutil.move("data/images_fixed", "data/images")
            
        # [Case 2] 'images/images' 처럼 이중 폴더로 풀렸을 경우 -> 꺼내기
        elif os.path.exists("data/images/images"):
            print("🔧 [Fix] 이중 폴더 구조 발견 -> 파일 꺼내기")
            for f in os.listdir("data/images/images"):
                shutil.move(os.path.join("data/images/images", f), "data/images")
            shutil.rmtree("data/images/images")

        # [Case 3] 압축이 data/ 폴더에 바로 풀려서 jpg들이 흩뿌려진 경우 -> 모으기
        # (혹시 몰라 data 폴더에 jpg 파일이 있으면 images 폴더로 이동시킴)
        jpg_files = [f for f in os.listdir("data") if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if jpg_files:
            print(f"🔧 [Fix] 흩어진 이미지 파일 {len(jpg_files)}개 발견 -> 'images' 폴더로 이동")
            os.makedirs("data/images", exist_ok=True)
            for f in jpg_files:
                shutil.move(os.path.join("data", f), os.path.join("data/images", f))

        # 최종 확인
        if os.path.exists("data/images"):
            file_count = len(os.listdir("data/images"))
            print(f"✅ [System] 이미지 준비 완료! (총 {file_count}장 감지됨)")
        else:
            print("🚨 [System] 경고: 'data/images' 폴더를 결국 찾지 못했습니다. 압축 파일 구조를 확인하세요.")
            print(f"📂 현재 data 폴더 내용: {os.listdir('data')}")

    except Exception as e:
        print(f"🚨 [System] 이미지 다운로드/해제 실패: {e}")

# --- 3. 로그 저장 설정 ---
LOG_REPO_ID = "hyeonjeong2203/K-Travel-Logs" 
LOG_FILE = "user_logs.csv"

def save_log(session_id, event_type, lang, t_type, comp, purp, food, u_input, b_response):
    """
    사용자 로그를 Hugging Face Dataset에 CSV로 누적 저장하는 함수
    (호출하는 쪽의 인자와 매칭되도록 수정됨)
    """
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("🚨 [Log] 토큰(HF_TOKEN)이 없어서 로그를 저장할 수 없습니다.")
        return

    # 1. 현재 시간 및 데이터 구성
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 데이터프레임 생성 (컬럼 순서 정리)
    new_data = pd.DataFrame([{
        "timestamp": timestamp,
        "session_id": session_id,
        "event_type": event_type,
        "language": lang,
        "tourism_type": t_type,
        "companion": comp,
        "purpose": purp,
        "food_category": food,
        "user_input": u_input,
        "bot_response": b_response
    }])

    # 2. 기존 로그 다운로드 시도 (누적 저장을 위해)
    try:
        # print("📥 [Log] 기존 로그 다운로드 중...") # 너무 잦은 출력 방지 위해 주석 처리 가능
        downloaded_path = hf_hub_download(
            repo_id=LOG_REPO_ID,
            filename=LOG_FILE,
            repo_type="dataset",
            token=hf_token
        )
        old_df = pd.read_csv(downloaded_path)
        final_df = pd.concat([old_df, new_data], ignore_index=True)
    except (EntryNotFoundError, FileNotFoundError, Exception) as e:
        # 파일이 없으면(첫 로그라면) 그냥 새 데이터만 사용
        print(f"ℹ️ [Log] 기존 로그 없음 또는 다운로드 실패({e}). 새로 생성합니다.")
        final_df = new_data

    # 3. 로컬에 임시 저장 후 업로드
    final_df.to_csv(LOG_FILE, index=False, encoding="utf-8-sig")
    
    try:
        api = HfApi(token=hf_token)
        api.upload_file(
            path_or_fileobj=LOG_FILE,
            path_in_repo=LOG_FILE,
            repo_id=LOG_REPO_ID,
            repo_type="dataset",
            commit_message=f"Log update: {timestamp}"
        )
        print("✅ [Log] 로그 저장 완료!")
    except Exception as e:
        print(f"🚨 [Log Error] 업로드 실패: {e}")
# =========================================================

# --- 4. Logic A (관광지/쇼핑/문화) 데이터 로드 ---
REGRESSION_MODEL_PATH = "RegressionRating_LGBM_Best.joblib"
predictor = EmbeddingPredictor(base_data_dir="data/embeddings") 
if os.path.exists(REGRESSION_MODEL_PATH):
    predictor.load_regression_model(REGRESSION_MODEL_PATH)

DATA_FILE_CORE = "data/Scoring_기말.xlsx"
UI_COMPANIONS = ["혼자", "애인", "가족", "친구", "단체", "기타"]
UI_PURPOSES = ["관광", "체험", "역사", "자연", "휴식", "쇼핑", "문화", "촬영지방문", "기타"]
ALL_PLACES_CORE = []

try:
    if os.path.exists(DATA_FILE_CORE):
        main_df_A = pd.read_excel(DATA_FILE_CORE)
        main_df_A['source'] = main_df_A['source'].astype(str).str.replace('.xlsx', '').str.strip()
        main_df_A['source'] = main_df_A['source'].apply(lambda x: unicodedata.normalize('NFC', str(x)))
        main_df_A['companion'] = main_df_A['companion'].astype(str).str.strip()
        main_df_A['purpose'] = main_df_A['purpose'].astype(str).str.strip()
        ALL_PLACES_CORE = main_df_A['source'].unique() 
        predictor.setup_ohe_features(ALL_PLACES_CORE, UI_COMPANIONS, UI_PURPOSES)
        print(f"✅ [Core] Scoring 데이터 로드 완료.")
except Exception as e:
    print(f"🚨 [Core] 데이터 로드 실패: {e}")

def core_matcher(target_name):
    return NameMatcher.check_match(ALL_PLACES_CORE, target_name)

files_map = {
    "관광지": "data/관광지_final.xlsx",
    "문화시설": "data/문화시설_final.xlsx",
    "쇼핑": "data/shopping.xlsx"
}

for type_key, file_path in files_map.items():
    try:
        if os.path.exists(file_path):
            df_new = pd.read_excel(file_path)
            predictor.load_and_register_data(type_key, df_new, core_matcher)
        else:
            print(f"⚠️ 파일 없음: {file_path}")
    except Exception as e:
        print(f"🚨 [Data] {type_key} 로드 실패: {e}")

# --- 5. Logic B (음식점) 데이터 로드 ---
try:
    if os.path.exists('data/음식점_final.xlsx'):
        df_food = pd.read_excel('data/음식점_final.xlsx')
        rename_map = {'name': '명칭', 'menu': '대표메뉴', 'address': '주소', 'description': '개요'}
        df_food.rename(columns=rename_map, inplace=True)
        for col in ['소분류', '대표메뉴', '명칭', '주소', '개요']:
            if col in df_food.columns:
                df_food[col] = df_food[col].fillna('').astype(str).str.strip()
        predictor.load_and_register_data("음식점", df_food, None)
except Exception as e:
    print(f"🚨 [Food] 음식점 데이터 로드 실패: {e}")

# --- 6. 번역 및 이미지 매니저 ---
class TranslationManager:
    def __init__(self, client):
        self.client = client

    def translate_to_korean(self, text, user_lang):
        if user_lang == "한국어" or not text: return text
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Translate to Korean. Output only translated text."},
                    {"role": "user", "content": text}
                ], temperature=0.0
            )
            return response.choices[0].message.content
        except: return text

    def translate_to_user_lang(self, text, target_lang):
        if target_lang == "한국어" or not text: return text
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"Translate this Korean text to {target_lang}. Keep emojis and formatting."},
                    {"role": "user", "content": text}
                ], temperature=0.1
            )
            return response.choices[0].message.content
        except: return text

translator = TranslationManager(client)

class ImageManager:
    def __init__(self, mapping_file="data/image_mapping.json"):
        self.mapping_file = mapping_file
        self.image_map = {} 
        self._load_map()

    def _normalize_name(self, name):
        normalized_str = unicodedata.normalize('NFC', str(name)) 
        return normalized_str.replace(" ", "").replace("_", "").strip().lower()

    def _load_map(self):
        print(f"📂 [ImageManager] 이미지 맵 로드 중... ({self.mapping_file})")
        if os.path.exists(self.mapping_file):
            try:
                with open(self.mapping_file, "r", encoding="utf-8") as f:
                    self.image_map = json.load(f)
                print(f"✅ [ImageManager] 매핑 로드 완료. (키 개수: {len(self.image_map)})")
            except Exception as e:
                print(f"🚨 [ImageManager] 매핑 파일 읽기 실패: {e}")
        else:
            print(f"⚠️ [ImageManager] 매핑 파일이 없습니다: {self.mapping_file}")
            self.image_map = {}

    def get_image_tag(self, place_name, tourism_type=None):
        if not place_name or not self.image_map: return ""
        
        target_clean = self._normalize_name(place_name)
        candidate_paths = []

        if target_clean in self.image_map:
            candidate_paths.extend(self.image_map[target_clean])
            
        for key, paths in self.image_map.items():
            if target_clean in key: 
                candidate_paths.extend(paths)
            elif len(key) >= 2 and key in target_clean: 
                candidate_paths.extend(paths)

        candidate_paths = list(set(candidate_paths))
        if not candidate_paths: return ""
            
        select_count = min(len(candidate_paths), 3)
        selected_paths = random.sample(candidate_paths, select_count)
        return self._render_gallery(place_name, selected_paths)

    def _render_gallery(self, alt_text, path_list):
        html_imgs = ""
        for idx, rel_path in enumerate(path_list):
            if not os.path.exists(rel_path):
                # 족보 경로가 틀렸을 경우 대비 (data/images/filename.jpg 만 추출해서 다시 찾기)
                fname = os.path.basename(rel_path)
                fallback_path = os.path.join("data/images", fname)
                if os.path.exists(fallback_path):
                    rel_path = fallback_path
                else:
                    continue

            try:
                with open(rel_path, "rb") as img_file:
                    b64_string = base64.b64encode(img_file.read()).decode('utf-8')
                
                ext = os.path.splitext(rel_path)[1].lower().replace('.', '')
                if ext == 'jpg': ext = 'jpeg'
                img_src = f"data:image/{ext};base64,{b64_string}"
                
                unique_alt = f"IMG_{alt_text}_{idx}"
                html_imgs += (
                    f"<img src='{img_src}' alt='{unique_alt}' "
                    f"style='height: 200px; width: auto; flex-shrink: 0; border-radius: 8px; margin-right: 10px; cursor: zoom-in;' >"
                )
            except: continue

        return f"<div style='display: flex; overflow-x: auto; padding-bottom: 10px; margin: 10px 0;'>{html_imgs}</div>"

image_manager = ImageManager("data/image_mapping.json")

# --- 2. NLU & Helper Functions ---
def extract_intent_keywords(user_text):
    """
    LLM을 사용하여 사용자 의도, 위치(거리계산용), 키워드(취향용), 그리고 '선택 의도'를 추출
    """
    if not client: return {"intent": "chat", "location": None, "keyword": None, "selection": None}
    
    system_prompt = """
    당신은 여행 추천 봇의 NLU(자연어 이해) 어시스턴트입니다.
    사용자의 입력을 분석하여 다음 정보를 JSON 형식으로 추출하세요.
    
    [1. Intent 분류 규칙]
    - "recommendation": 장소 추천 요청, 조건 변경 ("맛집 알려줘", "조용한 곳으로", "한강공원 가고 싶어")
    - "selection": 추천 목록 중 하나를 확정/선택, 결정 의사 ("1번으로 갈래", "거기로 정했어", "경복궁 갈래", 5번, 3)
    - "detail": 특정 장소에 대한 구체적인 정보/사실 확인 요청 ("경복궁 주차 돼?", "거기 메뉴 뭐 있어?", "어떤 곳이야?")
    - "chat": 단순 인사, 추천/정보와 무관한 일상 대화 ("안녕", "심심해")
    - "pass": 현재 추천된 장소들이 마음에 들지 않아 다른/다음 순위의 장소를 보여달라고 요청 ("다른 곳 추천해줘", "별로야", "마음에 안 들어", "이게 다야?", "더 보여줘", "다른 곳 없을까?", "이게 최선이야?")

    [2. 키(Keys) 추출 규칙]
    - "location_query": 검색의 '범위'나 '기준 지역' ("강남역 근처", "홍대쪽", "서초구"). 없으면 null.
    ★ 주의: 사용자가 '가고 싶다'고 직접 언급한 목적지는 location이 아닙니다. 
    - "keyword_query": (recommendation용) 장소의 분위기, 테마 등 구체적인 취향 또는 특정 장소 이름("조용한", "야경이 예쁜", "한강 공원"). 없으면 null.
    ★ 팁: "한강공원 가고 싶어"에서 '한강공원'은 목적지이므로 keyword_query입니다.
    - "selection_query": (selection용) 사용자가 선택한 대상 ("1번", "첫번째", "경복궁", "마지막"). 없으면 null.
    - "detail_query": (detail용) 정보를 묻는 대상 장소명 ("경복궁", "그 식당", '거기'). 언급이 없으면 null.
    - "is_relative_location": (boolean) "이 주변", "여기 근처", "거기서 가까운" 등 직전 장소를 기준으로 삼는 표현이 있는지 여부.
    - "is_famous_intent": (boolean) 
        - 사용자가 "유명한", "인기 있는", "핫플", "대표적인", "필수 코스" 등을 원하거나,        
        - "한국이 처음이라", "잘 몰라서 추천해줘" 처럼 **대중적이고 검증된 장소(Core)**를 원하는 뉘앙스가 있으면 **true**.
        - 그 외 특정 취향(조용한, 이색적인)을 찾으면 **false**.

    [3. 예시 데이터]
    - "강남역 맛집 추천해줘" -> {"intent": "recommendation", "location_query": "강남역", "keyword_query": "맛집", "detail_query": null} (기준점이므로 location)
    - "3번으로 갈래" -> {"intent": "selection", "selection_query": "3번", "detail_query": null}
    - "그라운드 시소 서촌갈래" -> {"intent": "selection", "selection_query": "그라운드 시소 서촌", "detail_query": null}
    - "한강공원 가고 싶어" -> {"intent": "recommendation", "location_query": null, "keyword_query": "한강공원", "detail_query": null} (목적지이므로 keyword)
    - "경복궁 주차장 있어?" -> {"intent": "detail", "detail_query": "경복궁", "location_query": null}
    - "여기로 갈게" -> {"intent": "selection", "keyword_query": null, "selection_query": "여기", "detail_query": null} (대상은 문맥에서 추론)
    - "그곳으로 정했어" -> {"intent": "selection", "selection_query": "그곳", "detail_query": null} (대상은 문맥에서 추론)
    - "거기 설명 좀 해줘" -> {"intent": "detail", "detail_query": null, "location_query": null} (대상은 문맥에서 추론)
    - "이 주변에 카페 있어?" -> {"intent": "recommendation", "location_query": null, "keyword_query": "카페", "is_relative_location": true}
    - "한국 처음인데 갈만한 곳 추천해줘" -> {"intent": "recommendation", "is_famous_intent": true, ...}
    
    [4. 주의사항]
    - 위치(Location) vs 키워드(Keyword):
       - "강남역 근처", "홍대쪽" -> 강남역은 검색의 '기준점(Radius)'이므로 'location_query'.
       - "한강공원 가고 싶어", "놀이공원 찾아줘" -> 한강공원은 가고자 하는 '목적지(Destination)'이므로 'keyword_query'. (location은 null).
    
    - 추천(Recommendation) vs 선택(Selection):
       - "~가고 싶어", "~어때?", "~찾아줘" -> 탐색 단계이므로 'intent: recommendation'.
       - "~갈래", "~로 할게", "~로 정했어" -> 결정 단계이므로 'intent: selection'.

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
        return {"intent": "chat", "location": None, "keyword": None, "selection": None}

def find_coordinates(location_name, place_type=None):
    """
    좌표 검색 로직: 
    1. [최우선] 지정된 place_type 데이터셋 검색 (가장 빠르고 정확함)
    2. [차선] 전체 데이터셋 우선순위 검색 (Fallback)
    3. [최후] 외부 Geocoding
    """
    if not location_name: return None
    
    clean_loc = str(location_name).replace(" ", "").strip()
    print(f"🔎 [Debug] 좌표 검색 시작: '{location_name}' (Type Hint: {place_type})")
    
    # 1. 지정된 관광 타입(place_type)이 있다면 해당 데이터셋 먼저 검색
    if place_type and predictor and hasattr(predictor, 'new_places_data'):
        # "관광지", "음식점" 등의 키가 존재하는지 확인
        if place_type in predictor.new_places_data:
            df = predictor.new_places_data[place_type]
            
            # 1차 시도: 정확 일치
            for idx, row in df.iterrows():
                p_name = str(row['명칭']).replace(" ", "")
                if clean_loc == p_name:
                    lat = row.get('위도', 0.0)
                    lon = row.get('경도', 0.0)
                    if lat != 0.0 and lon != 0.0:
                        print(f"📍 [Type Match] '{place_type}' 데이터셋에서 발견! -> {row['명칭']}")
                        return (lat, lon)
            
            # 2차 시도: 포함 관계 (예: '한복남' -> '한복남 경복궁점')
            for idx, row in df.iterrows():
                p_name = str(row['명칭']).replace(" ", "")
                if clean_loc in p_name:
                    lat = row.get('위도', 0.0)
                    lon = row.get('경도', 0.0)
                    if lat != 0.0 and lon != 0.0:
                        print(f"📍 [Type Match] '{place_type}' 데이터셋에서 포함 검색 성공! -> {row['명칭']}")
                        return (lat, lon)

    # 2. 전체 데이터셋 검색 
    # 타입 힌트가 없거나, 힌트 데이터셋에서 못 찾았을 경우 전체를 뒤집니다.
    search_priority = ["관광지", "문화시설", "쇼핑", "음식점"]
    
    if predictor and hasattr(predictor, 'new_places_data'):
        for type_key in search_priority:
            # 위에서 이미 검색한 타입은 중복 검색 방지
            if type_key == place_type: continue 
            if type_key not in predictor.new_places_data: continue
                
            df = predictor.new_places_data[type_key]
            for idx, row in df.iterrows():
                p_name = str(row['명칭']).replace(" ", "")
                if clean_loc in p_name: 
                    lat = row.get('위도', 0.0)
                    lon = row.get('경도', 0.0)
                    if lat != 0.0 and lon != 0.0:
                        print(f"📍 [All DB] 전체 데이터({type_key}) 검색 성공! -> {row['명칭']}")
                        return (lat, lon)

    # 3. 외부 Geocoding (Nominatim) - 최후의 수단
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

def get_place_context_for_rag(target_name):
    """
    모든 로드된 데이터셋(관광, 쇼핑, 문화, 음식점)에서 target_name을 검색하고,
    해당 행(Row)의 모든 정보를 텍스트로 변환하여 반환 (RAG용 Context)
    """
    context_text = ""
    found = False
    
    # 검색 대상 이름 정규화 (공백 제거)
    clean_target = str(target_name).replace(" ", "")

    # predictor에 로드된 모든 데이터셋(관광지, 문화시설, 쇼핑, ★음식점 포함★) 검색
    if predictor and hasattr(predictor, 'new_places_data'):
        for type_key, df in predictor.new_places_data.items():
            
            # 데이터프레임 순회
            for idx, row in df.iterrows():
                # 데이터의 장소명 정규화
                # (음식점 데이터도 '명칭' 컬럼으로 통일해서 로드했으므로 '명칭' 사용 가능)
                # 만약 안전장치가 필요하다면: row_name = str(row.get('명칭', row.get('name', ''))).replace(" ", "")
                row_name = str(row['명칭']).replace(" ", "")
                
                # 부분 일치 확인
                if clean_target in row_name:
                    # 발견! 정보 추출
                    context_text += f"=== 장소: {row['명칭']} ({type_key}) ===\n"
                    
                    for col in df.columns:
                        val = str(row[col]).strip()
                        # 'nan', 'NaT' 등 결측치와 빈 값 제외하고 출력
                        if val and val.lower() != "nan" and val.lower() != "nat":
                            context_text += f"- {col}: {val}\n"
                            
                    context_text += "\n"
                    found = True
                    break # 장소를 찾았으면 내부 루프 종료
            
            if found: break # 장소를 찾았으면 외부 루프 종료

    return context_text if found else None


# --- 3. 핸들러 함수 ---
def handle_chat_submit(message, history, state, tourism_type, companion, purpose, food_category_input, language_select):
    # Gradio에서 넘어온 언어 설정(language_select)을 로직 변수(lang_code)에 할당
    lang_code = language_select 

    # UI 선택값(일본어/영어) -> 로직용 한국어로 변환
    if not lang_code: lang_code = "ko" # 이제 lang_code가 존재하므로 에러 없음

    # 1. 현재 언어 데이터(t)와 한국어 원본(ko_t) 준비
    t = UI_TEXT.get(lang_code, UI_TEXT["ko"]) 
    ko_t = UI_TEXT["ko"]

    # 2. 매핑 함수: "현재 언어 리스트에서 몇 번째인지 찾아서 -> 한국어 리스트 값으로 변환"
    def to_korean(val, key):
        curr_list = t.get(key, [])   # 예: ["観光地", ...]
        ko_list = ko_t.get(key, [])  # 예: ["관광지", ...]
        
        # 값이 있고 현재 리스트에 존재하면 매핑
        if val and val in curr_list:
            idx = curr_list.index(val)
            if idx < len(ko_list):
                return ko_list[idx]
        return val # 없으면 원본 반환


    # 3. 변수 덮어쓰기 (이제부터 로직은 한국어로 실행됩니다)
    tourism_type = to_korean(tourism_type, 'choices_type')
    companion = to_korean(companion, 'choices_companion')
    purpose = to_korean(purpose, 'choices_purpose')
    food_category_input = to_korean(food_category_input, 'choices_food')
   

    # --- [Step 1] Input Translation (입력 번역: 외국어 -> 한국어) ---
    user_display_msg = message  # 사용자가 입력한 메시지(영어 등)를 보관해둡니다 (화면 표시용).
    logic_msg = message   # 로직 처리를 위한 메시지 변수 (한국어)

    # [수정] 숫자만 입력된 경우 (예: "1", "2") -> 번역 API 호출 SKIP & '번' 붙이기
    # 이유: GPT가 짧은 숫자를 만나면 번역 대신 엉뚱한 말을 할 위험이 있음.
    if message.strip().isdigit():
        # 로직(NLU)이 더 잘 알아듣도록 '번'을 붙여서 한국어로 전달
        logic_msg = message.strip() + "번" 
        print(f"🔢 [Skip Trans] 숫자 입력 감지: '{message}' -> '{logic_msg}' (API 호출 생략)")

    # 그 외 외국어인 경우에만 번역 실행
    elif language_select != "한국어":
        print(f"🌍 [Input Trans] '{message}' -> translating to Korean...")
        logic_msg = translator.translate_to_korean(message, language_select)
        print(f"🇰🇷 [Input Result] {logic_msg}")

    # 히스토리에는 사용자가 입력한 원문 표시
    history.append({"role": "user", "content": user_display_msg})
    
    # State 초기화 확인
    if "prev_place" not in state: state["prev_place"] = None
    if "current_candidates" not in state: state["current_candidates"] = [] 
    if "page_index" not in state: state["page_index"] = 0
    
    current_prev = state.get("prev_place")
    gallery_html = ""
    bot_msg = "" # [중요] 최종 답변을 담을 변수 미리 선언
    img_placeholder_map = {}    # ★ [NEW] 이미지 태그를 잠시 보관할 딕셔너리

    # 1. NLU 분석 (번역된 logic_msg 사용!)
    nlu_result = extract_intent_keywords(logic_msg)
    intent = nlu_result.get("intent", "chat")
    loc_query = nlu_result.get("location_query")
    key_query = nlu_result.get("keyword_query")
    sel_query = nlu_result.get("selection_query")
    det_query = nlu_result.get("detail_query")
    is_relative = nlu_result.get("is_relative_location", False) #  추가된 부분
    is_famous = nlu_result.get("is_famous_intent", False) # ★ [NEW] 유명 장소 의도 플래그 추출
  
    print(f"💬 [User] {logic_msg}")
    print(f"🤖 [NLU] {nlu_result}")

    # [추천 설명 부분 추가 함수]
    def get_smart_description(text):
        if not text or str(text) == 'nan': return ""
        clean_text = str(text).strip()
        sentences = [s.strip() for s in clean_text.split('.') if s.strip()] # 마침표 분리
        if not sentences: return clean_text[:80]   # 문장 구분이 안 되면 그냥 80자에서 자름 (점 없이)
        limit_cnt = 3 if len(sentences[0]) <= 50 else 2          # 첫 문장이 짧으면(50자 이하) 3문장, 길면 2문장
        summary = ". ".join(sentences[:limit_cnt])
        return summary + "."

    # [Core Logic] 각 Intent별 처리
    # (주의: 여기서 return 하지 않고 bot_msg만 채웁니다)
    # --- [분기 1] Detail (상세 정보 질문 - RAG) ---
    if intent == "detail":
        target_place = det_query
        
        # 대상이 명시되지 않았을 때 추론
        pronouns = ["여기", "거기", "이곳", "그곳", "저기", "요기"] # target_place가 없거나 대명사라면 -> 직전 장소(current_prev) 사용
        if not target_place or any(p in str(target_place).strip() for p in pronouns):
            if current_prev:
                target_place = current_prev
            elif state["current_candidates"]:
                # 후보군 중 첫 번째를 기본값으로
                first_item = state["current_candidates"][0]
                target_place = first_item.get('real_name', first_item.get('name'))
        
        if target_place:
            rag_context = get_place_context_for_rag(target_place)
            if rag_context:
                sys_prompt = f"""
                당신은 사용자의 즐거운 여행을 돕는 친절한 여행 가이드입니다.
                아래 [제공된 정보]를 바탕으로 사용자의 질문에 따뜻하게 5문장 이내로 답해주세요.
                정보가 있다면 참고해서 설명하고, 없다면 솔직하게 모른다고 하세요.

                [답변 생성 규칙]
                - 톤앤매너: 딱딱한 '다나까'체(~입니다)보다는 부드러운 '해요'체(~예요, ~랍니다)를 사용하세요.
                - 이모티콘 활용: 내용에 어울리는 이모지(🚗, 🌳, 🧚🏻‍♂️, 🕶️, 🌈, ☘️)를 적절히 섞어서 답변을 지루하지 않게 해주세요.
                - 공감과 추천: 정보 전달을 넘어, "산책하기 정말 좋아요!", "주차 걱정은 없겠네요!"와 같이 사용자 입장에서 공감하는 멘트를 덧붙여주세요.
                - 정보 부재 시: 정보가 없다면 "아쉽게도 그 부분은 제가 알 수가 없네요 😢"처럼 부드럽게 표현해주세요.

                [제공된 정보]
                {rag_context}
                """
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": message}],
                        temperature=0.1
                    )
                    bot_msg = response.choices[0].message.content
                except: bot_msg = "죄송합니다. 답변을 생성하는 중 오류가 발생했습니다."
            else:
                bot_msg = f"죄송합니다. 데이터베이스에서 '{target_place}'에 대한 상세 정보를 찾을 수 없습니다."
        else:
            bot_msg = "어떤 장소에 대해 궁금하신가요? 장소 이름을 말씀해 주세요."


    # --- [분기 2] 사용자 선택 (Selection) ---
    elif intent == "selection" and sel_query:
        selected_name = None
        import re
        # [Helper] 데이터셋마다 다른 이름 키(key)를 안전하게 가져오는 함수 정의
        def get_safe_name(item_dict):
            # 1순위: real_name (관광지), 2순위: name (관광지 백업), 3순위: 명칭 (음식점)
            return item_dict.get('real_name', item_dict.get('name', item_dict.get('명칭')))

        candidates = state.get("current_candidates", [])
        
        # 현재 페이지 정보 계산
        current_page = state.get("page_index", 0)
        page_size = 5
        start_offset = current_page * page_size
        
        # 현재 페이지에 실제로 보여지는 아이템 범위 계산
        remaining_items = len(candidates) - start_offset
        visible_count = min(remaining_items, page_size)
        end_offset = start_offset + visible_count 
        
        clean_query = sel_query.replace(" ", "").strip()
        
        # [A] 순서/숫자 표현 처리 (첫번째, 마지막, 1번...)
        rel_idx = -1
        if "첫번째" in clean_query or "첫번재" in clean_query: rel_idx = 0
        elif "두번째" in clean_query: rel_idx = 1
        elif "세번째" in clean_query: rel_idx = 2
        elif "네번째" in clean_query: rel_idx = 3
        elif "다섯번째" in clean_query: rel_idx = 4
        elif "마지막" in clean_query: rel_idx = visible_count - 1 # 현재 화면의 마지막
            
        num_match = re.search(r'(\d+)번', sel_query) or re.search(r'(\d+)', sel_query)
        
        if rel_idx != -1: # 상대적 순서 (첫번째, 마지막)
            final_idx = start_offset + rel_idx
            if 0 <= final_idx < len(candidates):
                item = candidates[final_idx]
                selected_name = get_safe_name(item) # ★ 수정된 함수 사용
        elif num_match: # 절대적 숫자 (1번, 6번)
            idx = int(num_match.group(1)) - 1
            if 0 <= idx < len(candidates):
                item = candidates[idx]
                selected_name = get_safe_name(item) # ★ 수정된 함수 사용
        
        # [B] 이름으로 선택 (핵심 수정 부분: 우선순위 검색)
        if not selected_name:
            clean_search_query = sel_query.replace(" ", "")
            
            # Step 1. ★ 현재 보고 있는 페이지(visible list)에서 먼저 검색 ★
            # (예: 6~10위를 보고 있다면 여기서 먼저 찾음)
            found_in_visible = False
            for i in range(start_offset, end_offset):
                if i >= len(candidates): break
                item = candidates[i]
                c_name = get_safe_name(item) # ★ 수정된 함수 사용
                if clean_search_query in str(c_name).replace(" ", ""):
                    selected_name = c_name
                    found_in_visible = True
                    break
            
            # Step 2. 현재 페이지에 없다면 -> 전체 리스트 검색
            # (예: 6~10위를 보고 있지만 "1위 장소 갈래"라고 한 경우)
            if not found_in_visible:
                for item in candidates:
                    c_name = item.get('real_name', item.get('name'))
                    if clean_search_query in str(c_name).replace(" ", ""):
                        selected_name = c_name
                        break
            
            # Step 3. 후보군 전체에도 없으면 사용자 입력 그대로 사용 (최후의 수단)
            if not selected_name: selected_name = sel_query

        if selected_name:
            state["prev_place"] = selected_name 
            saved_type = state.get("current_list_type", tourism_type)
            state["prev_type"] = saved_type
            state["current_candidates"] = [] 
            state["page_index"] = 0

            if "prev_type" in state:
                state["prev_type"] = state.get("current_list_type", tourism_type)
            
            # 1. 상세 정보 텍스트 가져오기 (RAG)
            rag_context = get_place_context_for_rag(selected_name)
            summary_text = ""
            if rag_context:
                # 2. LLM에게 포맷팅 요청
                sys_prompt_selection = f"""
                당신은 사용자의 즐거운 여행을 돕는 친절한 여행 가이드입니다.
                사용자가 선택한 장소 '{selected_name}'에 대한 정보를 [데이터]에서 찾아 아래 형식으로 요약해주세요.

                [1. 출력 포맷 가이드]
                📝 **개요**: [3~4문장으로 핵심 요약]
                📍 **주소**: [주소]
                📞 **문의/안내**: [전화번호]
                ⏰ **영업시간**: [시간]
                💡 **상세정보**: [입장료, 주차, 메뉴 등 데이터에 있는 가장 특징적인 정보 2~3개]

                [2. 엄격한 규칙]
                - 데이터에 해당 항목(예: 전화번호, 입장료)이 'Unknown', 'nan', '없음'이거나 비어있으면 그 줄은 아예 출력하지 마세요.
                - 예시: "📞 문의/안내: Unknown" -> 그 줄은 아예 출력하지 말고, 빈 줄도 남기지 마세요.
                - 다른 미사여구(인사말 등)는 붙이지 마세요.
                - 상세정보를 출력할 때 대분류, 중분류, 소분류 등 카테고리 정보는 빼주세요.

                [3. 데이터]
                {rag_context}
                """
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "system", "content": sys_prompt_selection}],
                        temperature=0.1 # 포맷 준수를 위해 낮게 설정
                    )
                    summary_text = response.choices[0].message.content

                    # 실수로 Unknown을 뱉었을 경우 강제 삭제
                    filtered_lines = []
                    for line in summary_text.split('\n'):
                        # "Unknown"이나 "nan"이 포함된 줄은 리스트에 담지 않음 (삭제)
                        if "Unknown" in line or "nan" in line:
                            continue
                        filtered_lines.append(line)
                    
                    summary_text = "\n".join(filtered_lines)

                except Exception as e:
                    summary_text = "📝 **개요**: 상세 정보를 불러오는 중 오류가 발생했습니다."
            else:
                summary_text = "(해당 장소의 상세 데이터를 찾을 수 없습니다.)"
            
            # 4. 최종 메시지 조합 (상세정보 + 사이드바 안내 멘트)
            bot_msg = (
                f"✅ **'{selected_name}'**(으)로 결정하셨군요!\n\n"
                f"{summary_text}\n\n"
                f"──────────────────────\n"
                f"이어서 다음 장소를 추천해드릴까요?\n\n"
                f"💡 **Tip**: 분위기나 목적을 바꾸고 싶다면, 왼쪽 **사이드바에서 옵션(동반인, 목적 등)을 변경**하고 다시 말씀해주셔도 됩니다!"
            )

            # 물결표를 ' - ' (하이픈)으로 바꿔서 취소선 문제를 없앱니다.
            bot_msg = bot_msg.replace("~", " - ")
            # ★ [수정] 플레이스홀더 삽입
            img_tag = image_manager.get_image_tag(selected_name, tourism_type)
            if img_tag:
                marker = "<<IMG_SELECTION>>"
                bot_msg += f"\n\n{marker}"
                img_placeholder_map[marker] = img_tag

        else:
            bot_msg = "⚠️ 장소를 찾지 못했습니다. 번호나 이름을 정확히 말씀해주세요."

    # [분기 2.5] Pass (더 보기 / 거절)
    elif intent == "pass":
        candidates = state.get("current_candidates", [])
        if not candidates:
            bot_msg = "😅 **현재 보여드릴 추천 목록이 초기화되었습니다.**"
        else:
            current_page = state.get("page_index", 0) + 1
            state["page_index"] = current_page
            start_idx = current_page * 5
            end_idx = start_idx + 5
            
            if start_idx >= len(candidates):
                bot_msg = "더 이상 추천할 장소가 없습니다. 😢 다른 조건으로 검색해보시겠어요?"
            else:
                target_items = candidates[start_idx:end_idx]
                bot_msg = f"🔄 **다른 장소들을 보여드릴게요!** ({start_idx+1}~{min(end_idx, len(candidates))}위)\n\n"
                if loc_query: bot_msg += f"📍 지역: **'{loc_query}'** 근처\n"
                bot_msg += "\n"
                
                rank = start_idx + 1
                for row in target_items:
                    # [수정] 데이터 타입(음식점/관광지)에 따른 키 처리 통합
                    # 1. 이름 가져오기 (관광지: real_name/name, 음식점: 명칭)
                    p_name = row.get('real_name', row.get('name', row.get('명칭')))
                    
                    # 2. 점수 가져오기 (관광지: score, 음식점: final_score)
                    score = row.get('score', row.get('final_score', 0.0))
                    score_txt = f"({score:.2f}⭐️)" if score > 0 else ""
                    
                    # 3. 주소 가져오기
                    addr = row.get('주소', '정보 없음')

                    # 4. 설명 구성 (음식점이면 메뉴 표시, 아니면 개요 표시)
                    menu = row.get('대표메뉴', '')
                    desc_raw = row.get('개요', '')
                    
                    if menu and menu != 'nan': # 음식점인 경우
                        # 메뉴 + 개요 요약
                        final_desc = f"🍜 대표메뉴: {menu}"
                        smart_desc = get_smart_description(desc_raw)
                        if smart_desc:
                             final_desc += f"\n   {smart_desc}"
                    else: # 일반 관광지인 경우
                        final_desc = get_smart_description(desc_raw)

                    # 메시지 조합
                    bot_msg += (
                        f"**{rank}. {p_name}** {score_txt}\n" 
                        f"   📍 주소: {addr}\n"
                        f"   {final_desc}\n"
                    )
                    
                    # 이미지 처리
                    img_tag = image_manager.get_image_tag(p_name)
                    if img_tag:
                        marker = f"<<IMG_{rank}>>"
                        bot_msg += f"{marker}\n\n"
                        img_placeholder_map[marker] = img_tag
                    else:
                        bot_msg += "\n\n"
                    rank += 1
                
                bot_msg += "\n💡 마음에 드는 장소의 **번호나 이름**을 말씀해 주세요."
                bot_msg = bot_msg.replace("~", " - ")

    # --- [분기 3] 추천 요청 (Recommendation) ---
    elif intent == "recommendation":
        if not tourism_type:
            history.append({"role": "assistant", "content": "🚨 **관광 타입**을 먼저 선택해주세요!"})
            return history, "", state

        # [공통] 위치 기준점 설정 (Logic A, B 모두 사용)
        target_location = None
        target_type_hint = state.get("prev_type")

        # 1. 상대적 위치 키워드 체크 (이 근처, 그 주변 등)
        relative_keywords = ["여기", "근처", "주변", "가까운", "이쪽", "요기", "거기", "그쪽"]
        is_actually_relative = False
        
        if loc_query:
            if any(word in loc_query for word in relative_keywords):
                is_actually_relative = True

        # 2. 타겟 위치 결정 로직 (우선순위 재정립)
        
        # Case A: "이 근처" 같은 표현이 있고 + 직전 장소가 있을 때 -> 직전 장소 기준
        if (is_relative or is_actually_relative) and current_prev:
            target_location = current_prev
            target_type_hint = state.get("prev_type")
            print(f"📍 [Context] 상대적 위치('{loc_query}') 감지 -> '{current_prev}' 좌표 사용")

        # Case B: 명시적인 지명(예: '강남역')이 있고 + 상대적 표현이 아닐 때 -> 입력된 지명 사용
        elif loc_query and not is_actually_relative:
            target_location = loc_query
            target_type_hint = None
            print(f"📍 [Input] 명시적 위치 감지 -> '{loc_query}' 사용")

        # Case C: 위치 언급은 없지만 직전 장소가 있을 때 (문맥 유지) -> 직전 장소 기준
        elif current_prev and not loc_query:
            target_location = current_prev
            target_type_hint = state.get("prev_type")
            print(f"📍 [Context] 위치 언급 없음 -> '{current_prev}' 유지")
            
        # Case D: 그 외 (처음 시작인데 "이 근처"라고 했거나, 위치 정보가 아예 없는 경우)
        else:
            target_location = loc_query # 이 경우 "이 근처"가 들어가서 외부 지도 검색 실패할 확률 높음 (정상 동작)
            target_type_hint = None

        # 좌표 찾기 (최종 결정된 target_location 사용)
        loc_coords = find_coordinates(target_location, place_type=target_type_hint)

        # 현재 생성할 추천 목록의 타입을 State에 임시 저장
        if "current_list_type" in state:
            state["current_list_type"] = tourism_type

        # [Logic B] 음식점 (NLU + Embedding + ★거리 반영★)
        if tourism_type == "음식점":
            if not food_category_input:
                history.append({"role": "assistant", "content": "🚨 **음식 소분류**를 선택해주세요!"})
                return history, "", state, gallery_html
            
            # 1. 검색 쿼리 결정
            search_query = key_query if key_query else food_category_input
            if not search_query: search_query = message 
            
            # 2. 임베딩 유사도 계산 (전체 음식점) -> similarity (1~5점)
            sim_scores = predictor.get_query_similarity("음식점", search_query)
            
            # 3. 데이터프레임 준비
            df_food = predictor.new_places_data["음식점"].copy()
            df_food['similarity'] = df_food['명칭'].map(sim_scores).fillna(0.0)
            
            # 4. ★ 거리 점수 계산 (Distance Score) 추가 ★
            df_food['dist_score'] = 0.0 # 기본값
            
            if loc_coords:
                r_lat, r_lon = loc_coords
                # 거리 계산 (km)
                dists_km = haversine_vectorized(r_lat, r_lon, df_food['위도'].values, df_food['경도'].values)
                
                # 거리 점수 변환 (1.5km 기준 감쇠)
                LAMBDA_DIST = 1.5 
                dist_weights = np.exp(-dists_km / LAMBDA_DIST)
                
                # 1~5점 스케일로 변환
                df_food['dist_score'] = 1.0 + (dist_weights * 4.0)
            else:
                # 위치 정보가 없으면 거리 점수는 중간값(3.0) 처리
                df_food['dist_score'] = 3.0 

            # 5. ★ 최종 점수 합산 (유사도 60% + 거리 40%) ★
            if loc_coords:
                df_food['final_score'] = (df_food['similarity'] * 0.6) + (df_food['dist_score'] * 0.4)
            else:
                df_food['final_score'] = df_food['similarity']

            # 6. 카테고리 필터링
            filtered_df = df_food[df_food['소분류'] == food_category_input]
            
            if filtered_df.empty:
                bot_msg = f"아쉽게도 '{food_category_input}' 카테고리에 해당하는 데이터가 없습니다."
                state["current_candidates"] = []
            else:
                # 7. 정렬 및 Top 30 선정 (final_score 기준)
                top30_food = filtered_df.sort_values(by='final_score', ascending=False).head(30)
                
                # State 저장
                state["current_candidates"] = top30_food.to_dict('records')
                state["page_index"] = 0
                
                target_items = top30_food.head(5)
                
                # 메시지 작성 (위치 정보 포함)
                location_msg = f"📍 **'{target_location}'** 근처 " if target_location and loc_coords else ""
                bot_msg = f"🍽️ {location_msg}**'{search_query}'** 느낌의 **{food_category_input}** 맛집 (Top 5)\n\n"
                
                rank = 1
                for _, r in target_items.iterrows():
                    p_name = r['명칭']
                    menu = r.get('대표메뉴', '-')
                    addr = r.get('주소', '-')
                    desc = r.get('개요', '')
                    final_desc = get_smart_description(desc)

                    # [수정] 평점(final_score) 가져오기
                    # 음식점 로직에서는 컬럼명이 'final_score'입니다.
                    score = r.get('final_score', 0.0)
                    score_txt = f"({score:.2f}⭐️)" if score > 0 else ""
                    
                    bot_msg += (
                        f"**{rank}. {p_name}** {score_txt}\n"
                        f"   🍜 대표메뉴: {menu}\n"
                        f"   📍 주소: {addr}\n"
                        f"   {final_desc}\n"
                    )
                    
                    # ★ [수정] 플레이스홀더 삽입
                    img_tag = image_manager.get_image_tag(p_name, tourism_type)
                    if img_tag:
                        marker = f"<<IMG_{rank}>>"
                        bot_msg += f"{marker}\n\n"
                        img_placeholder_map[marker] = img_tag
                    else:
                        bot_msg += "\n\n"

                    rank += 1
                
                bot_msg += "💡 마음에 드는 장소가 있다면 **'1번으로 갈래'** 또는 **장소명**을 말씀해주세요."
                # 물결표를 ' - ' (하이픈)으로 바꿔서 취소선 문제를 없앱니다.
                bot_msg = bot_msg.replace("~", " - ")

        # [Logic A] 관광지/쇼핑/문화
        else:
            if not companion or not purpose:
                history.append({"role": "assistant", "content": "🚨 **동반인**과 **목적**을 선택해주세요!"})
                return history, "", state
            
            loc_coords = find_coordinates(target_location)
            
            top30_df = get_combined_recommendations(
                main_df_A, predictor, user_query_raw=message, 
                user_companion=companion, user_purpose=purpose, 
                user_tourism_type=tourism_type, prev_place_name=current_prev,
                location_coords=loc_coords, keyword_query=key_query,
                is_famous_intent=is_famous
            )
            
            if top30_df.empty:
                bot_msg = "조건에 맞는 장소를 찾지 못했습니다. 다른 조건으로 검색해보세요."
                state["current_candidates"] = []
            else:
                current_candidates_list = top30_df.to_dict('records')
                state["current_candidates"] = current_candidates_list
                state["page_index"] = 0 
                target_items = current_candidates_list[0:5]
                
                bot_msg = f"🔎 **{tourism_type}** 추천 결과 (Top 5)\n"
                
                # 위치 정보 표시
                if is_relative and current_prev:
                    bot_msg += f"📍 **'{current_prev}'** 주변에서 찾았습니다.\n"
                elif target_location:
                    bot_msg += f"📍 지역: **'{target_location}'** 근처\n"
                bot_msg += "\n"
                
                rank = 1
                for row in target_items:
                    p_name = row.get('real_name', row.get('name'))
                    score = row.get('score', 0.0)
                    desc_raw = row.get('개요', '')
                    final_desc = get_smart_description(desc_raw) # [적용] 깔끔한 요약 함수
                    
                    bot_msg += (
                        f"**{rank}. {p_name}** ({score:.2f}⭐️)\n"
                        f"   📍 주소: {row.get('주소', '정보 없음')}\n"                         
                        f"   {final_desc}\n"
                    )
                    
                    # ★ [수정] 플레이스홀더 삽입
                    img_tag = image_manager.get_image_tag(p_name, tourism_type)
                    if img_tag:
                        marker = f"<<IMG_{rank}>>"
                        bot_msg += f"{marker}\n\n"
                        img_placeholder_map[marker] = img_tag
                    else:
                        bot_msg += "\n\n"

                    rank += 1
                bot_msg += "\n💡 마음에 드는 장소의 **번호나 이름**을 말씀해 주세요."

    # --- [분기 4] 단순 대화 (Chat) ---
    else:
        if client:
            try:
                sys_msg = "당신은 사용자의 즐거운 여행을 돕는 친절한 여행 가이드입니다." 
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": sys_msg}] + history[-3:],
                    temperature=0.1
                )
                bot_msg = response.choices[0].message.content
            except: bot_msg = "오류가 발생했습니다."
        else:
            bot_msg = "API 키가 없습니다."

    # [Step 2] Output Translation (출력 번역: 한국어 -> 외국어)
    # 모든 로직이 끝나고 여기로 모입니다.
    
    # 현재 함수가 인식하고 있는 언어 설정과 봇 메시지 확인
    print(f"\n🛑 [DEBUG START] ---------------------------------------")
    print(f"🛑 [DEBUG] 선택된 언어(language_select): '{language_select}'")
    print(f"🛑 [DEBUG] 봇 메시지 길이: {len(bot_msg)} 자")
    print(f"🛑 [DEBUG] 봇 메시지 앞부분: {bot_msg[:30]}...")

    final_response = bot_msg 

    # 1. 텍스트 번역 (플레이스홀더 <<IMG_X>> 는 텍스트로 취급되어 번역기 통과)
    if bot_msg and language_select != "한국어":
        print(f"🌍 [Output Trans] Translating text with markers...")
        try:
            # Tip: 시스템 프롬프트에 '<<...>>' 형태는 번역하지 말라고 하면 더 정확함.
            # 하지만 GPT는 보통 이런 코드는 건드리지 않습니다.
            final_response = translator.translate_to_user_lang(bot_msg, language_select)
        except Exception as e:
            print(f"🚨 [Trans Error] {e}")
            final_response = bot_msg

    # 2. 번역 후, 플레이스홀더를 실제 이미지 태그로 '치환' (Restore Images)
    if img_placeholder_map:
        print(f"🖼️ [Image Restore] Restoring {len(img_placeholder_map)} images...")
        for marker, img_tag in img_placeholder_map.items():
            # 만약 번역기가 공백을 넣어서 << IMG_1 >> 처럼 변형했을 경우를 대비해 
            # replace를 조금 더 유연하게 하거나, 그냥 marker를 찾아서 바꿈
            if marker in final_response:
                final_response = final_response.replace(marker, img_tag)
            else:
                # 혹시 번역기가 마커를 건드려서(예: Image 1) 못 찾는 경우 대비 (보완책)
                # 이 경우는 드물지만, 발생하면 맨 뒤에라도 붙여줍니다.
                print(f"⚠️ Marker {marker} lost in translation. Appending at end.")
                final_response += f"\n{img_tag}"

    # [로그 저장 추가] 함수 맨 마지막 return 직전에 추가
    session_id = state.get("session_id", "unknown")
    
    # 봇의 마지막 응답 가져오기
    last_bot_msg = history[-1]['content'] if history else ""

    save_log(
        session_id=session_id,
        event_type="CHAT",
        lang=language_select,
        t_type=tourism_type,
        comp=companion,
        purp=purpose,
        food=food_category_input,
        u_input=message, # 사용자가 입력한 메시지
        b_response=last_bot_msg # 봇이 대답한 메시지
    )
    history.append({"role": "assistant", "content": final_response})
    
    return history, "", state, gallery_html
        

# --- 4. UI 구성 ---
# 번역 데이터 로드 
try:
    with open("data/translations.json", "r", encoding="utf-8") as f:
        UI_TEXT = json.load(f)
    print("✅ [App] 번역 데이터 로드 완료")
except:
    print("🚨 번역 파일이 없습니다. make_translator.py를 먼저 실행하세요.")
    exit()

# 언어 선택 드롭다운용 매핑 (화면 표시 이름 : JSON 키)
LANG_MAP = {
    "English": "en",
    "Korean": "ko",
    "Japanese": "ja",
    "Chinese": "zh",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Russian": "ru",
    "Vietnamese": "vi",
    "Thai": "th"
}

with gr.Blocks(title="AI K-Travel Bot") as demo:
    
    header = gr.Markdown("# ✈️ AI K-Travel Bot") # 초기값 영어

    md_intro1 = gr.Markdown("1. First select your **Language**, then select **travel options like Tourism Type, Companion, and Purpose** from the left sidebar.")
    md_intro2 = gr.Markdown("2. Click **'Start Travel Planning'** button.")
    md_intro3 = gr.Markdown("3. Chat with AI to get personalized recommendations!")

    gr.HTML("<br>")

    chat_state = gr.State(value={
        "session_id": str(uuid.uuid4()),  # ★ 고유 세션 ID 생성
        "state": "IDLE", 
        "prev_place": None, 
        "prev_type": None,
        "current_candidates": [],
        "page_index": 0,
        "user_language": "en"
    })
    en_text = UI_TEXT.get("en", {})

    with gr.Row():
        # === [1열] 좌측 사이드바 (언어 선택 + 여행 옵션) ===
        with gr.Column(scale=1, min_width=250): 
            
            # 1열: 여행 옵션
            md_opt_title = gr.Markdown("## 🌐 Travel Options")

            dd_language = gr.Dropdown(
                choices=list(LANG_MAP.keys()),
                value="English",
                label="Language", 
                interactive=True
            )

            gr.Markdown("---")

            dd_tourism = gr.Dropdown(
                choices=en_text.get("choices_type", ["Sightseeing", "Shopping", "Culture", "Food"]), 
                label="Tourism Type"
            )
            dd_food_category = gr.Dropdown(
                choices=en_text.get("choices_food", ["Korean", "Western", "Japanese", "Chinese", "Cafe", "Unique"]), 
                label="Food Category",
                visible=False,  # <--- 처음에는 숨김
                interactive=True
            )
            
            gr.Markdown("---")
            
            dd_companion = gr.Dropdown(
                choices=en_text.get("choices_companion", ["Alone", "Partner", "Family", "Friend", "Group", "Other"]), 
                label="Companion"
            )
            dd_purpose = gr.Dropdown(
                choices=en_text.get("choices_purpose", ["Tour", "Experience", "History", "Nature", "Rest", "Shopping", "Culture", "Filming Site", "Other"]), 
                label="Purpose"
            )
            
            gr.Markdown("<br>")
            btn_start = gr.Button("🚀 Start Travel Planning", variant="primary", size="lg")

        # === [2열] 우측 챗봇 영역 ===
        with gr.Column(scale=4): 
            md_chat_title = gr.Markdown("## 💬 AI Chat")
            chatbot = gr.Chatbot(
                elem_id="chat-window", 
                type="messages", 
                bubble_full_width=False, 
                height=800)
            
            gallery_output = gr.HTML(label="Gallery", visible=True)

            with gr.Row():
                txt_input = gr.Textbox(
                    show_label=False, 
                    placeholder="Ex: Recommend a quiet cafe near Gangnam Station",
                    scale=8,
                    container=False
                )
                btn_submit = gr.Button("Send", scale=1, variant="secondary")
    
    def update_food_visibility(selected_type):
        # 다국어 지원을 위해 'Food', 'Restaurant', '음식점' 등의 키워드가 포함되면 표시
        # (번역 JSON에 있는 단어들과 매칭)
        target_keywords = ["음식점", "Restaurant", "レストラン", "餐厅", "Restaurantes", "Restaurants", "Рестораны", "Nhà Hàng", "ร้านอาหาร"] 
        
        if selected_type and any(k in selected_type for k in target_keywords):
            return gr.update(visible=True)
        else:
            # 숨길 때는 값을 None으로 초기화해야 로직 꼬임을 방지함
            return gr.update(visible=False, value=None)

    # --- [이벤트 함수 1] Tourism Type 변경 시 Food Category 보이기/숨기기 ---
    dd_tourism.change(
        fn=update_food_visibility,
        inputs=[dd_tourism],
        outputs=[dd_food_category]
    )

    # --- [이벤트 함수 2] 언어 변경 ---
    def change_ui_language(selection):
        code = LANG_MAP.get(selection, "en")
        t = UI_TEXT.get(code, UI_TEXT["en"])

        new_types = t.get('choices_type', [])
        new_foods = t.get('choices_food', [])
        new_companions = t.get('choices_companion', [])
        new_purposes = t.get('choices_purpose', [])

        return (
            code, # lang_state (히든 상태값) 업데이트
            gr.update(value=f"# {t['title']}"),
            gr.update(value=t['intro_1']),
            gr.update(value=t['intro_2']),
            gr.update(value=t['intro_3']),
            gr.update(value=f"### {t['opt_title']}"),
            gr.update(label=t['label_type'], choices=new_types, value=new_types[0] if new_types else None),
            gr.update(label=t['label_food_cat'], choices=new_foods, value=new_foods[0] if new_foods else None),
            gr.update(label=t['label_companion'], choices=new_companions, value=new_companions[0] if new_companions else None),
            gr.update(label=t['label_purpose'], choices=new_purposes, value=new_purposes[0] if new_purposes else None),
            gr.update(value=t['btn_start']),
            gr.update(value=f"### {t['chat_title']}"),
            gr.update(placeholder=t['placeholder']),
            gr.update(value=t['btn_submit']),
            gr.update(label=t['label_gallery'])
        )
    
    lang_state = gr.State("en") 

    dd_language.change(
        fn=change_ui_language,
        inputs=[dd_language],
        outputs=[
            lang_state, header,
            md_intro1, md_intro2, md_intro3,
            md_opt_title, dd_tourism, dd_food_category,
            dd_companion, dd_purpose, btn_start, md_chat_title,
            txt_input, btn_submit, gallery_output
            ]
    )

    # --- 시작 버튼 클릭 핸들러 ---
    def handle_start_click(history, state, tourism_type, companion, purpose, food_cat, lang_code): 
        # 1. 언어 설정에 맞는 텍스트 가져오기
        # (lang_code가 'ko'면 한국어, 'en'이면 영어 딕셔너리를 가져옴)
        t = UI_TEXT.get(lang_code, UI_TEXT["ko"])

        # 2. Validation (JSON에 있는 에러 메시지 사용)
        if not tourism_type:
            history.append({"role": "assistant", "content": t['err_type']})
            return history, state, "", gr.update(interactive=True)
        
        # 3. 로직 처리를 위한 한국어 데이터 매핑 
        # (화면에는 영어가 보여도, 내부 로직은 한국어로 돌아가야 하므로 변환)
        ko_t = UI_TEXT["ko"]
        curr_t = t
        
        # [헬퍼 함수] 현재 언어의 값을 한국어 원본 값으로 바꿔주는 함수
        # 예: "Restaurant" -> "음식점", "Alone" -> "혼자"
        def to_korean(val, key):
            if val in curr_t[key]:
                idx = curr_t[key].index(val)
                return ko_t[key][idx]
            return val

        k_type = to_korean(tourism_type, 'choices_type')
        k_comp = to_korean(companion, 'choices_companion')
        k_purp = to_korean(purpose, 'choices_purpose')
        
        # State 생성 (내부는 한국어 데이터로 저장)
        new_state = {"state": "AWAITING_QUERY", "companion": k_comp, "purpose": k_purp, "type": k_type, "prev_place": None}

        # 4. 환영 메시지 생성 (JSON 템플릿 사용 -> 실시간 번역 필요 없음!)
        bot_msg = ""
        
        if k_type == "음식점":
            # 템플릿: "🍽️ **'{cat}'** 전문 식당을 추천해 드릴게요!..."
            cat_display = food_cat if food_cat else "Food" 
            bot_msg = t['welcome_food'].format(cat=cat_display)
        else:
            # 템플릿: "👋 **'{comp}'**와 함께하는 **'{purp}'** 여행을 시작합니다!..."
            comp_display = companion if companion else ""
            purp_display = purpose if purpose else ""
            bot_msg = t['welcome_tour'].format(comp=comp_display, purp=purp_display)

        # 메시지 추가
        history.append({"role": "assistant", "content": bot_msg})

        # [로그 저장 추가]
        session_id = state.get("session_id", "unknown")
        save_log(
            session_id=session_id,
            event_type="START_CLICK",
            lang=lang_code,
            t_type=tourism_type,
            comp=companion,
            purp=purpose,
            food=food_cat,
            u_input="START_BUTTON_CLICKED",
            b_response=bot_msg
        )
        
        # 버튼 텍스트 변경 및 비활성화
        return history, new_state, "", gr.update(interactive=False, value=t['btn_running'])

    # --- [이벤트 연결 수정] ---
    # ★ 중요: inputs에 'dd_language'가 아니라 'lang_state'를 넣어야 합니다.
    
    btn_start.click(
        fn=handle_start_click,
        inputs=[chatbot, chat_state, dd_tourism, dd_companion, dd_purpose, dd_food_category, lang_state], 
        outputs=[chatbot, chat_state, gallery_output, btn_start]
    )

    txt_input.submit(
        fn=handle_chat_submit,
        inputs=[txt_input, chatbot, chat_state, dd_tourism
                , dd_companion, dd_purpose, dd_food_category, lang_state], 
        outputs=[chatbot, txt_input, chat_state, gallery_output]
    )
    
    btn_submit.click(
        fn=handle_chat_submit,
        inputs=[txt_input, chatbot, chat_state, dd_tourism, dd_companion, dd_purpose, dd_food_category, lang_state],
        outputs=[chatbot, txt_input, chat_state, gallery_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)