# My-Seoul-Mate: 외국인 관광객 대상 서울 관광지 추천 시스템

 My-Seoul-Mate는 서울을 방문하는 외국인 관광객을 위해 취향과 대화의 맥락을 반영하여 최적의 관광지를 추천하는 LLM 기반 하이브리드 추천 챗봇 서비스입니다.

 기존의 단순 검색이나 나열식 정보 제공을 넘어, 계층적 그룹 협업 필터링과 NLU 기반 동적 가중치 알고리즘을 결합하여 개인화된 추천을 제공합니다.
 또한, 프롬프트 엔지니어링 및 RAG 기술을 적용하여 보다 정확하고 신뢰도 있는 여행 정보를 제공합니다. 


## WorkFlow 

1. **데이터 수집 및 전처리:**
    - TripAdvisor(리뷰 7,500건) 및 한국관광공사(관광지 2,200곳) 데이터 수집.
    - 다국어 리뷰의 국문 표준화(Pivot Language: English) 및 LLM 기반 감성 분석($R_{adj}$ 보정).
2. **사용자 클러스터링 및 지식 베이스 구축:**
    - BERTopic 및 LLM을 활용한 리뷰 속성(동반인, 목적) 분류 및 사용자 그룹 정의.
    - 관광지 메타데이터 및 개요의 벡터화(SBERT) 및 RAG용 지식 베이스 구축.
3. **하이브리드 모델링:** 
    - 비식별 데이터의 한계를 극복하기 위한 3단계 계층적 그룹 협업 필터링(Hierarchical Group-based CF) 구축.
    - LightGBM 회귀 모델을 활용하여 다중 속성 기반의 평점 예측 및 장기 선호도 산출.
4. **NLU 기반 맥락 인식 알고리즘 구현:** 
    - 대화 시나리오(초기 탐색/연속 대화/조건 변경 등)에 따라 4가지 컴포넌트(장기 선호, 분위기, 거리, 쿼리)의 가중치를 동적으로 조절하는 로직 구현.
5. **서비스 배포:**
    - Gradio 기반 대화형 UI 구현 및 Hugging Face Spaces 배포.
    - RAG 기반 할루시네이션 제어 및 실시간 번역 파이프라인 최적화.


## Key Contents

1. **하이브리드 추천 엔진 (Static):**
    - **Hierarchical Group-based CF:** 신규 사용자(Cold-Start)를 유사한 성향의 그룹(예: 가족-관광형)으로 매핑하여 초기 데이터 희소성 해결.
    - **LightGBM Regression:** 리뷰 텍스트 임베딩, 동반인, 목적 등 다양한 피처를 학습하여 정교한 예상 평점($S_{long}$) 예측 (MAE: 0.3385).
    - **Bayesian Smoothing:** 계층 간 데이터 불균형을 해소하기 위해 상위 레이어의 통계 정보를 활용한 평점 보정.
2. **맥락 인식 동적 가중치 시스템 (Dynamic):**
    - **State Management:** 전체 대화 로그 대신 현재 상태(State)와 직전 쿼리만 추적하여 토큰 비용 절감 및 응답 속도 최적화.
    - **4-Component Scoring:**
        - **$S_{long}$ (장기 선호):** 사용자 프로필 기반 정적 점수.
        - **$S_{sim}$ (분위기 연속성):** 직전 추천 장소와의 임베딩 유사도를 계산하여 여행 흐름 유지.
        - **$S_{dist}$ (거리 효율성):** Haversine 공식 및 지수 감쇠 함수를 적용하여 1.5km 반경 내 최적 동선 제안.
        - **$S_{query}$ (명시적 취향):** 발화 내 키워드와 장소 간의 유사도 반영.
    - **Scenario-based Weighting:** 초기 탐색, 위치 변경, 연속 추천 등 상황에 따라 $\gamma$ 가중치를 동적으로 변경.
3. **검색 증강 생성 (RAG):**
    - LLM이 없는 정보를 생성하지 않도록, 검증된 DB(영업시간, 주소 등) 내의 정보만을 검색하여 답변 생성.
    - **Multilingual Support:** 사용자 입력(다국어) → 한국어 로직 처리 → 사용자 언어 출력의 실시간 파이프라인 구축.


## Tech Stack
- **Language:** Python
- **LLM & NLP:** OpenAI GPT-3.5, SBERT (Multilingual), BERTopic
- **Machine Learning:** LightGBM, Scikit-learn, Pandas, NumPy
- **Service:** Gradio, OpenAI API, TourAPI, Hugging Face Spaces
- **Tools:** Google Colab, Visual Studio Code, Google Drive (Data Management), Notion, Git


## Live Demo

https://huggingface.co/spaces/Wonder-Buddies/My-Seoul-Mate

허깅페이스(Hugging Face)에서 실제 동작하는 서비스를 체험해보실 수 있습니다. 


## Directory 
```
My-Seoul-Mate
├── DATA
│   ├── embeddings          # SBERT 기반 관광지/리뷰 임베딩 벡터 (.npy)
│   ├── images(_git)        # UI 렌더링용 장소 이미지 리소스
│   ├── Scoring_Final.xlsx  # 전처리 완료된 평점 및 메타데이터
│   ├── 관광지_Final.xlsx     # 관광타입별 장소 아이템 데이터
│   ├── 문화시설_Final.xlsx   
│   └── 음식점_Final.xlsx    
├── MODELS
│   └── RegressionRating_LGBM_Best.joblib  # 학습된 LightGBM 회귀 모델
├── SERVICE                      # 데모 서비스 코드 
│   ├── app.py                   # Gradio 메인 애플리케이션 
│   ├── embedding_logic.py       # 임베딩 로드 및 유사도/거리 계산 로직
│   └── recommendation_logic.py  # 하이브리드 추천 엔진 및 필터링 로직
└── requirements.txt             # 환경 설정
```
