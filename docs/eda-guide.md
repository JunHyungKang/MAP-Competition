# 📊 EDA (Exploratory Data Analysis) 사용 가이드

이 가이드는 MAP-Competition 프로젝트의 EDA 대시보드와 분석 도구들을 사용하는 방법을 설명합니다.

## 📋 문서 개요

### 목적 및 범위
이 문서는 MAP-Competition 프로젝트의 데이터를 탐색적 데이터 분석(EDA)하는 방법을 안내합니다. Plotly Dash와 Streamlit을 활용한 인터랙티브 대시보드 사용법과 데이터 분석 기법을 포함합니다.

### 주요 내용
- EDA 목표 및 데이터 개요
- Plotly Dash와 Streamlit 대시보드 사용법
- 인터랙티브 시각화 및 필터링 기능
- 데이터 분석 결과 해석 방법
- 모델 개발을 위한 인사이트 도출

### 대상 독자
- 데이터 분석을 시작하는 개발자
- 인터랙티브 대시보드에 관심이 있는 사용자
- 수학 오해 데이터 분석에 관심이 있는 연구자

## 🚀 빠른 시작

### 핵심 단계 요약
1. **환경 설정 확인** (1분) - [환경 설정 가이드](setup-guide.md) 참조
2. **대시보드 실행** (2분) - Plotly Dash 또는 Streamlit 선택
3. **데이터 탐색** (10-30분) - 인터랙티브 시각화 활용
4. **인사이트 도출** (시간 가변) - 분석 결과 해석

### 주요 명령어
```bash
cd eda
python run_dash.py      # Plotly Dash (추천)
# 또는
python run_streamlit.py # Streamlit
```

### 예상 소요 시간
- **대시보드 실행**: 2-3분
- **기본 데이터 탐색**: 10-30분
- **심화 분석**: 1-2시간

## 🎯 EDA 목표

## 🎯 EDA 목표

### 주요 목표
- 📈 **데이터 이해**: 수학 오해 데이터의 구조와 특성 파악
- 🔍 **패턴 발견**: Category와 Misconception 간의 관계 분석
- 📊 **시각화**: 인터랙티브 차트를 통한 직관적 데이터 탐색
- 🎯 **인사이트 도출**: 모델 개발에 필요한 특징 발견

## 📊 데이터 개요

이 프로젝트는 학생들의 수학 오해(misconception)를 분석하는 데이터셋을 탐색적 데이터 분석(EDA)할 수 있는 대시보드를 제공합니다.

### 데이터 구조
- **Train 데이터**: 36,696개 샘플, 7개 컬럼
- **Test 데이터**: 3개 샘플
- **15개의 고유 문제**에 대해 평균 2,446개의 답변

### 주요 컬럼
- `QuestionId`: 문제 ID
- `QuestionText`: 문제 텍스트
- `MC_Answer`: 정답
- `StudentExplanation`: 학생 설명
- `Category`: 분류 (True_Correct, False_Misconception 등)
- `Misconception`: 오해 유형 (Incomplete, Additive, Duplication 등)

## 🚀 빠른 시작

### 1. Plotly Dash 대시보드 (추천)

```bash
# EDA 폴더로 이동
cd eda

# Dash 대시보드 실행
python run_dash.py
```

- **URL**: `http://localhost:8050`
- **특징**: 최고의 인터랙티브성, 실시간 필터링
- **장점**: 전문적인 대시보드 느낌, 모든 차트가 연동

### 2. Streamlit 대시보드

```bash
# EDA 폴더로 이동
cd eda

# Streamlit 대시보드 실행
python run_streamlit.py
```

- **URL**: `http://localhost:8501`
- **특징**: 간단한 사용법, 데이터 다운로드 기능
- **장점**: 빠른 프로토타이핑, 상세 분석 섹션

### 3. 기본 데이터 분석

```bash
# EDA 폴더로 이동
cd eda

# 기본 분석 실행
python data_analysis.py
```

- **출력**: 콘솔에 기본 통계 정보 출력
- **파일**: `data_analysis_plots.png` 생성

## 📈 대시보드 기능

### 공통 기능

#### 1. 실시간 필터링
- **Category 필터**: True_Correct, False_Misconception 등
- **Misconception 필터**: Incomplete, Additive, Duplication 등
- **QuestionId 필터**: 특정 문제별 분석

#### 2. 통계 카드
- **총 샘플 수**: 필터링된 데이터의 총 개수
- **고유 문제 수**: 선택된 조건의 고유 문제 수
- **평균 답변 길이**: 학생 설명의 평균 길이
- **Misconception 비율**: 오해가 포함된 답변의 비율

#### 3. 시각화 차트
- **파이 차트**: Category 분포
- **바 차트**: Top 10 Misconception 유형
- **히스토그램**: 텍스트 길이 분포
- **히트맵**: Category vs Misconception 관계

### Plotly Dash 특별 기능

#### 1. 인터랙티브 차트
- **줌/팬**: 차트 확대/축소 및 이동
- **호버 정보**: 마우스 오버 시 상세 정보 표시
- **선택 기능**: 차트 요소 클릭으로 필터링

#### 2. 실시간 업데이트
- **동시 업데이트**: 필터 변경 시 모든 차트 동시 갱신
- **즉시 반영**: 변경사항이 즉시 화면에 반영

#### 3. 전문적인 UI
- **깔끔한 레이아웃**: 비즈니스 대시보드 수준의 디자인
- **반응형 디자인**: 다양한 화면 크기에 대응

### Streamlit 특별 기능

#### 1. 데이터 다운로드
- **CSV 다운로드**: 필터링된 데이터를 CSV 파일로 저장
- **동적 파일명**: 필터 조건이 파일명에 반영

#### 2. 상세 분석
- **가장 긴 답변 Top 10**: 확장 가능한 섹션으로 상세 확인
- **가장 짧은 답변 Top 10**: 개별 샘플 분석 가능

## 📊 주요 발견사항

### Category 분포
- **True_Correct**: 40% (14,802개) - 정답 선택 + 올바른 설명
- **False_Misconception**: 26% (9,457개) - 오답 선택 + 오개념 설명
- **False_Neither**: 18% (6,542개) - 오답 선택 + 중립적 설명
- **True_Neither**: 14% (5,265개) - 정답 선택 + 중립적 설명

### 주요 Misconception 유형
1. **Incomplete**: 1,454개 - 불완전한 설명
2. **Additive**: 929개 - 덧셈 관련 오개념
3. **Duplication**: 704개 - 중복 관련 오개념
4. **Subtraction**: 620개 - 뺄셈 관련 오개념
5. **Positive**: 566개 - 양수 관련 오개념

### 텍스트 길이 분석
- **문제 텍스트**: 평균 97자, 표준편차 66자
- **학생 설명**: 평균 70자, 표준편차 39자

## 🎯 EDA UI 비교

| 기능 | Plotly Dash | Streamlit |
|------|-------------|-----------|
| **인터랙티브성** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **성능** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **개발 속도** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **커스터마이징** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **학습 곡선** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **EDA 적합성** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

## 💡 사용 팁

### Plotly Dash 사용 시
1. **필터 조합**: 여러 필터를 조합하여 특정 패턴 발견
2. **차트 상호작용**: 차트를 클릭하여 상세 정보 확인
3. **실시간 탐색**: 필터 변경 시 즉시 결과 확인

### Streamlit 사용 시
1. **데이터 다운로드**: 관심 있는 필터링 결과를 CSV로 저장
2. **상세 분석**: 확장 가능한 섹션에서 개별 샘플 확인
3. **빠른 프로토타이핑**: 새로운 분석 아이디어를 빠르게 구현

## 🔧 문제 해결

### 일반적인 문제들

#### 1. Dash 실행 오류
```bash
# 최신 Dash 버전 설치
pip install dash --upgrade

# 다른 포트 사용
python eda_dashboard.py --port 8051
```

#### 2. Streamlit 실행 오류
```bash
# Streamlit 설치 확인
pip install streamlit

# 다른 포트 사용
streamlit run streamlit_eda.py --server.port 8502
```

#### 3. 데이터 경로 오류
```bash
# 데이터 파일 확인
ls ../data/

# 경로 수정 (필요시)
# eda/ 폴더의 파일들에서 '../data/' 경로 확인
```

### 성능 문제 해결
1. **데이터 캐싱**: Streamlit의 `@st.cache_data` 활용
2. **필터 최적화**: 필요한 컬럼만 로드
3. **차트 최적화**: 대용량 데이터는 샘플링 후 시각화

## 🔍 추가 분석 아이디어

1. **텍스트 마이닝**: 학생 설명에서 자주 나오는 단어 분석
2. **시계열 분석**: 문제별 답변 패턴 변화
3. **클러스터링**: 유사한 오해 패턴 그룹화
4. **예측 모델**: 새로운 답변의 Category/Misconception 예측

## 🔧 기술 스택

### Plotly Dash 버전
- **Dash**: 웹 프레임워크
- **Plotly**: 인터랙티브 차트
- **Pandas**: 데이터 처리
- **NumPy**: 수치 계산

### Streamlit 버전
- **Streamlit**: 웹 앱 프레임워크
- **Plotly**: 인터랙티브 차트
- **Pandas**: 데이터 처리
- **Seaborn**: 통계 시각화

## 📁 파일 구조

```
MAP-Competition/
├── eda/                        # EDA 대시보드 및 분석 도구
│   ├── README.md              # EDA 폴더 설명서
│   ├── EDA_README.md          # 상세한 EDA 가이드
│   ├── data_analysis.py       # 기본 데이터 분석 스크립트
│   ├── eda_dashboard.py       # Plotly Dash 대시보드
│   ├── streamlit_eda.py       # Streamlit 대시보드
│   ├── run_dash.py            # Dash 실행 스크립트
│   └── run_streamlit.py       # Streamlit 실행 스크립트
├── data/                       # 데이터 파일
│   ├── train.csv              # 훈련 데이터
│   ├── test.csv               # 테스트 데이터
│   └── sample_submission.csv  # 샘플 제출 파일
└── requirements.txt            # 필요한 패키지 목록
```

## 📚 다음 단계

데이터 분석이 완료되면 다음 문서들을 참조하여 모델 개발을 진행하세요:

1. **[모델 개발 가이드](model-development-guide.md)**: EDA 결과를 바탕으로 수학 오해 예측 모델 개발
2. **[도구 사용 가이드](langchain-langgraph-guide.md)**: LangChain과 LangGraph를 활용한 고급 모델 개발

### 권장 진행 순서
1. **데이터 탐색**: 대시보드를 활용한 데이터 패턴 발견
2. **특성 엔지니어링**: 발견한 패턴을 바탕으로 특성 생성
3. **모델 개발**: [모델 개발 가이드](model-development-guide.md)를 참조하여 모델 설계
4. **고급 도구 활용**: [도구 사용 가이드](langchain-langgraph-guide.md)를 참조하여 LLM 워크플로우 구축

## 📞 지원

EDA 사용 중 문제가 발생하면 이슈를 생성해주세요. 