# 📊 EDA (Exploratory Data Analysis) 폴더

수학 오해 데이터를 탐색적 데이터 분석할 수 있는 대시보드와 도구들이 포함되어 있습니다.

## 📁 파일 구조

```
eda/
├── README.md                    # 이 파일
├── EDA_README.md               # 상세한 EDA 가이드
├── data_analysis.py            # 기본 데이터 분석 스크립트
├── eda_dashboard.py            # Plotly Dash 대시보드
├── streamlit_eda.py            # Streamlit 대시보드
├── run_dash.py                 # Dash 실행 스크립트
└── run_streamlit.py            # Streamlit 실행 스크립트
```

## 🚀 빠른 시작

### 1. Plotly Dash 대시보드 (추천)
```bash
cd eda
python run_dash.py
```
- 브라우저에서 `http://localhost:8050` 접속
- **특징**: 최고의 인터랙티브성, 실시간 필터링
- **장점**: 전문적인 대시보드 느낌, 모든 차트가 연동

### 2. Streamlit 대시보드
```bash
cd eda
python run_streamlit.py
```
- 브라우저에서 `http://localhost:8501` 접속
- **특징**: 간단한 사용법, 데이터 다운로드 기능
- **장점**: 빠른 프로토타이핑, 상세 분석 섹션

### 3. 기본 데이터 분석
```bash
cd eda
python data_analysis.py
```
- **출력**: 콘솔에 기본 통계 정보 출력
- **파일**: `data_analysis_plots.png` 생성

## 🔧 문제 해결

### Dash 실행 오류
```bash
# 최신 Dash 버전 설치
pip install dash --upgrade

# 다른 포트 사용
python eda_dashboard.py --port 8051
```

### Streamlit 실행 오류
```bash
# Streamlit 설치 확인
pip install streamlit

# 다른 포트 사용
streamlit run streamlit_eda.py --server.port 8502
```

### 데이터 경로 오류
```bash
# 데이터 파일 확인
ls ../data/

# 경로 수정 (필요시)
# eda/ 폴더의 파일들에서 '../data/' 경로 확인
```

## 📊 대시보드 기능

### 공통 기능
- ✅ **실시간 필터링**: Category, Misconception, QuestionId별 필터링
- ✅ **통계 카드**: 총 샘플 수, 고유 문제 수, 평균 답변 길이, Misconception 비율
- ✅ **시각화**: 파이 차트, 바 차트, 히스토그램, 히트맵

### Plotly Dash 특별 기능
- ✅ **인터랙티브 차트**: 줌, 팬, 호버 정보
- ✅ **실시간 업데이트**: 필터 변경 시 모든 차트 동시 업데이트
- ✅ **전문적인 UI**: 깔끔한 레이아웃과 색상

### Streamlit 특별 기능
- ✅ **데이터 다운로드**: 필터링된 데이터 CSV 다운로드
- ✅ **상세 분석**: 가장 긴/짧은 답변 Top 10
- ✅ **확장 가능한 섹션**: 접을 수 있는 상세 정보

## 🔧 기술 스택

### Plotly Dash
- **Dash**: 웹 프레임워크
- **Plotly**: 인터랙티브 차트
- **Pandas**: 데이터 처리
- **NumPy**: 수치 계산

### Streamlit
- **Streamlit**: 웹 앱 프레임워크
- **Plotly**: 인터랙티브 차트
- **Pandas**: 데이터 처리
- **Seaborn**: 통계 시각화

## 📈 주요 데이터 발견사항

### Category 분포
- **True_Correct**: 40% (14,802개)
- **False_Misconception**: 26% (9,457개)
- **False_Neither**: 18% (6,542개)
- **True_Neither**: 14% (5,265개)

### 주요 Misconception 유형
1. **Incomplete**: 1,454개
2. **Additive**: 929개
3. **Duplication**: 704개
4. **Subtraction**: 620개
5. **Positive**: 566개

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

## 🔍 추가 분석 아이디어

1. **텍스트 마이닝**: 학생 설명에서 자주 나오는 단어 분석
2. **시계열 분석**: 문제별 답변 패턴 변화
3. **클러스터링**: 유사한 오해 패턴 그룹화
4. **예측 모델**: 새로운 답변의 Category/Misconception 예측

## 📞 문제 해결

### 대시보드가 실행되지 않는 경우
1. **포트 충돌**: 다른 포트 사용
2. **패키지 설치**: `pip install dash streamlit plotly pandas`
3. **데이터 경로**: `../data/` 폴더에 CSV 파일이 있는지 확인

### 성능 문제
1. **데이터 캐싱**: Streamlit의 `@st.cache_data` 활용
2. **필터 최적화**: 필요한 컬럼만 로드
3. **차트 최적화**: 대용량 데이터는 샘플링 후 시각화

## 📚 상세 가이드

더 자세한 정보는 `EDA_README.md` 파일을 참조하세요. 