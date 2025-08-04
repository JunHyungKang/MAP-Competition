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

## 📚 상세 가이드

완전한 EDA 사용 가이드는 [docs/analysis/eda-guide.md](../../docs/analysis/eda-guide.md)를 참고하세요.

## 🔗 관련 문서

- [메인 프로젝트 문서](../../README.md)
- [환경 설정 가이드](../../docs/setup/)
- [모델 개발 가이드](../../docs/development/)
- [도구 사용 가이드](../../docs/tools/)

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

## 📞 지원

EDA 사용 중 문제가 발생하면 이슈를 생성해주세요. 