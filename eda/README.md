# 📊 EDA (Exploratory Data Analysis) 사용 가이드

> **문서 정리 및 개선 이력**
> - 중복 문서 제거 및 폴더 내 문서 간소화(2024)
> - 상세 정보는 본 README와 docs/eda-guide.md에 집중, 폴더 개요는 간결하게 유지
> - 문서 구조와 링크 검증, 접근성·유지보수성·사용자 경험 대폭 개선

이 가이드는 MAP-Competition 프로젝트의 EDA 대시보드와 분석 도구들을 사용하는 방법을 설명합니다.

## 🚀 빠른 시작

### 핵심 명령어
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

## 📁 파일 구조

```
eda/
├── README.md                    # 이 파일 - EDA 가이드
├── data_analysis.py            # 기본 데이터 분석 스크립트
├── eda_dashboard.py            # Plotly Dash 대시보드
├── streamlit_eda.py            # Streamlit 대시보드
├── run_dash.py                 # Dash 실행 스크립트
├── run_streamlit.py            # Streamlit 실행 스크립트
├── documentation_cleanup_summary.md  # 문서 정리 작업 기록
├── duplicate_analysis.md       # 중복 분석 결과
└── data_analysis_plots.png    # 분석 결과 이미지
```

## 🎯 EDA 목표

### 주요 목표
- 📈 **데이터 이해**: 수학 오해 데이터의 구조와 특성 파악
- 🔍 **패턴 발견**: Category와 Misconception 간의 관계 분석
- 📊 **시각화**: 인터랙티브 차트를 통한 직관적 데이터 탐색
- 🎯 **인사이트 도출**: 모델 개발에 필요한 특징 발견

## 🚀 대시보드 실행

### 1. Plotly Dash 대시보드 (추천)

```bash
cd eda
python run_dash.py
```

- **URL**: `http://localhost:8050`
- **특징**: 최고의 인터랙티브성, 실시간 필터링

### 2. Streamlit 대시보드

```bash
cd eda
python run_streamlit.py
```

- **URL**: `http://localhost:8501`
- **특징**: 간단한 사용법, 데이터 다운로드 기능

### 3. 기본 데이터 분석

```bash
cd eda
python data_analysis.py
```

- **출력**: 콘솔에 기본 통계 정보 출력
- **파일**: `data_analysis_plots.png` 생성

## 📈 대시보드 기능

### 주요 기능
- **실시간 필터링**: Category, Misconception, QuestionId 필터
- **통계 카드**: 총 샘플 수, 고유 문제 수, 평균 답변 길이, Misconception 비율
- **시각화 차트**: 파이 차트, 바 차트, 히스토그램, 히트맵
- **인터랙티브 기능**: 줌/팬, 호버 정보, 선택 기능 (Dash)
- **데이터 다운로드**: CSV 파일 저장 (Streamlit)

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

## 🔧 문제 해결

### 일반적인 문제들

#### 1. Dash 실행 오류
```bash
pip install dash --upgrade
python eda_dashboard.py --port 8051
```

#### 2. Streamlit 실행 오류
```bash
pip install streamlit
streamlit run streamlit_eda.py --server.port 8502
```

#### 3. 데이터 경로 오류
```bash
ls ../data/
```

## 🔍 추가 분석 아이디어

- **텍스트 마이닝**: 학생 설명에서 자주 나오는 단어 분석
- **클러스터링**: 유사한 오해 패턴 그룹화
- **예측 모델**: 새로운 답변의 Category/Misconception 예측

## 📚 다음 단계

데이터 분석이 완료되면 다음 문서들을 참조하여 모델 개발을 진행하세요:

1. **[모델 개발 가이드](../docs/model-development-guide.md)**: EDA 결과를 바탕으로 수학 오해 예측 모델 개발
2. **[도구 사용 가이드](../docs/README.md)**: LangChain과 LangGraph를 활용한 고급 모델 개발

## 📋 관련 문서 및 최종 구조

- **[문서 정리 작업 기록](documentation_cleanup_summary.md)**: 폴더 내 문서 정리 및 개선 이력
- **[중복 분석 결과](duplicate_analysis.md)**: 문서 간 중복 내용의 상세 분석
- [메인 프로젝트 문서](../README.md)
- [환경 설정 가이드](../docs/README.md)
- [모델 개발 가이드](../docs/model-development-guide.md)

> **최종 문서 구조**
> - docs/eda-guide.md: 완전한 EDA 가이드(상세)
> - eda/README.md: 폴더 개요 및 빠른 실행 안내(간소화)
> - 기타 문서: 참고용 기록 및 분석 결과

## 📞 지원

EDA 사용 중 문제가 발생하면 이슈를 생성해주세요. 