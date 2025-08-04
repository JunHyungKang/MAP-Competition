# 📚 MAP-Competition 문서

이 폴더는 MAP-Competition 프로젝트의 모든 문서를 체계적으로 관리합니다. 각 문서는 독립적으로 완전한 정보를 제공하면서도, 자연스러운 진행 흐름을 통해 프로젝트를 단계별로 완성할 수 있도록 설계되었습니다.

## 📁 문서 구조

```
docs/
├── README.md                    # 이 파일 - 문서 인덱스
├── setup-guide.md              # 환경 설정 가이드
├── eda-guide.md                # 데이터 분석 가이드
├── model-development-guide.md   # 모델 개발 가이드
└── langchain-langgraph-guide.md # 도구 사용 가이드
```

## 🎯 각 문서의 목적

### 🛠️ [환경 설정 가이드](setup-guide.md)
프로젝트 개발 환경을 설정하는 방법을 설명합니다.
- **Python 가상환경 설정**: 프로젝트 전용 가상환경 구성
- **의존성 패키지 설치**: 필요한 라이브러리 설치 가이드
- **개발 환경 구성**: IDE 설정 및 개발 도구 구성
- **문제 해결 가이드**: 일반적인 설정 문제 해결 방법

### 📊 [데이터 분석 가이드](eda-guide.md)
데이터 분석과 시각화를 위한 도구 사용법을 설명합니다.
- **EDA (Exploratory Data Analysis)**: 탐색적 데이터 분석 가이드
- **데이터 시각화**: Plotly Dash 및 Streamlit 대시보드 사용법
- **분석 결과 해석**: 데이터 분석 결과의 의미와 인사이트
- **고급 분석 기법**: 심화된 데이터 분석 방법론

### 🤖 [모델 개발 가이드](model-development-guide.md)
수학 오해 예측 모델을 개발하는 방법을 설명합니다.
- **모델 아키텍처**: 다양한 모델 구조 및 설계 방법
- **훈련 및 평가**: 모델 훈련 방법과 성능 평가 기법
- **성능 최적화**: 모델 성능 향상을 위한 기법들
- **제출 파일 생성**: 대회 제출을 위한 파일 생성 방법

### 🛠️ [도구 사용 가이드](langchain-langgraph-guide.md)
프로젝트에서 사용하는 도구들의 사용법을 설명합니다.
- **LangChain & LangGraph**: LLM 워크플로우 구축 도구
- **기타 도구들**: 프로젝트에서 활용하는 다양한 도구들
- **도구별 설정**: 각 도구의 설정 및 구성 방법
- **활용 사례**: MAP 대회에서의 구체적 활용 방안

## 🚀 빠른 시작

### 🎯 프로젝트 개요
MAP-Competition은 학생들의 수학 오해(misconception)를 예측하는 대회입니다. 이 프로젝트는 3단계 예측 모델을 개발하여 학생들의 답변에서 오해를 식별합니다.

### 📋 단계별 진행 가이드

#### 1단계: 환경 설정 (10-15분)
```bash
# 프로젝트 클론 및 환경 설정
git clone <repository-url>
cd MAP-Competition
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
**다음 단계**: [환경 설정 가이드](setup-guide.md) 참조

#### 2단계: 데이터 분석 (30분-2시간)
```bash
# EDA 대시보드 실행
cd eda
python run_dash.py      # Plotly Dash (추천)
# 또는
python run_streamlit.py # Streamlit
```
**다음 단계**: [데이터 분석 가이드](eda-guide.md) 참조

#### 3단계: 모델 개발 (2-8시간)
```bash
# 기본 모델 훈련
python train_basic_model.py

# 고급 모델 훈련
python train_advanced_model.py
```
**다음 단계**: [모델 개발 가이드](model-development-guide.md) 참조

#### 4단계: 고급 도구 활용 (2-4시간)
```bash
# LangChain 설치 및 설정
pip install langchain langchain-community langgraph

# 고급 워크플로우 실행
python advanced_workflow.py
```
**다음 단계**: [도구 사용 가이드](langchain-langgraph-guide.md) 참조

## 🔗 문서 간 네비게이션

### 📋 권장 진행 순서
1. **[환경 설정 가이드](setup-guide.md)** → 2. **[데이터 분석 가이드](eda-guide.md)** → 3. **[모델 개발 가이드](model-development-guide.md)** → 4. **[도구 사용 가이드](langchain-langgraph-guide.md)**

### 🎯 각 단계별 목표

#### 📋 1단계: 환경 설정
- **목표**: 개발 환경 구축 및 기본 도구 설치
- **완료 기준**: EDA 대시보드가 정상 실행됨
- **다음 단계**: [데이터 분석 가이드](eda-guide.md)로 이동

#### 📊 2단계: 데이터 분석
- **목표**: 데이터 구조 파악 및 패턴 발견
- **완료 기준**: 주요 인사이트 도출 및 모델 설계 방향 결정
- **다음 단계**: [모델 개발 가이드](model-development-guide.md)로 이동

#### 🤖 3단계: 모델 개발
- **목표**: 3단계 예측 모델 구축 및 성능 최적화
- **완료 기준**: MAP@3 성능 지표 달성 및 제출 파일 생성
- **다음 단계**: [도구 사용 가이드](langchain-langgraph-guide.md)로 이동

#### 🛠️ 4단계: 고급 도구 활용
- **목표**: LangChain과 LangGraph를 활용한 고급 워크플로우 구축
- **완료 기준**: 복잡한 추론 과정의 자동화 및 성능 향상
- **최종 목표**: 대회 우승을 위한 최적 모델 완성

## 📝 문서 작성 가이드

새로운 문서를 추가할 때는 다음 사항을 고려해주세요:

1. **명확한 제목**: 문서의 내용을 명확히 표현하는 제목 사용
2. **구조화된 내용**: 목차와 섹션을 활용하여 읽기 쉽게 구성
3. **상호 참조**: 관련 문서들과의 링크 설정
4. **최신성 유지**: 정기적으로 내용 업데이트
5. **독립성**: 각 문서가 독립적으로 완전한 정보를 제공
6. **개요 섹션**: 문서 시작 부분에 목적, 범위, 대상 독자 명시
7. **빠른 시작**: 핵심 단계 요약과 예상 소요 시간 제공

## 🤝 기여

문서 개선이나 새로운 문서 추가에 대한 제안이 있으시면 이슈를 생성해주세요.

## 📚 관련 리소스

### 🔗 프로젝트 내부 리소스
- [프로젝트 메인 README](../README.md) - 전체 프로젝트 개요
- [EDA 대시보드](../eda/) - 인터랙티브 데이터 분석 도구
- [참고 노트북](../reference_notebooks/) - 다양한 접근법 예제

### 🎯 대회 관련 리소스
- [MAP 대회 페이지](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings) - 대회 정보 및 데이터
- [대회 토론](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/discussion) - 참가자 토론 및 아이디어

### 📖 학습 리소스
- [수학 교육 연구](https://www.nctm.org/) - 수학 교육 방법론
- [자연어 처리](https://huggingface.co/) - 최신 NLP 모델 및 도구
- [머신러닝](https://scikit-learn.org/) - 머신러닝 라이브러리

## ✅ 완료 체크리스트

프로젝트 진행 상황을 확인하세요:

- [ ] **환경 설정**: Python 가상환경 및 필요한 패키지 설치
- [ ] **데이터 분석**: EDA 대시보드를 통한 데이터 탐색 완료
- [ ] **모델 개발**: 기본 모델 구축 및 성능 평가
- [ ] **고급 도구**: LangChain과 LangGraph 활용
- [ ] **최종 제출**: 최적화된 모델로 제출 파일 생성

모든 항목이 체크되면 MAP 대회 준비가 완료된 것입니다! 🎉 