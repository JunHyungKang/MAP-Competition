# MAP - Charting Student Math Misunderstandings

## 📚 대회 개요

이 대회는 학생들의 수학 오개념을 예측하는 NLP 모델을 개발하는 것입니다. 학생들의 개방형 응답을 바탕으로 잠재적인 수학 오개념을 정확히 예측하여, 교사들이 학생들의 잘못된 사고를 식별하고 해결하는 데 도움을 주는 솔루션을 만드는 것이 목표입니다.

### 🎯 주요 목표
- ✅ 학생들의 수학 추론 설명을 분석하여 잠재적 오개념 식별
- ✅ 다양한 문제에 걸쳐 일반화되는 오개념 예측 모델 개발
- ✅ 교사들이 학생들의 잘못된 사고를 더 쉽게 파악할 수 있도록 지원

## 📊 데이터 설명

### 데이터셋 개요
Eedi 플랫폼에서 학생들은 진단 질문(Diagnostic Questions, DQs)에 답합니다. 이는 하나의 정답과 세 개의 오답(방해 요소)을 포함한 객관식 문제입니다. 객관식 답을 선택한 후, 학생들은 때때로 선택한 답을 정당화하는 서면 설명을 제공하도록 요청받았습니다. 이러한 설명이 MAP 데이터셋의 주요 초점이며, 학생들의 추론에서 잠재적 오개념을 식별하고 해결하는 데 사용됩니다.

### 데이터셋 특징
- **총 샘플 수**: 36,696개 (훈련 데이터)
- **고유 문제 수**: 15개
- **평균 답변 수**: 문제당 2,446개
- **주요 컬럼**: QuestionId, QuestionText, MC_Answer, StudentExplanation, Category, Misconception

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

### 모델 목표
대회의 목표는 다음 3단계를 수행하는 모델을 개발하는 것입니다:

1. **선택한 답이 정확한지 판단** (Category에서 True 또는 False; 예: True_Correct)
2. **설명에 오개념이 포함되어 있는지 평가** (Category에서 Correct, Misconception, 또는 Neither; 예: True_Correct)
3. **있다면 존재하는 특정 오개념 식별**

### 데이터 파일

#### train.csv / test.csv
- **QuestionId**: 고유 질문 식별자
- **QuestionText**: 질문의 텍스트
- **MC_Answer**: 학생이 선택한 객관식 답
- **StudentExplanation**: 특정 객관식 답을 선택한 이유에 대한 학생의 설명
- **Category**: [훈련 데이터만] 학생의 객관식 답과 설명 간의 관계 분류 (예: True_Misconception은 정확한 객관식 답 선택과 오개념을 드러내는 설명을 나타냄)
- **Misconception**: [훈련 데이터만] 학생의 설명에서 식별된 수학 오개념. Category에 오개념이 포함된 경우에만 적용되며, 그렇지 않으면 'NA'

#### sample_submission.csv
올바른 형식의 제출 파일 예시:
- **row_id**: 행 식별자
- **Category:Misconception**: 콜론(:)으로 연결된 예측 분류 Category와 Misconception. 최대 3개의 예측을 공백으로 구분하여 만들 수 있습니다.

### 데이터 특징
- 진단 질문은 Eedi 플랫폼에서 이미지 형식으로 제시됨
- 모든 질문 내용(수학 표현식 포함)은 정확성을 위해 인간 참여 OCR 프로세스를 통해 추출됨
- 테스트 데이터는 약 16,000개의 행을 포함
- 파일 크기: 7.94 MB
- 라이선스: MIT

## 📊 평가 지표

**Mean Average Precision @ 3 (MAP@3)**을 사용하여 평가됩니다:

```
MAP@3 = (1/N) * Σ(AP@3)
```

여기서:
- N: 관찰 수
- AP@3: 각 관찰에 대한 평균 정밀도 (상위 3개 예측 기준)
- 각 관찰당 하나의 정확한 라벨만 존재

## 📁 프로젝트 구조

```
MAP-Competition/
├── README.md                    # 이 파일 - 프로젝트 메인 설명서
├── docs/                        # 📚 문서 폴더
│   ├── README.md               # 문서 구조 개요
│   ├── setup/                  # 🛠️ 환경 설정 가이드
│   ├── analysis/               # 📊 데이터 분석 가이드
│   ├── development/            # 🤖 모델 개발 가이드
│   └── tools/                  # 🛠️ 도구 사용 가이드
├── data/                        # 데이터 파일
│   ├── train.csv              # 훈련 데이터
│   ├── test.csv               # 테스트 데이터
│   └── sample_submission.csv  # 제출 샘플
├── eda/                        # EDA 대시보드 및 분석 도구
│   ├── README.md              # EDA 폴더 설명서
│   ├── EDA_README.md          # 상세한 EDA 가이드
│   ├── data_analysis.py       # 기본 데이터 분석 스크립트
│   ├── eda_dashboard.py       # Plotly Dash 대시보드
│   ├── streamlit_eda.py       # Streamlit 대시보드
│   ├── run_dash.py            # Dash 실행 스크립트
│   └── run_streamlit.py       # Streamlit 실행 스크립트
├── reference_notebooks/         # 참고 노트북
├── requirements.txt             # 의존성 패키지
└── .gitignore
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd MAP-Competition

# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터 다운로드

캐글에서 다음 데이터를 다운로드하여 `data/` 폴더에 저장하세요:
- `train.csv` → `data/train/`
- `test.csv` → `data/test/`
- `sample_submission.csv` → `data/`

### 3. EDA 대시보드 실행

#### Plotly Dash 대시보드 (추천)
```bash
cd eda
python run_dash.py
```
- 브라우저에서 `http://localhost:8050` 접속

#### Streamlit 대시보드
```bash
cd eda
python run_streamlit.py
```
- 브라우저에서 `http://localhost:8501` 접속

#### 기본 데이터 분석
```bash
cd eda
python data_analysis.py
```

### 4. 노트북 실행 순서

1. **데이터 탐색**: `reference_notebooks/` 폴더의 노트북들
2. **특성 엔지니어링**: EDA 대시보드를 활용한 데이터 분석
3. **모델 개발**: 참고 노트북 기반 모델 개발
4. **제출 파일 생성**: 최종 예측 및 제출

## 📚 문서 가이드

### 📋 환경 설정
- [환경 설정 가이드](docs/setup/) - Python 가상환경, 의존성 설치, 개발 환경 구성

### 📊 데이터 분석
- [EDA 가이드](docs/analysis/) - 탐색적 데이터 분석, 대시보드 사용법, 분석 결과 해석

### 🤖 모델 개발
- [모델 개발 가이드](docs/development/) - 모델 아키텍처, 훈련 방법, 성능 최적화

### 🛠️ 도구 사용
- [도구 사용 가이드](docs/tools/) - LangChain & LangGraph 활용, 기타 도구 사용법

## 📝 제출 형식

제출 파일은 다음 형식을 따라야 합니다:

```csv
row_id,Category:Misconception
36696,True_Correct:NA False_Neither:NA False_Misconception:Incomplete
36697,True_Correct:NA False_Neither:NA False_Misconception:Incomplete
...
```

### 제출 요구사항:
- 각 `row_id`에 대해 최대 3개의 `Category:Misconception` 예측
- 예측은 공백으로 구분
- 파일명: `submission.csv`
- 헤더 포함

## 🏆 대회 일정

- **시작일**: 2025년 7월 10일
- **참가 마감**: 2025년 10월 8일 (UTC 23:59)
- **팀 병합 마감**: 2025년 10월 8일 (UTC 23:59)
- **최종 제출 마감**: 2025년 10월 15일 (UTC 23:59)

## 💰 상금

- **1위**: $20,000
- **2위**: $12,000
- **3위**: $8,000
- **4-6위**: 각 $5,000

## 🔧 코드 요구사항

- **CPU 노트북**: 최대 9시간 실행 시간
- **GPU 노트북**: 최대 9시간 실행 시간
- **인터넷 접근**: 비활성화
- **외부 데이터**: 자유롭게 사용 가능 (사전 훈련된 모델 포함)
- **제출 파일명**: `submission.csv`

## 🔧 기술 스택

### 데이터 분석
- **Pandas**: 데이터 처리 및 분석
- **NumPy**: 수치 계산
- **Matplotlib/Seaborn**: 정적 시각화
- **Plotly**: 인터랙티브 시각화

### 웹 대시보드
- **Dash**: 인터랙티브 웹 대시보드
- **Streamlit**: 빠른 프로토타이핑
- **HTML/CSS**: UI 커스터마이징

### 머신러닝 (예정)
- **Scikit-learn**: 전통적인 ML 모델
- **Transformers**: BERT, RoBERTa 등
- **PyTorch/TensorFlow**: 딥러닝 모델

## 📚 참고 자료

- [대회 링크](https://kaggle.com/competitions/map-charting-student-math-misunderstandings)
- [코드 대회 FAQ](https://www.kaggle.com/docs/competitions#tutorial-competition-code-competition)
- [오개념 프레임워크 보고서](링크 추가 예정)

## 🤝 기여

1. 이 저장소를 포크하세요
2. 새로운 브랜치를 생성하세요 (`git checkout -b feature/amazing-feature`)
3. 변경사항을 커밋하세요 (`git commit -m 'Add amazing feature'`)
4. 브랜치에 푸시하세요 (`git push origin feature/amazing-feature`)
5. Pull Request를 생성하세요

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 👥 팀원

- [팀원 이름 추가]

## 📞 문의

대회 관련 질문이나 이슈가 있으시면 이슈를 생성해 주세요.

---

**참고**: 이 대회는 Vanderbilt University와 The Learning Agency가 주최하며, Gates Foundation과 Walton Family Foundation의 지원을 받고 있습니다. 