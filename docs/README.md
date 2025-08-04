# 📚 MAP-Competition 문서

이 폴더는 MAP-Competition 프로젝트의 모든 문서를 체계적으로 관리합니다. 각 문서는 독립적으로 완전한 정보를 제공하면서도, 자연스러운 진행 흐름을 통해 프로젝트를 단계별로 완성할 수 있도록 설계되었습니다.

## 📁 문서 구조

```
docs/
├── README.md                    # 이 파일 - 완전한 프로젝트 가이드
└── model-development-guide.md   # 모델 개발 가이드
```

## 🎯 각 문서의 목적

### 📊 [EDA 가이드](../eda/README.md)
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

#### 2단계: 데이터 분석 (30분-2시간)
```bash
# EDA 대시보드 실행
cd eda
python run_dash.py      # Plotly Dash (추천)
# 또는
python run_streamlit.py # Streamlit
```

#### 3단계: 모델 개발 (2-8시간)
```bash
# 기본 모델 훈련
python train_basic_model.py

# 고급 모델 훈련
python train_advanced_model.py
```

#### 4단계: 고급 도구 활용 (2-4시간)
```bash
# LangChain 설치 및 설정
pip install langchain langchain-community langgraph

# 고급 워크플로우 실행
python advanced_workflow.py
```

## 🛠️ 환경 설정 가이드

### 📋 사전 요구사항

#### 시스템 요구사항
- **Python**: 3.8 이상
- **RAM**: 최소 8GB (16GB 권장)
- **저장공간**: 최소 10GB
- **OS**: Windows, macOS, Linux

#### 권장 사양
- **CPU**: 4코어 이상
- **RAM**: 16GB 이상
- **GPU**: NVIDIA GPU (선택사항, 딥러닝 모델용)
- **저장공간**: SSD 50GB 이상

### 🚀 단계별 설정

#### 1. 저장소 클론
```bash
# GitHub에서 저장소 클론
git clone <repository-url>
cd MAP-Competition

# 또는 새로 시작하는 경우
mkdir MAP-Competition
cd MAP-Competition
```

#### 2. Python 가상환경 생성

#### macOS/Linux
```bash
# Python 가상환경 생성
python3 -m venv .venv

# 가상환경 활성화
source .venv/bin/activate

# 가상환경이 활성화되었는지 확인
which python
# 출력: /path/to/MAP-Competition/.venv/bin/python
```

#### Windows
```bash
# Python 가상환경 생성
python -m venv .venv

# 가상환경 활성화
.venv\Scripts\activate

# 가상환경이 활성화되었는지 확인
where python
# 출력: C:\path\to\MAP-Competition\.venv\Scripts\python.exe
```

#### 3. 의존성 패키지 설치
```bash
# 기본 패키지 설치
pip install -r requirements.txt

# 또는 개별 설치
pip install pandas numpy matplotlib seaborn plotly
pip install dash dash-bootstrap-components
pip install streamlit
pip install scikit-learn
```

#### 4. 데이터 준비
```bash
# data 폴더 생성
mkdir -p data

# 데이터 파일을 data/ 폴더에 복사
# train.csv, test.csv, sample_submission.csv
```

#### 5. 환경 확인
```bash
# Python 버전 확인
python --version

# 설치된 패키지 확인
pip list

# 데이터 파일 확인
ls data/
```

## 📦 설치된 주요 패키지들

### 데이터 처리
- `pandas>=1.5.0` - 데이터 분석
- `numpy>=1.21.0` - 수치 계산

### 머신러닝
- `scikit-learn>=1.1.0` - 전통적 ML
- `xgboost>=1.6.0` - 그래디언트 부스팅
- `lightgbm>=3.3.0` - LightGBM

### 딥러닝 & NLP
- `torch>=2.0.0` - PyTorch
- `transformers>=4.30.0` - Hugging Face Transformers
- `tokenizers>=0.13.0` - 토크나이저
- `datasets>=2.0.0` - 데이터셋 관리
- `accelerate>=0.20.0` - 분산 훈련

### 오픈소스 LLM 관련
- `sentencepiece>=0.1.99` - 토크나이저
- `protobuf>=3.20.0` - 직렬화

### 텍스트 처리
- `nltk>=3.7` - 자연어 처리
- `textblob>=0.17.0` - 텍스트 분석

### 시각화
- `matplotlib>=3.5.0` - 기본 시각화
- `seaborn>=0.11.0` - 통계 시각화
- `plotly>=5.10.0` - 인터랙티브 시각화

### 개발 도구
- `jupyter>=1.0.0` - Jupyter 노트북
- `black>=22.0.0` - 코드 포맷터
- `flake8>=4.0.0` - 코드 린터

### 캐글 관련
- `kaggle>=1.5.0` - 캐글 API

## 🔧 개발 환경 설정

### 1. Jupyter 노트북 커널 등록
```bash
# 가상환경 활성화 후
python -m ipykernel install --user --name=.venv --display-name="MAP Environment"
```

### 2. 코드 포맷팅 설정
```bash
# Black 설정
black --line-length=88 .

# Flake8 설정
flake8 --max-line-length=88 .
```

### 3. VS Code 설정
```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./.venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true
}
```

### 4. Git 설정
```bash
# Git 초기 설정
git init
git add .
git commit -m "Initial commit"

# .gitignore 확인
cat .gitignore
```

## 🎯 오픈소스 LLM 사용 준비

가상환경이 활성화된 상태에서 다음 모듈들을 사용할 수 있습니다:

```python
# 추출된 모듈들 사용
from reference_notebooks.extracted_modules_for_opensource_llm import *

# 데이터 처리
processor = MAPDataProcessor()
train_data = processor.load_and_preprocess_data("train.csv")

# 프롬프트 엔지니어링
prompt_engineer = PromptEngineer()
prompts = prompt_engineer.create_prompts(train_data, template='detailed')

# 토크나이징
tokenizer_manager = TokenizationManager("microsoft/phi-2", max_length=512)

# 모델 훈련
trainer = ModelTrainer("microsoft/phi-2", num_classes=processor.n_classes)
```

## 🔧 문제 해결

### 일반적인 문제들

#### 1. 가상환경 활성화 실패
```bash
# macOS/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

# 만약 위 명령어가 작동하지 않으면
python -m venv .venv --clear
```

#### 2. 패키지 설치 오류
```bash
# pip 업그레이드
pip install --upgrade pip

# 캐시 클리어
pip cache purge

# 개별 패키지 설치
pip install pandas
pip install numpy
```

#### 3. Dash 실행 오류
```bash
# Dash 최신 버전 설치
pip install dash --upgrade

# 포트 충돌 시 다른 포트 사용
python eda/eda_dashboard.py --port 8051
```

#### 4. Streamlit 실행 오류
```bash
# Streamlit 설치 확인
pip install streamlit

# 포트 충돌 시 다른 포트 사용
streamlit run eda/streamlit_eda.py --server.port 8502
```

### GPU 설정 (선택사항)

#### CUDA 설치 (NVIDIA GPU)
```bash
# CUDA Toolkit 설치
# https://developer.nvidia.com/cuda-downloads

# PyTorch GPU 버전 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# TensorFlow GPU 버전 설치
pip install tensorflow[gpu]
```

#### Apple Silicon (M1/M2) 설정
```bash
# PyTorch Apple Silicon 버전
pip install torch torchvision torchaudio

# TensorFlow Apple Silicon 버전
pip install tensorflow-macos tensorflow-metal
```

## 📊 환경 테스트

### 1. 기본 테스트
```bash
# Python 가져오기 테스트
python -c "import pandas as pd; import numpy as np; print('✅ 기본 패키지 OK')"

# 데이터 로드 테스트
python -c "import pandas as pd; df = pd.read_csv('data/train.csv'); print(f'✅ 데이터 로드 OK: {len(df)} 행')"
```

### 2. EDA 대시보드 테스트
```bash
# Dash 대시보드 테스트
cd eda
python run_dash.py

# 새 터미널에서 Streamlit 테스트
cd eda
python run_streamlit.py
```

### 3. 성능 테스트
```bash
# 메모리 사용량 확인
python -c "import psutil; print(f'메모리 사용량: {psutil.virtual_memory().percent}%')"

# CPU 사용량 확인
python -c "import psutil; print(f'CPU 사용량: {psutil.cpu_percent()}%')"
```

## 💡 사용 팁

### 1. 패키지 업데이트
```bash
pip install --upgrade pip
pip install --upgrade -r requirements.txt
```

### 2. 새로운 패키지 설치
```bash
pip install package_name
pip freeze > requirements.txt  # requirements.txt 업데이트
```

### 3. 가상환경 재생성 (필요시)
```bash
# 기존 가상환경 삭제
rm -rf .venv

# 새 가상환경 생성
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 🚨 주의사항

### 1. 가상환경 활성화 필수
- 프로젝트 작업 전 반드시 가상환경을 활성화하세요
- 터미널에서 `(.venv)` 표시가 보여야 합니다

### 2. 패키지 버전 관리
- 새로운 패키지 설치 시 `requirements.txt`를 업데이트하세요
- 팀원과 공유할 때는 동일한 환경을 유지하세요

### 3. 메모리 관리
- 오픈소스 LLM은 메모리를 많이 사용합니다
- 큰 모델 사용 시 배치 크기를 조절하세요

## 📝 환경 변수 설정

### 환경 변수 파일 생성
```bash
# .env 파일 생성
touch .env

# .env 파일 내용
echo "PYTHONPATH=."
echo "CUDA_VISIBLE_DEVICES=0"
echo "OMP_NUM_THREADS=4"
```

### 환경 변수 로드
```bash
# .env 파일 로드
source .env

# 또는 Python에서
import os
from dotenv import load_dotenv
load_dotenv()
```

## 🔍 환경 진단

### 시스템 정보 확인
```bash
# Python 버전
python --version

# 패키지 버전
pip freeze

# 시스템 정보
python -c "import platform; print(platform.platform())"

# 메모리 정보
python -c "import psutil; print(f'총 메모리: {psutil.virtual_memory().total / 1024**3:.1f} GB')"
```

### 성능 벤치마크
```bash
# 데이터 로딩 성능
python -c "import time; import pandas as pd; start=time.time(); df=pd.read_csv('data/train.csv'); print(f'로딩 시간: {time.time()-start:.2f}초')"

# 메모리 사용량
python -c "import pandas as pd; import sys; df=pd.read_csv('data/train.csv'); print(f'메모리 사용량: {sys.getsizeof(df) / 1024**2:.1f} MB')"
```

## 🤖 모델 개발 가이드

### 🎯 모델 목표

#### 3단계 예측 모델
1. **답변 정확성 판단**: True/False 예측
2. **오개념 포함 여부**: Correct/Misconception/Neither 예측
3. **구체적 오개념 식별**: 35개 오개념 유형 중 하나 예측

#### 평가 지표
- **MAP@3**: Mean Average Precision @ 3
- 각 샘플당 최대 3개 예측 가능
- 예측 형식: `Category:Misconception`

### 📊 데이터 이해

#### 입력 데이터
- **QuestionText**: 수학 문제 텍스트
- **MC_Answer**: 학생이 선택한 객관식 답
- **StudentExplanation**: 학생의 설명 텍스트

#### 출력 데이터
- **Category**: 6개 클래스
  - True_Correct, True_Misconception, True_Neither
  - False_Correct, False_Misconception, False_Neither
- **Misconception**: 35개 오개념 유형 + NA

#### 데이터 특징
- **총 샘플**: 36,696개 (훈련)
- **고유 문제**: 15개
- **텍스트 길이**: 평균 70자 (학생 설명)
- **불균형**: Category와 Misconception 분포 불균형

### 🏗️ 모델 아키텍처

#### 1. 기본 접근법

##### 텍스트 분류 모델
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 모델 초기화
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=len(category_labels)
)

# 입력 텍스트 구성
def prepare_input(question, answer, explanation):
    return f"Question: {question} Answer: {answer} Explanation: {explanation}"
```

##### 멀티태스크 학습
```python
class MultiTaskModel(nn.Module):
    def __init__(self, base_model, num_categories, num_misconceptions):
        super().__init__()
        self.base_model = base_model
        self.category_classifier = nn.Linear(768, num_categories)
        self.misconception_classifier = nn.Linear(768, num_misconceptions)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids, attention_mask)
        pooled_output = outputs.pooler_output
        
        category_logits = self.category_classifier(pooled_output)
        misconception_logits = self.misconception_classifier(pooled_output)
        
        return category_logits, misconception_logits
```

#### 2. 고급 접근법

##### 앙상블 모델
```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# 여러 모델 조합
models = [
    ('lr', LogisticRegression()),
    ('svm', SVC(probability=True)),
    ('rf', RandomForestClassifier())
]

ensemble = VotingClassifier(models, voting='soft')
```

##### 딥러닝 앙상블
```python
class EnsembleModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
    
    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)
```

### 📈 성능 최적화

#### 1. 하이퍼파라미터 튜닝
```python
import optuna

def objective(trial):
    # 하이퍼파라미터 정의
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    
    # 모델 훈련
    model = train_model(lr, batch_size)
    score = evaluate_model(model)
    
    return score

# 최적화 실행
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

#### 2. 교차 검증
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # 모델 훈련 및 평가
    model = train_model(X_train, y_train)
    score = evaluate_model(model, X_val, y_val)
    print(f'Fold {fold}: {score}')
```

#### 3. 데이터 증강
```python
def augment_text(text):
    # 동의어 치환
    synonyms = {'big': 'large', 'small': 'tiny'}
    for word, synonym in synonyms.items():
        text = text.replace(word, synonym)
    return text

# 증강된 데이터 생성
augmented_data = []
for text in original_texts:
    augmented_text = augment_text(text)
    augmented_data.append(augmented_text)
```

### 🎯 제출 파일 생성

#### 1. 예측 후처리
```python
def postprocess_predictions(predictions):
    # Top 3 예측 선택
    top3_indices = np.argsort(predictions, axis=1)[:, -3:]
    
    # Category:Misconception 형식으로 변환
    formatted_predictions = []
    for indices in top3_indices:
        formatted = []
        for idx in reversed(indices):
            category = category_labels[idx // num_misconceptions]
            misconception = misconception_labels[idx % num_misconceptions]
            formatted.append(f"{category}:{misconception}")
        formatted_predictions.append(" ".join(formatted))
    
    return formatted_predictions
```

#### 2. CSV 파일 생성
```python
import pandas as pd

def create_submission_file(test_data, predictions):
    submission_df = pd.DataFrame({
        'row_id': test_data['row_id'],
        'Category:Misconception': predictions
    })
    
    submission_df.to_csv('submission.csv', index=False)
    return submission_df
```

## 🛠️ 고급 도구 활용 (LangChain & LangGraph)

### 🎯 LangChain 기본 사용법

#### 1. 설치 및 설정
```bash
# LangChain 설치
pip install langchain langchain-community langchain-core

# LangGraph 설치
pip install langgraph

# 오픈소스 LLM (Ollama)
brew install ollama
ollama pull phi2
```

#### 2. 기본 체인 생성
```python
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

# 모델 초기화
llm = Ollama(model="phi2")

# 프롬프트 생성
prompt = PromptTemplate.from_template(
    "다음 수학 문제를 분석하세요: {question}"
)

# 체인 생성
chain = prompt | llm

# 실행
result = chain.invoke({"question": "0.355 vs 0.8"})
```

### 🔧 LangGraph 고급 워크플로우

#### 1. 상태 정의
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class AnalysisState(TypedDict):
    question: str
    student_answer: str
    student_explanation: str
    correctness_analysis: str
    misconception_analysis: str
    final_prediction: str
```

#### 2. 노드 함수들
```python
def analyze_correctness(state: AnalysisState) -> AnalysisState:
    """답의 정확성 분석"""
    # 구현...
    return state

def analyze_misconception(state: AnalysisState) -> AnalysisState:
    """오개념 분석"""
    # 구현...
    return state

def generate_prediction(state: AnalysisState) -> AnalysisState:
    """최종 예측 생성"""
    # 구현...
    return state
```

#### 3. 워크플로우 구성
```python
# 그래프 생성
workflow = StateGraph(AnalysisState)

# 노드 추가
workflow.add_node("correctness", analyze_correctness)
workflow.add_node("misconception", analyze_misconception)
workflow.add_node("prediction", generate_prediction)

# 엣지 추가
workflow.add_edge("correctness", "misconception")
workflow.add_edge("misconception", "prediction")
workflow.add_edge("prediction", END)

# 컴파일
app = workflow.compile()
```

### 🚨 주의사항

#### 1. 메모리 관리
- LangGraph는 복잡한 워크플로우를 메모리에 저장
- 큰 데이터셋 사용 시 배치 처리 고려

#### 2. 오픈소스 LLM 설정
- Ollama 설치 필요: `brew install ollama`
- 모델 다운로드: `ollama pull phi2`

#### 3. 캐글 환경 제약
- 인터넷 접근 제한으로 인해 로컬 모델만 사용 가능
- 사전 다운로드된 모델 활용

## 🔗 문서 간 네비게이션

### 📋 권장 진행 순서
1. **[환경 설정](#-환경-설정-가이드)** → 2. **[EDA 가이드](../eda/README.md)** → 3. **[모델 개발 가이드](model-development-guide.md)** → 4. **[고급 도구 활용](#-고급-도구-활용-langchain--langgraph)**

### 🎯 각 단계별 목표

#### 📋 1단계: 환경 설정
- **목표**: 개발 환경 구축 및 기본 도구 설치
- **완료 기준**: EDA 대시보드가 정상 실행됨
- **다음 단계**: [EDA 가이드](../eda/README.md)로 이동

#### 📊 2단계: 데이터 분석
- **목표**: 데이터 구조 파악 및 패턴 발견
- **완료 기준**: 주요 인사이트 도출 및 모델 설계 방향 결정
- **다음 단계**: [모델 개발 가이드](model-development-guide.md)로 이동

#### 🤖 3단계: 모델 개발
- **목표**: 3단계 예측 모델 구축 및 성능 최적화
- **완료 기준**: MAP@3 성능 지표 달성 및 제출 파일 생성
- **다음 단계**: [고급 도구 활용](#-고급-도구-활용-langchain--langgraph)으로 이동

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