# MAP 대회 가상환경 설정 가이드

## 🎯 가상환경 개요

MAP 대회 참가를 위한 Python 가상환경이 설정되었습니다. 이 가상환경은 오픈소스 LLM 사용에 필요한 모든 패키지들이 설치되어 있습니다.

## 📁 프로젝트 구조

```
MAP/
├── README.md                                    # 대회 개요 및 가이드
├── requirements.txt                             # 필요한 패키지 목록
├── venv_setup_guide.md                         # 이 파일 (가상환경 가이드)
├── .venv/                                      # 가상환경 폴더
│   ├── bin/                                    # 가상환경 실행 파일들
│   ├── lib/                                    # 설치된 패키지들
│   └── ...
└── reference_notebooks/                         # 참고 자료
    ├── modernbert-large-cv-0-938.ipynb         # 원본 노트북
    ├── modernbert-large-cv-0-938_한글정리.md   # 노트북 분석
    ├── extracted_modules_for_opensource_llm.py  # 추출된 모듈들
    └── opensource_llm_guide.md                 # 오픈소스 LLM 가이드
```

## 🚀 가상환경 사용법

### 1. 가상환경 활성화

**macOS/Linux:**
```bash
source .venv/bin/activate
```

**Windows:**
```bash
.venv\Scripts\activate
```

### 2. 가상환경 비활성화
```bash
deactivate
```

### 3. 설치된 패키지 확인
```bash
pip list
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

## 📚 다음 단계

1. **데이터 다운로드**: 캐글에서 MAP 대회 데이터 다운로드
2. **노트북 생성**: Jupyter 노트북에서 실험 시작
3. **모델 선택**: Phi-2, Gemma 등 오픈소스 LLM 선택
4. **특성 엔지니어링**: 답의 정확성 특성 활용
5. **프롬프트 최적화**: 오픈소스 LLM용 프롬프트 설계

## 🆘 문제 해결

### 가상환경 활성화 실패
```bash
# Python 버전 확인
python3 --version

# 가상환경 재생성
python3 -m venv .venv --clear
```

### 패키지 설치 실패
```bash
# pip 업그레이드
pip install --upgrade pip

# 캐시 클리어
pip cache purge

# 개별 패키지 설치
pip install package_name --no-cache-dir
```

### 메모리 부족
```bash
# 배치 크기 줄이기
# 그래디언트 체크포인팅 사용
# 더 작은 모델 선택
```

이제 가상환경이 준비되었으니 MAP 대회 참가를 시작하세요! 🚀 