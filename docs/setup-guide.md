# 🛠️ 환경 설정 가이드

이 가이드는 MAP-Competition 프로젝트를 위한 완전한 개발 환경을 설정하는 방법을 설명합니다.

## 📋 문서 개요

### 목적 및 범위
이 문서는 MAP-Competition 프로젝트의 개발 환경을 처음부터 완전히 설정하는 방법을 단계별로 안내합니다. Python 가상환경 설정부터 데이터 준비까지 모든 과정을 포함합니다.

### 주요 내용
- Python 가상환경 설정 및 관리
- 필요한 패키지 설치 및 의존성 관리
- 개발 도구 설정 및 구성
- 데이터 파일 준비 및 확인
- 일반적인 문제 해결 방법

### 대상 독자
- MAP-Competition 프로젝트를 처음 시작하는 개발자
- Python 개발 환경 설정이 필요한 사용자
- 프로젝트 의존성 관리에 관심이 있는 개발자

## 🚀 빠른 시작

### 핵심 단계 요약
1. **저장소 클론** (2분)
2. **Python 가상환경 생성** (1분)
3. **의존성 패키지 설치** (5-10분)
4. **데이터 준비** (2분)
5. **환경 확인** (1분)

### 주요 명령어
```bash
git clone <repository-url>
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 예상 소요 시간
- **전체 설정**: 10-15분
- **문제 해결 포함**: 20-30분

## 📋 사전 요구사항

## 📋 사전 요구사항

### 시스템 요구사항
- **Python**: 3.8 이상
- **RAM**: 최소 8GB (16GB 권장)
- **저장공간**: 최소 10GB
- **OS**: Windows, macOS, Linux

### 권장 사양
- **CPU**: 4코어 이상
- **RAM**: 16GB 이상
- **GPU**: NVIDIA GPU (선택사항, 딥러닝 모델용)
- **저장공간**: SSD 50GB 이상

## 🚀 단계별 설정

### 1. 저장소 클론

```bash
# GitHub에서 저장소 클론
git clone <repository-url>
cd MAP-Competition

# 또는 새로 시작하는 경우
mkdir MAP-Competition
cd MAP-Competition
```

### 2. Python 가상환경 생성

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

### 3. 의존성 패키지 설치

```bash
# 기본 패키지 설치
pip install -r requirements.txt

# 또는 개별 설치
pip install pandas numpy matplotlib seaborn plotly
pip install dash dash-bootstrap-components
pip install streamlit
pip install scikit-learn
```

### 4. 데이터 준비

```bash
# data 폴더 생성
mkdir -p data

# 데이터 파일을 data/ 폴더에 복사
# train.csv, test.csv, sample_submission.csv
```

### 5. 환경 확인

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

## 📚 다음 단계

환경 설정이 완료되면 다음 문서들을 참조하여 프로젝트를 진행하세요:

1. **[데이터 분석 가이드](eda-guide.md)**: 데이터 탐색 및 시각화
2. **[모델 개발 가이드](model-development-guide.md)**: 수학 오해 예측 모델 개발
3. **[도구 사용 가이드](langchain-langgraph-guide.md)**: LangChain과 LangGraph 활용

### 권장 진행 순서
1. **데이터 다운로드**: 캐글에서 MAP 대회 데이터 다운로드
2. **데이터 분석**: [EDA 가이드](eda-guide.md)를 참조하여 데이터 탐색
3. **모델 개발**: [모델 개발 가이드](model-development-guide.md)를 참조하여 모델 구축
4. **고급 도구 활용**: [도구 사용 가이드](langchain-langgraph-guide.md)를 참조하여 LLM 워크플로우 구축

## 📚 추가 리소스

### 유용한 링크
- [Python 공식 문서](https://docs.python.org/)
- [Pandas 공식 문서](https://pandas.pydata.org/docs/)
- [Dash 공식 문서](https://dash.plotly.com/)
- [Streamlit 공식 문서](https://docs.streamlit.io/)

### 문제 해결 커뮤니티
- [Stack Overflow](https://stackoverflow.com/)
- [GitHub Issues](https://github.com/your-repo/issues)
- [Kaggle Forums](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/discussion)

## ✅ 체크리스트

환경 설정이 완료되었는지 확인하세요:

- [ ] Python 3.8+ 설치됨
- [ ] 가상환경 생성 및 활성화됨
- [ ] 모든 패키지 설치됨
- [ ] 데이터 파일 준비됨
- [ ] EDA 대시보드 실행됨
- [ ] 기본 테스트 통과됨

모든 항목이 체크되면 환경 설정이 완료된 것입니다! 🎉 