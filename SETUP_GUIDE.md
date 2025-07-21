# 🛠️ 환경 설정 가이드

이 가이드는 MAP 프로젝트를 위한 개발 환경을 설정하는 방법을 설명합니다.

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
cd MAP

# 또는 새로 시작하는 경우
mkdir MAP
cd MAP
```

### 2. Python 가상환경 생성

#### macOS/Linux
```bash
# Python 가상환경 생성
python3 -m venv venv

# 가상환경 활성화
source venv/bin/activate

# 가상환경이 활성화되었는지 확인
which python
# 출력: /path/to/MAP/venv/bin/python
```

#### Windows
```bash
# Python 가상환경 생성
python -m venv venv

# 가상환경 활성화
venv\Scripts\activate

# 가상환경이 활성화되었는지 확인
where python
# 출력: C:\path\to\MAP\venv\Scripts\python.exe
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

## 🔧 문제 해결

### 일반적인 문제들

#### 1. 가상환경 활성화 실패
```bash
# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

# 만약 위 명령어가 작동하지 않으면
python -m venv venv --clear
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

## 🛠️ 개발 도구 설정

### VS Code 설정
```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true
}
```

### Jupyter Notebook 설정
```bash
# Jupyter 설치
pip install jupyter notebook

# Jupyter 실행
jupyter notebook
```

### Git 설정
```bash
# Git 초기 설정
git init
git add .
git commit -m "Initial commit"

# .gitignore 확인
cat .gitignore
```

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