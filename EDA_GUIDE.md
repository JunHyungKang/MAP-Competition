# 📊 EDA (Exploratory Data Analysis) 사용 가이드

이 가이드는 MAP 프로젝트의 EDA 대시보드와 분석 도구들을 사용하는 방법을 설명합니다.

## 🎯 EDA 목표

### 주요 목표
- 📈 **데이터 이해**: 수학 오해 데이터의 구조와 특성 파악
- 🔍 **패턴 발견**: Category와 Misconception 간의 관계 분석
- 📊 **시각화**: 인터랙티브 차트를 통한 직관적 데이터 탐색
- 🎯 **인사이트 도출**: 모델 개발에 필요한 특징 발견

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

## 📊 대시보드 기능

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

#### 3. 빠른 프로토타이핑
- **간단한 API**: 몇 줄로 새로운 분석 추가 가능
- **실시간 개발**: 코드 변경 시 즉시 반영

## 🔍 데이터 분석 가이드

### 1. 기본 통계 분석

#### Category 분포 분석
```python
# Category별 분포 확인
category_counts = df['Category'].value_counts()
print(category_counts)

# 비율 계산
category_ratio = df['Category'].value_counts(normalize=True) * 100
print(category_ratio)
```

#### Misconception 분석
```python
# Misconception이 있는 데이터만 필터링
misconception_df = df.dropna(subset=['Misconception'])

# Top 10 Misconception 확인
top_misconceptions = misconception_df['Misconception'].value_counts().head(10)
print(top_misconceptions)
```

### 2. 텍스트 분석

#### 길이 분석
```python
# 텍스트 길이 계산
df['QuestionText_length'] = df['QuestionText'].str.len()
df['StudentExplanation_length'] = df['StudentExplanation'].str.len()

# 통계 정보
print(f"문제 텍스트 평균 길이: {df['QuestionText_length'].mean():.2f}")
print(f"학생 설명 평균 길이: {df['StudentExplanation_length'].mean():.2f}")
```

#### 문제별 분석
```python
# 문제별 답변 수
question_counts = df['QuestionId'].value_counts()
print(f"가장 많은 답변이 있는 문제: {question_counts.index[0]} ({question_counts.iloc[0]}개)")
```

### 3. 교차 분석

#### Category vs Misconception
```python
# 교차표 생성
cross_tab = pd.crosstab(df['Category'], df['Misconception'])
print(cross_tab)

# 히트맵으로 시각화
import seaborn as sns
sns.heatmap(cross_tab, annot=True, fmt='d')
```

#### 문제별 Category 분포
```python
# 문제별 Category 분포
question_category = df.groupby(['QuestionId', 'Category']).size().unstack(fill_value=0)
print(question_category)
```

## 💡 사용 팁

### Plotly Dash 사용 시

#### 1. 필터 조합 활용
- **다중 필터**: Category와 Misconception을 동시에 선택하여 특정 패턴 발견
- **단계적 분석**: 전체 → Category → Misconception 순으로 세분화

#### 2. 차트 상호작용
- **클릭 필터링**: 차트 요소를 클릭하여 해당 항목만 필터링
- **호버 정보**: 마우스 오버로 상세 통계 확인

#### 3. 실시간 탐색
- **즉시 피드백**: 필터 변경 시 모든 차트가 동시에 업데이트
- **패턴 발견**: 여러 필터를 조합하여 숨겨진 패턴 발견

### Streamlit 사용 시

#### 1. 데이터 다운로드
- **관심 데이터 저장**: 특정 필터 조건의 데이터를 CSV로 저장
- **분석 결과 공유**: 팀원과 분석 결과 공유

#### 2. 상세 분석
- **개별 샘플 확인**: 확장 가능한 섹션에서 실제 답변 내용 확인
- **패턴 분석**: 긴/짧은 답변의 특징 분석

#### 3. 빠른 프로토타이핑
- **새로운 분석 추가**: 간단한 코드로 새로운 차트나 분석 추가
- **실험적 접근**: 다양한 분석 아이디어를 빠르게 테스트

## 🔍 고급 분석 기법

### 1. 텍스트 마이닝

#### 자주 나오는 단어 분석
```python
from collections import Counter
import re

# 텍스트 전처리
def clean_text(text):
    return re.sub(r'[^\w\s]', '', text.lower())

# 단어 빈도 분석
words = ' '.join(df['StudentExplanation'].apply(clean_text)).split()
word_counts = Counter(words)
print(word_counts.most_common(20))
```

#### 감정 분석
```python
from textblob import TextBlob

# 감정 점수 계산
df['sentiment'] = df['StudentExplanation'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Category별 감정 분석
sentiment_by_category = df.groupby('Category')['sentiment'].mean()
print(sentiment_by_category)
```

### 2. 시계열 분석

#### 문제별 패턴 변화
```python
# 시간 순서로 정렬 (가정)
df_sorted = df.sort_values('row_id')

# 문제별 답변 패턴
for question_id in df['QuestionId'].unique():
    question_data = df[df['QuestionId'] == question_id]
    print(f"Question {question_id}: {len(question_data)} 답변")
```

### 3. 클러스터링

#### 유사한 오해 패턴 그룹화
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.clustering import KMeans

# 텍스트 벡터화
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['StudentExplanation'])

# 클러스터링
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# 클러스터별 분석
cluster_analysis = df.groupby('cluster')['Category'].value_counts()
print(cluster_analysis)
```

## 📊 시각화 팁

### 1. 효과적인 차트 선택

#### 분포 시각화
- **파이 차트**: Category 분포 (5개 이하 카테고리)
- **바 차트**: Misconception 분포 (순위 표시)
- **히스토그램**: 텍스트 길이 분포

#### 관계 시각화
- **히트맵**: Category vs Misconception 관계
- **산점도**: 길이 vs Category 관계
- **박스플롯**: Category별 길이 분포

### 2. 색상 및 스타일

#### 일관된 색상 사용
```python
# Category별 색상 매핑
category_colors = {
    'True_Correct': '#2E8B57',
    'False_Misconception': '#DC143C',
    'False_Neither': '#FF8C00',
    'True_Neither': '#4169E1'
}
```

#### 접근성 고려
- **색맹 친화적**: 빨강-초록 조합 피하기
- **대비**: 충분한 명암 대비 확보
- **레이블**: 모든 차트에 명확한 레이블

## 🔧 문제 해결

### 일반적인 문제들

#### 1. 대시보드가 실행되지 않는 경우
```bash
# 포트 확인
lsof -i :8050  # Dash 포트
lsof -i :8501  # Streamlit 포트

# 다른 포트 사용
python run_dash.py --port 8051
streamlit run streamlit_eda.py --server.port 8502
```

#### 2. 데이터 로딩 오류
```bash
# 데이터 경로 확인
ls data/
cat data/train.csv | head -5

# 경로 수정
# eda/ 폴더의 파일들에서 '../data/' 경로 확인
```

#### 3. 메모리 부족
```bash
# 메모리 사용량 확인
python -c "import psutil; print(f'사용률: {psutil.virtual_memory().percent}%')"

# 샘플링으로 테스트
df_sample = df.sample(n=1000, random_state=42)
```

### 성능 최적화

#### 1. 데이터 캐싱
```python
# Streamlit 캐싱 활용
@st.cache_data
def load_data():
    return pd.read_csv('../data/train.csv')
```

#### 2. 효율적인 필터링
```python
# 인덱스 활용
df.set_index('QuestionId', inplace=True)
filtered_df = df.loc[selected_questions]
```

#### 3. 차트 최적화
```python
# 대용량 데이터 샘플링
if len(df) > 10000:
    df_sample = df.sample(n=10000, random_state=42)
    # 샘플링된 데이터로 차트 생성
```

## 📚 추가 리소스

### 유용한 링크
- [Plotly Dash 공식 문서](https://dash.plotly.com/)
- [Streamlit 공식 문서](https://docs.streamlit.io/)
- [Pandas 공식 문서](https://pandas.pydata.org/docs/)
- [Seaborn 공식 문서](https://seaborn.pydata.org/)

### 학습 자료
- [데이터 시각화 모범 사례](https://www.storytellingwithdata.com/)
- [인터랙티브 대시보드 설계](https://www.nngroup.com/articles/dashboard-design/)
- [색상 이론](https://www.interaction-design.org/literature/topics/color-theory)

## ✅ 체크리스트

EDA 분석이 완료되었는지 확인하세요:

- [ ] 기본 통계 정보 확인
- [ ] Category 분포 분석
- [ ] Misconception 패턴 발견
- [ ] 텍스트 길이 분석
- [ ] 문제별 특성 파악
- [ ] 시각화 차트 생성
- [ ] 인사이트 도출
- [ ] 모델 개발 방향 설정

모든 항목이 체크되면 EDA 분석이 완료된 것입니다! 🎉 