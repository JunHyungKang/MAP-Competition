# LangChain & LangGraph 가이드 - MAP 대회

## 🎯 설치된 LangChain & LangGraph 패키지들

MAP 대회를 위한 LangChain과 LangGraph 관련 패키지들이 최신 버전으로 설치되었습니다.

### 📦 설치된 패키지 목록

| 패키지명 | 버전 | 설명 |
|----------|------|------|
| `langchain` | 0.3.26 | LangChain 핵심 라이브러리 |
| `langchain-community` | 0.3.27 | 커뮤니티 통합 |
| `langchain-core` | 0.3.69 | 핵심 기능 |
| `langchain-openai` | 0.3.28 | OpenAI 통합 |
| `langchain-anthropic` | 0.3.17 | Anthropic 통합 |
| `langchain-google-genai` | 2.1.8 | Google Gemini 통합 |
| `langgraph` | 0.5.3 | LangGraph (워크플로우) |
| `langchain-experimental` | 0.3.4 | 실험적 기능 |
| `langsmith` | 0.4.8 | LangSmith (모니터링) |

## 🚀 MAP 대회에서의 활용 방안

### 1. **LangChain을 활용한 프롬프트 체인**

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama

# 프롬프트 템플릿 생성
prompt = PromptTemplate.from_template(
    "다음 수학 문제와 학생의 답변을 분석하여 오개념을 식별하세요:\n\n"
    "문제: {question}\n"
    "학생 답변: {student_answer}\n"
    "학생 설명: {student_explanation}\n\n"
    "분석:"
)

# 오픈소스 LLM 사용 (Ollama)
llm = Ollama(model="phi2")

# 체인 생성
chain = prompt | llm | StrOutputParser()

# 사용 예시
result = chain.invoke({
    "question": "0.355와 0.8 중 어느 것이 더 큰가요?",
    "student_answer": "0.355",
    "student_explanation": "355가 8보다 크니까 0.355가 더 커요"
})
```

### 2. **LangGraph를 활용한 복잡한 워크플로우**

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage

# 상태 정의
class AnalysisState(TypedDict):
    question: str
    student_answer: str
    student_explanation: str
    correctness_analysis: str
    misconception_analysis: str
    final_prediction: str

# 노드 함수들
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

### 3. **다중 모델 앙상블**

```python
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser

# 여러 모델 초기화
models = {
    "phi2": Ollama(model="phi2"),
    "gemma": Ollama(model="gemma2"),
    "llama": Ollama(model="llama2")
}

# 앙상블 체인
def ensemble_analysis(input_data):
    results = {}
    for name, model in models.items():
        chain = prompt | model | JsonOutputParser()
        results[name] = chain.invoke(input_data)
    return results

# 사용
ensemble_result = ensemble_analysis({
    "question": "수학 문제",
    "student_answer": "학생 답변",
    "student_explanation": "학생 설명"
})
```

## 💡 MAP 대회 특화 활용법

### 1. **특성 엔지니어링과 LangChain 결합**

```python
from reference_notebooks.extracted_modules_for_opensource_llm import MAPDataProcessor, PromptEngineer

# 데이터 처리
processor = MAPDataProcessor()
train_data = processor.load_and_preprocess_data("train.csv")
train_data = processor.engineer_correctness_feature(train_data)

# LangChain과 결합
from langchain_core.prompts import PromptTemplate

# MAP 특화 프롬프트
map_prompt = PromptTemplate.from_template("""
수학 교육 전문가로서 학생의 답변을 분석하세요.

문제: {question}
학생 답변: {mc_answer}
답의 정확성: {is_correct}
학생 설명: {student_explanation}

다음 중 하나를 선택하세요:
1. True_Correct: 정답이고 설명도 정확
2. True_Misconception: 정답이지만 설명에 오개념
3. False_Correct: 오답이지만 설명은 정확
4. False_Misconception: 오답이고 설명에도 오개념
5. False_Neither: 오답이고 설명도 부적절

분석 결과:
""")

# 체인 생성
chain = map_prompt | llm | StrOutputParser()
```

### 2. **LangGraph를 활용한 단계별 분석**

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class MAPAnalysisState(TypedDict):
    question_id: str
    question_text: str
    mc_answer: str
    student_explanation: str
    is_correct: bool
    step1_analysis: str
    step2_analysis: str
    step3_analysis: str
    final_category: str
    final_misconception: str

def step1_analyze_correctness(state: MAPAnalysisState) -> MAPAnalysisState:
    """1단계: 답의 정확성 분석"""
    # 구현...
    return state

def step2_analyze_explanation(state: MAPAnalysisState) -> MAPAnalysisState:
    """2단계: 설명 분석"""
    # 구현...
    return state

def step3_identify_misconception(state: MAPAnalysisState) -> MAPAnalysisState:
    """3단계: 오개념 식별"""
    # 구현...
    return state

# 워크플로우 생성
workflow = StateGraph(MAPAnalysisState)
workflow.add_node("step1", step1_analyze_correctness)
workflow.add_node("step2", step2_analyze_explanation)
workflow.add_node("step3", step3_identify_misconception)

workflow.add_edge("step1", "step2")
workflow.add_edge("step2", "step3")
workflow.add_edge("step3", END)

app = workflow.compile()
```

### 3. **LangSmith를 활용한 실험 추적**

```python
import os
from langsmith import Client

# LangSmith 설정
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "map-competition"

# 실험 추적
client = Client()

# 체인 실행 및 추적
with client.trace("map-analysis") as tracer:
    result = chain.invoke({
        "question": "수학 문제",
        "mc_answer": "학생 답변",
        "is_correct": True,
        "student_explanation": "학생 설명"
    })
```

## 🔧 실제 사용 예시

### 1. **기본 사용법**

```python
# 가상환경 활성화 후 (.venv)
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
print(result)
```

### 2. **고급 사용법 (LangGraph)**

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class MathAnalysisState(TypedDict):
    input_text: str
    question_analysis: str
    misconception_detection: str
    final_output: str

def analyze_question(state: MathAnalysisState) -> MathAnalysisState:
    # 질문 분석 로직
    return state

def detect_misconception(state: MathAnalysisState) -> MathAnalysisState:
    # 오개념 감지 로직
    return state

def generate_output(state: MathAnalysisState) -> MathAnalysisState:
    # 최종 출력 생성
    return state

# 그래프 생성
workflow = StateGraph(MathAnalysisState)
workflow.add_node("analyze", analyze_question)
workflow.add_node("detect", detect_misconception)
workflow.add_node("output", generate_output)

workflow.add_edge("analyze", "detect")
workflow.add_edge("detect", "output")
workflow.add_edge("output", END)

app = workflow.compile()
```

## 🚨 주의사항

### 1. **메모리 관리**
- LangGraph는 복잡한 워크플로우를 메모리에 저장
- 큰 데이터셋 사용 시 배치 처리 고려

### 2. **오픈소스 LLM 설정**
- Ollama 설치 필요: `brew install ollama`
- 모델 다운로드: `ollama pull phi2`

### 3. **캐글 환경 제약**
- 인터넷 접근 제한으로 인해 로컬 모델만 사용 가능
- 사전 다운로드된 모델 활용

## 📚 다음 단계

1. **Ollama 설치**: `brew install ollama`
2. **모델 다운로드**: `ollama pull phi2`
3. **실험 시작**: Jupyter 노트북에서 LangChain 사용
4. **워크플로우 설계**: LangGraph로 복잡한 분석 체인 구축
5. **성능 최적화**: LangSmith로 실험 추적

이제 LangChain과 LangGraph를 활용하여 MAP 대회에서 더욱 정교한 분석을 수행할 수 있습니다! 🚀 