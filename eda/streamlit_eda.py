import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import re

# 페이지 설정
st.set_page_config(
    page_title="수학 오해 데이터 EDA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 한글 폰트 설정
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


@st.cache_data
def load_data():
    """데이터 로드 및 전처리"""
    train_df = pd.read_csv("../data/train.csv")
    test_df = pd.read_csv("../data/test.csv")

    # 텍스트 길이 계산
    train_df["QuestionText_length"] = train_df["QuestionText"].str.len()
    train_df["StudentExplanation_length"] = train_df["StudentExplanation"].str.len()

    return train_df, test_df


def main():
    st.title("📊 수학 오해 데이터 EDA 대시보드")
    st.markdown("---")

    # 데이터 로드
    train_df, test_df = load_data()

    # 사이드바 필터
    st.sidebar.header("🔍 필터 설정")

    # Category 필터
    categories = ["전체"] + list(train_df["Category"].unique())
    selected_category = st.sidebar.selectbox("Category 선택", categories)

    # Misconception 필터
    misconception_df = train_df.dropna(subset=["Misconception"])
    misconceptions = ["전체"] + list(misconception_df["Misconception"].unique())
    selected_misconception = st.sidebar.selectbox("Misconception 선택", misconceptions)

    # QuestionId 필터
    questions = ["전체"] + [
        f"Question {qid}" for qid in train_df["QuestionId"].unique()
    ]
    selected_question = st.sidebar.selectbox("QuestionId 선택", questions)

    # 필터링된 데이터
    filtered_df = train_df.copy()

    if selected_category != "전체":
        filtered_df = filtered_df[filtered_df["Category"] == selected_category]

    if selected_misconception != "전체":
        filtered_df = filtered_df[
            filtered_df["Misconception"] == selected_misconception
        ]

    if selected_question != "전체":
        question_id = int(selected_question.split()[-1])
        filtered_df = filtered_df[filtered_df["QuestionId"] == question_id]

    # 통계 카드
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("총 샘플 수", f"{len(filtered_df):,}")

    with col2:
        st.metric("고유 문제 수", filtered_df["QuestionId"].nunique())

    with col3:
        avg_length = filtered_df["StudentExplanation_length"].mean()
        st.metric("평균 답변 길이", f"{avg_length:.1f}자")

    with col4:
        misconception_ratio = (
            len(filtered_df.dropna(subset=["Misconception"])) / len(filtered_df) * 100
        )
        st.metric("Misconception 비율", f"{misconception_ratio:.1f}%")

    st.markdown("---")

    # 차트 섹션
    st.header("📈 데이터 시각화")

    # 첫 번째 행
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Category 분포")
        category_counts = filtered_df["Category"].value_counts()
        fig = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Category 분포",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top 10 Misconception 유형")
        misconception_counts = (
            filtered_df.dropna(subset=["Misconception"])["Misconception"]
            .value_counts()
            .head(10)
        )
        fig = px.bar(
            x=misconception_counts.values,
            y=misconception_counts.index,
            orientation="h",
            title="Top 10 Misconception 유형",
        )
        fig.update_layout(xaxis_title="개수", yaxis_title="Misconception 유형")
        st.plotly_chart(fig, use_container_width=True)

    # 두 번째 행
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("텍스트 길이 분포")
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("QuestionText 길이 분포", "StudentExplanation 길이 분포"),
        )

        fig.add_trace(
            go.Histogram(
                x=filtered_df["QuestionText_length"], name="QuestionText", nbinsx=30
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Histogram(
                x=filtered_df["StudentExplanation_length"],
                name="StudentExplanation",
                nbinsx=30,
            ),
            row=2,
            col=1,
        )

        fig.update_layout(height=500, title_text="텍스트 길이 분포")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("문제별 답변 수")
        question_counts = filtered_df["QuestionId"].value_counts()
        fig = px.bar(
            x=question_counts.index, y=question_counts.values, title="문제별 답변 수"
        )
        fig.update_layout(xaxis_title="QuestionId", yaxis_title="답변 수")
        st.plotly_chart(fig, use_container_width=True)

    # 세 번째 행 - 히트맵
    st.subheader("Category vs Misconception 히트맵")
    misconception_filtered_df = filtered_df.dropna(subset=["Misconception"])
    if len(misconception_filtered_df) > 0:
        cross_tab = pd.crosstab(
            misconception_filtered_df["Category"],
            misconception_filtered_df["Misconception"],
        )
        fig = px.imshow(
            cross_tab,
            title="Category vs Misconception 히트맵",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("선택된 필터에 해당하는 Misconception 데이터가 없습니다.")

    # 네 번째 행 - 상세 분석
    st.markdown("---")
    st.header("🔍 상세 분석")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("문제별 Category 분포")
        question_category = (
            filtered_df.groupby(["QuestionId", "Category"]).size().unstack(fill_value=0)
        )
        fig = px.bar(question_category, title="문제별 Category 분포", barmode="stack")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("답변 길이 vs Category")
        fig = px.box(
            filtered_df,
            x="Category",
            y="StudentExplanation_length",
            title="Category별 답변 길이 분포",
        )
        st.plotly_chart(fig, use_container_width=True)

    # 다섯 번째 행 - 샘플 데이터
    st.markdown("---")
    st.header("📋 샘플 데이터")

    # 샘플 데이터 표시
    sample_cols = ["QuestionId", "Category", "Misconception", "StudentExplanation"]
    sample_data = filtered_df[sample_cols].head(20)

    st.dataframe(sample_data, use_container_width=True, height=400)

    # 다운로드 버튼
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="필터링된 데이터 다운로드 (CSV)",
        data=csv,
        file_name=f"filtered_data_{selected_category}_{selected_misconception}_{selected_question}.csv",
        mime="text/csv",
    )

    # 추가 분석
    st.markdown("---")
    st.header("📊 추가 분석")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("가장 긴 답변 Top 10")
        longest_explanations = filtered_df.nlargest(10, "StudentExplanation_length")[
            [
                "QuestionId",
                "Category",
                "StudentExplanation",
                "StudentExplanation_length",
            ]
        ]
        for idx, row in longest_explanations.iterrows():
            with st.expander(
                f"Question {row['QuestionId']} - {row['StudentExplanation_length']}자"
            ):
                st.write(f"**Category:** {row['Category']}")
                st.write(f"**답변:** {row['StudentExplanation']}")

    with col2:
        st.subheader("가장 짧은 답변 Top 10")
        shortest_explanations = filtered_df.nsmallest(10, "StudentExplanation_length")[
            [
                "QuestionId",
                "Category",
                "StudentExplanation",
                "StudentExplanation_length",
            ]
        ]
        for idx, row in shortest_explanations.iterrows():
            with st.expander(
                f"Question {row['QuestionId']} - {row['StudentExplanation_length']}자"
            ):
                st.write(f"**Category:** {row['Category']}")
                st.write(f"**답변:** {row['StudentExplanation']}")


if __name__ == "__main__":
    main()
