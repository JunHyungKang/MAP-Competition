import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import warnings

warnings.filterwarnings("ignore")

# 한글 폰트 설정
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


def load_data():
    """데이터 로드"""
    train_df = pd.read_csv("../data/train.csv")
    test_df = pd.read_csv("../data/test.csv")
    sample_submission_df = pd.read_csv("../data/sample_submission.csv")

    return train_df, test_df, sample_submission_df


def basic_info(df, name):
    """기본 정보 출력"""
    print(f"\n=== {name} 데이터 기본 정보 ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Data types:\n{df.dtypes}")

    if "Category" in df.columns:
        print(f"\nCategory 분포:")
        print(df["Category"].value_counts())

    if "Misconception" in df.columns:
        print(f"\nMisconception 분포:")
        print(df["Misconception"].value_counts())


def analyze_text_length(df):
    """텍스트 길이 분석"""
    if "QuestionText" in df.columns:
        df["QuestionText_length"] = df["QuestionText"].str.len()
        df["StudentExplanation_length"] = df["StudentExplanation"].str.len()

        print("\n=== 텍스트 길이 분석 ===")
        print(
            f"QuestionText 길이 - 평균: {df['QuestionText_length'].mean():.2f}, 표준편차: {df['QuestionText_length'].std():.2f}"
        )
        print(
            f"StudentExplanation 길이 - 평균: {df['StudentExplanation_length'].mean():.2f}, 표준편차: {df['StudentExplanation_length'].std():.2f}"
        )

        return df


def analyze_questions(df):
    """문제 분석"""
    if "QuestionId" in df.columns:
        print("\n=== 문제 분석 ===")
        unique_questions = df["QuestionId"].nunique()
        total_answers = len(df)
        avg_answers_per_question = total_answers / unique_questions

        print(f"고유 문제 수: {unique_questions}")
        print(f"총 답변 수: {total_answers}")
        print(f"문제당 평균 답변 수: {avg_answers_per_question:.2f}")

        # 가장 많은 답변이 있는 문제들
        question_counts = df["QuestionId"].value_counts().head(10)
        print(f"\n가장 많은 답변이 있는 문제들:")
        for qid, count in question_counts.items():
            question_text = df[df["QuestionId"] == qid]["QuestionText"].iloc[0]
            print(f"QuestionId {qid}: {count}개 답변 - {question_text[:100]}...")


def create_visualizations(df):
    """시각화 생성"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Category 분포
    if "Category" in df.columns:
        category_counts = df["Category"].value_counts()
        axes[0, 0].pie(
            category_counts.values, labels=category_counts.index, autopct="%1.1f%%"
        )
        axes[0, 0].set_title("Category 분포")

    # 2. Misconception 분포
    if "Misconception" in df.columns:
        misconception_counts = df["Misconception"].value_counts()
        axes[0, 1].bar(range(len(misconception_counts)), misconception_counts.values)
        axes[0, 1].set_xticks(range(len(misconception_counts)))
        axes[0, 1].set_xticklabels(misconception_counts.index, rotation=45)
        axes[0, 1].set_title("Misconception 분포")

    # 3. 텍스트 길이 분포
    if "QuestionText_length" in df.columns:
        axes[1, 0].hist(df["QuestionText_length"], bins=30, alpha=0.7)
        axes[1, 0].set_title("QuestionText 길이 분포")
        axes[1, 0].set_xlabel("길이")
        axes[1, 0].set_ylabel("빈도")

    if "StudentExplanation_length" in df.columns:
        axes[1, 1].hist(
            df["StudentExplanation_length"], bins=30, alpha=0.7, color="orange"
        )
        axes[1, 1].set_title("StudentExplanation 길이 분포")
        axes[1, 1].set_xlabel("길이")
        axes[1, 1].set_ylabel("빈도")

    plt.tight_layout()
    plt.savefig("data_analysis_plots.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    """메인 함수"""
    print("=== 수학 오해 데이터 분석 ===")

    # 데이터 로드
    train_df, test_df, sample_submission_df = load_data()

    # 기본 정보 출력
    basic_info(train_df, "Train")
    basic_info(test_df, "Test")
    basic_info(sample_submission_df, "Sample Submission")

    # 텍스트 길이 분석
    train_df = analyze_text_length(train_df)

    # 문제 분석
    analyze_questions(train_df)

    # 시각화
    create_visualizations(train_df)

    print("\n=== 분석 완료 ===")
    print("시각화 결과가 'data_analysis_plots.png' 파일로 저장되었습니다.")


if __name__ == "__main__":
    main()
