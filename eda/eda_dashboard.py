import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from collections import Counter
import re

# 데이터 로드
train_df = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv")

# 텍스트 길이 계산
train_df["QuestionText_length"] = train_df["QuestionText"].str.len()
train_df["StudentExplanation_length"] = train_df["StudentExplanation"].str.len()

# Misconception이 NA가 아닌 데이터만 필터링
train_df_with_misconception = train_df.dropna(subset=["Misconception"])

app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.H1(
            "수학 오해 데이터 EDA 대시보드",
            style={"textAlign": "center", "color": "#2c3e50", "marginBottom": 30},
        ),
        # 필터 섹션
        html.Div(
            [
                html.H3("필터", style={"color": "#34495e"}),
                html.Div(
                    [
                        html.Label("Category 선택:"),
                        dcc.Dropdown(
                            id="category-filter",
                            options=[{"label": "전체", "value": "all"}]
                            + [
                                {"label": cat, "value": cat}
                                for cat in train_df["Category"].unique()
                            ],
                            value="all",
                            style={"width": "50%", "marginBottom": 10},
                        ),
                        html.Label("Misconception 선택:"),
                        dcc.Dropdown(
                            id="misconception-filter",
                            options=[{"label": "전체", "value": "all"}]
                            + [
                                {"label": mis, "value": mis}
                                for mis in train_df_with_misconception[
                                    "Misconception"
                                ].unique()
                            ],
                            value="all",
                            style={"width": "50%", "marginBottom": 10},
                        ),
                        html.Label("QuestionId 선택:"),
                        dcc.Dropdown(
                            id="question-filter",
                            options=[{"label": "전체", "value": "all"}]
                            + [
                                {"label": f"Question {qid}", "value": qid}
                                for qid in train_df["QuestionId"].unique()
                            ],
                            value="all",
                            style={"width": "50%", "marginBottom": 10},
                        ),
                    ],
                    style={"marginBottom": 30},
                ),
            ],
            style={
                "backgroundColor": "#ecf0f1",
                "padding": 20,
                "borderRadius": 10,
                "marginBottom": 30,
            },
        ),
        # 통계 카드
        html.Div(
            [
                html.Div(
                    [
                        html.H4(
                            "총 샘플 수",
                            style={"textAlign": "center", "color": "#2c3e50"},
                        ),
                        html.H2(
                            id="total-samples",
                            style={"textAlign": "center", "color": "#e74c3c"},
                        ),
                    ],
                    style={
                        "backgroundColor": "#fff",
                        "padding": 20,
                        "borderRadius": 10,
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                        "margin": 5,
                    },
                ),
                html.Div(
                    [
                        html.H4(
                            "고유 문제 수",
                            style={"textAlign": "center", "color": "#2c3e50"},
                        ),
                        html.H2(
                            id="unique-questions",
                            style={"textAlign": "center", "color": "#3498db"},
                        ),
                    ],
                    style={
                        "backgroundColor": "#fff",
                        "padding": 20,
                        "borderRadius": 10,
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                        "margin": 5,
                    },
                ),
                html.Div(
                    [
                        html.H4(
                            "평균 답변 길이",
                            style={"textAlign": "center", "color": "#2c3e50"},
                        ),
                        html.H2(
                            id="avg-explanation-length",
                            style={"textAlign": "center", "color": "#f39c12"},
                        ),
                    ],
                    style={
                        "backgroundColor": "#fff",
                        "padding": 20,
                        "borderRadius": 10,
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                        "margin": 5,
                    },
                ),
                html.Div(
                    [
                        html.H4(
                            "Misconception 비율",
                            style={"textAlign": "center", "color": "#2c3e50"},
                        ),
                        html.H2(
                            id="misconception-ratio",
                            style={"textAlign": "center", "color": "#9b59b6"},
                        ),
                    ],
                    style={
                        "backgroundColor": "#fff",
                        "padding": 20,
                        "borderRadius": 10,
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                        "margin": 5,
                    },
                ),
            ],
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "marginBottom": 30,
            },
        ),
        # 차트 섹션
        html.Div(
            [
                # 첫 번째 행
                html.Div(
                    [
                        html.Div(
                            [dcc.Graph(id="category-pie-chart")], style={"width": "50%"}
                        ),
                        html.Div(
                            [dcc.Graph(id="misconception-bar-chart")],
                            style={"width": "50%"},
                        ),
                    ],
                    style={"display": "flex", "marginBottom": 30},
                ),
                # 두 번째 행
                html.Div(
                    [
                        html.Div(
                            [dcc.Graph(id="text-length-distribution")],
                            style={"width": "50%"},
                        ),
                        html.Div(
                            [dcc.Graph(id="question-distribution")],
                            style={"width": "50%"},
                        ),
                    ],
                    style={"display": "flex", "marginBottom": 30},
                ),
                # 세 번째 행
                html.Div(
                    [dcc.Graph(id="category-misconception-heatmap")],
                    style={"width": "100%"},
                ),
                # 네 번째 행 - 샘플 데이터 테이블
                html.Div(
                    [
                        html.H3(
                            "샘플 데이터",
                            style={"color": "#34495e", "marginBottom": 20},
                        ),
                        html.Div(id="sample-data-table"),
                    ],
                    style={"marginTop": 30},
                ),
            ],
            style={"backgroundColor": "#f8f9fa", "padding": 20, "borderRadius": 10},
        ),
    ]
)


# 콜백 함수들
@callback(
    [
        Output("total-samples", "children"),
        Output("unique-questions", "children"),
        Output("avg-explanation-length", "children"),
        Output("misconception-ratio", "children"),
    ],
    [
        Input("category-filter", "value"),
        Input("misconception-filter", "value"),
        Input("question-filter", "value"),
    ],
)
def update_stats(category, misconception, question):
    # 필터링
    filtered_df = train_df.copy()

    if category != "all":
        filtered_df = filtered_df[filtered_df["Category"] == category]

    if misconception != "all":
        filtered_df = filtered_df[filtered_df["Misconception"] == misconception]

    if question != "all":
        filtered_df = filtered_df[filtered_df["QuestionId"] == question]

    # 통계 계산
    total_samples = len(filtered_df)
    unique_questions = filtered_df["QuestionId"].nunique()
    avg_explanation_length = f"{filtered_df['StudentExplanation_length'].mean():.1f}자"

    # Misconception 비율 (NA 제외)
    non_na_misconception = filtered_df.dropna(subset=["Misconception"])
    misconception_ratio = f"{len(non_na_misconception) / len(filtered_df) * 100:.1f}%"

    return total_samples, unique_questions, avg_explanation_length, misconception_ratio


@callback(
    Output("category-pie-chart", "figure"),
    [
        Input("category-filter", "value"),
        Input("misconception-filter", "value"),
        Input("question-filter", "value"),
    ],
)
def update_category_pie(category, misconception, question):
    filtered_df = train_df.copy()

    if category != "all":
        filtered_df = filtered_df[filtered_df["Category"] == category]
    if misconception != "all":
        filtered_df = filtered_df[filtered_df["Misconception"] == misconception]
    if question != "all":
        filtered_df = filtered_df[filtered_df["QuestionId"] == question]

    category_counts = filtered_df["Category"].value_counts()

    fig = px.pie(
        values=category_counts.values,
        names=category_counts.index,
        title="Category 분포",
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    fig.update_layout(height=400)
    return fig


@callback(
    Output("misconception-bar-chart", "figure"),
    [
        Input("category-filter", "value"),
        Input("misconception-filter", "value"),
        Input("question-filter", "value"),
    ],
)
def update_misconception_bar(category, misconception, question):
    filtered_df = train_df_with_misconception.copy()

    if category != "all":
        filtered_df = filtered_df[filtered_df["Category"] == category]
    if misconception != "all":
        filtered_df = filtered_df[filtered_df["Misconception"] == misconception]
    if question != "all":
        filtered_df = filtered_df[filtered_df["QuestionId"] == question]

    misconception_counts = filtered_df["Misconception"].value_counts().head(10)

    fig = px.bar(
        x=misconception_counts.values,
        y=misconception_counts.index,
        orientation="h",
        title="Top 10 Misconception 유형",
        color_discrete_sequence=["#e74c3c"],
    )
    fig.update_layout(height=400, xaxis_title="개수", yaxis_title="Misconception 유형")
    return fig


@callback(
    Output("text-length-distribution", "figure"),
    [
        Input("category-filter", "value"),
        Input("misconception-filter", "value"),
        Input("question-filter", "value"),
    ],
)
def update_text_length_dist(category, misconception, question):
    filtered_df = train_df.copy()

    if category != "all":
        filtered_df = filtered_df[filtered_df["Category"] == category]
    if misconception != "all":
        filtered_df = filtered_df[filtered_df["Misconception"] == misconception]
    if question != "all":
        filtered_df = filtered_df[filtered_df["QuestionId"] == question]

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
    return fig


@callback(
    Output("question-distribution", "figure"),
    [
        Input("category-filter", "value"),
        Input("misconception-filter", "value"),
        Input("question-filter", "value"),
    ],
)
def update_question_dist(category, misconception, question):
    filtered_df = train_df.copy()

    if category != "all":
        filtered_df = filtered_df[filtered_df["Category"] == category]
    if misconception != "all":
        filtered_df = filtered_df[filtered_df["Misconception"] == misconception]
    if question != "all":
        filtered_df = filtered_df[filtered_df["QuestionId"] == question]

    question_counts = filtered_df["QuestionId"].value_counts()

    fig = px.bar(
        x=question_counts.index,
        y=question_counts.values,
        title="문제별 답변 수",
        color_discrete_sequence=["#3498db"],
    )
    fig.update_layout(height=400, xaxis_title="QuestionId", yaxis_title="답변 수")
    return fig


@callback(
    Output("category-misconception-heatmap", "figure"),
    [
        Input("category-filter", "value"),
        Input("misconception-filter", "value"),
        Input("question-filter", "value"),
    ],
)
def update_heatmap(category, misconception, question):
    filtered_df = train_df_with_misconception.copy()

    if category != "all":
        filtered_df = filtered_df[filtered_df["Category"] == category]
    if misconception != "all":
        filtered_df = filtered_df[filtered_df["Misconception"] == misconception]
    if question != "all":
        filtered_df = filtered_df[filtered_df["QuestionId"] == question]

    # Category와 Misconception 교차표
    cross_tab = pd.crosstab(filtered_df["Category"], filtered_df["Misconception"])

    fig = px.imshow(
        cross_tab,
        title="Category vs Misconception 히트맵",
        color_continuous_scale="Blues",
    )
    fig.update_layout(height=400)
    return fig


@callback(
    Output("sample-data-table", "children"),
    [
        Input("category-filter", "value"),
        Input("misconception-filter", "value"),
        Input("question-filter", "value"),
    ],
)
def update_sample_table(category, misconception, question):
    filtered_df = train_df.copy()

    if category != "all":
        filtered_df = filtered_df[filtered_df["Category"] == category]
    if misconception != "all":
        filtered_df = filtered_df[filtered_df["Misconception"] == misconception]
    if question != "all":
        filtered_df = filtered_df[filtered_df["QuestionId"] == question]

    # 샘플 데이터 (처음 10개)
    sample_data = filtered_df.head(10)[
        ["QuestionId", "Category", "Misconception", "StudentExplanation"]
    ]

    return html.Table(
        [
            html.Thead(html.Tr([html.Th(col) for col in sample_data.columns])),
            html.Tbody(
                [
                    html.Tr([html.Td(str(cell)) for cell in row])
                    for row in sample_data.values
                ]
            ),
        ],
        style={
            "width": "100%",
            "border": "1px solid #ddd",
            "borderCollapse": "collapse",
        },
    )


if __name__ == "__main__":
    app.run(debug=True, port=8050)
