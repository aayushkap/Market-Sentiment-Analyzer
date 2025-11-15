import streamlit as st
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



st.set_page_config(page_title="Gold Sentiment Analyzer", layout="wide")
st.title("Gold Sentiment Analyzer")


def add_line_chart(daily_actual, daily_predicted, smooth):
    """
    function to add sentiment line chart over time
    """
    num_days = len(daily_actual)
    if smooth:
        if num_days > 480:
            agg_level = "M"
            smoothing_window = 4
        elif num_days > 180:
            agg_level = "W"
            smoothing_window = 4
        else:
            agg_level = "D"
            smoothing_window = 7
    else:
        agg_level = None
        smoothing_window = None

    if agg_level in ["W", "M"]:
        daily_actual_plot = (
            daily_actual.set_index("Date").resample(agg_level).mean().reset_index()
        )
        daily_predicted_plot = (
            daily_predicted.set_index("Date").resample(agg_level).mean().reset_index()
        )
    elif smoothing_window:
        daily_actual_plot = daily_actual.copy()
        daily_predicted_plot = daily_predicted.copy()
        daily_actual_plot["avg_sentiment"] = (
            daily_actual_plot["avg_sentiment"]
            .rolling(window=smoothing_window, center=True, min_periods=1)
            .mean()
        )
        daily_predicted_plot["avg_sentiment"] = (
            daily_predicted_plot["avg_sentiment"]
            .rolling(window=smoothing_window, center=True, min_periods=1)
            .mean()
        )
    else:
        daily_actual_plot = daily_actual
        daily_predicted_plot = daily_predicted

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily_actual_plot["Date"],
            y=daily_actual_plot["avg_sentiment"],
            mode="lines",
            name="Actual",
            line=dict(color="#2E86AB", width=4),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=daily_predicted_plot["Date"],
            y=daily_predicted_plot["avg_sentiment"],
            mode="lines",
            name="Predicted",
            line=dict(color="rgba(255,0,0,0.75)", width=1),
        )
    )

    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color="gray",
        annotation_text="Neutral",
        annotation_position="right",
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Average Sentiment Score",
        hovermode="x unified",
        template="plotly_white",
        height=500,
        yaxis=dict(range=[-1.1, 1.1]),
    )

    st.plotly_chart(fig, use_container_width=True)


def add_pie_charts(filtered_df):
    """
    pie charts
    """
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Actual sentiment distribution")
        actual_counts = filtered_df["Price Sentiment"].value_counts()
        fig_pie_actual = go.Figure(
            data=[
                go.Pie(
                    labels=actual_counts.index,
                    values=actual_counts.values,
                    marker=dict(colors=["#EF476F", "#FFD166", "#06D6A0"]),
                    hole=0.4,
                )
            ]
        )
        fig_pie_actual.update_layout(height=350)
        st.plotly_chart(fig_pie_actual, use_container_width=True)

    with col2:
        st.subheader("Predicted sentiment distribution")
        predicted_counts = filtered_df["predicted_sentiment"].value_counts()
        fig_pie_pred = go.Figure(
            data=[
                go.Pie(
                    labels=predicted_counts.index,
                    values=predicted_counts.values,
                    marker=dict(colors=["#EF476F", "#FFD166", "#06D6A0"]),
                    hole=0.4,
                )
            ]
        )
        fig_pie_pred.update_layout(height=350)
        st.plotly_chart(fig_pie_pred, use_container_width=True)


def add_results_table(filtered_df):
    """
    results table
    """
    st.subheader("Row wise prediction details")
    display_df = filtered_df[
        [
            "Dates",
            "News",
            "Price Sentiment",
            "predicted_sentiment",
            "confidence",
            "match",
        ]
    ].sort_values("Dates", ascending=False)

    def highlight_match(row):
        if row["match"]:
            return ["background-color: #06D6A0; color: black;"] * len(row)
        else:
            return ["background-color: #EF476F"] * len(row)

    styled_df = display_df.style.apply(highlight_match, axis=1).format(
        {"confidence": "{:.2%}", "Dates": lambda x: x.strftime("%Y-%m-%d")}
    )

    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
        height=400,
        column_config={"match": None},
    )


def add_confusion_matrix(filtered_df, labels):
    """
    accordian cm
    """
    with st.expander("Confusion matrix"):
        cm = confusion_matrix(
            filtered_df["Price Sentiment"],
            filtered_df["predicted_sentiment"],
            labels=labels,
        )

        fig_cm = ff.create_annotated_heatmap(
            z=cm, x=labels, y=labels, colorscale="Blues", showscale=True
        )

        fig_cm.update_layout(
            title="Actual vs predicted sentiment",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=400,
        )

        st.plotly_chart(fig_cm, use_container_width=True)


with st.spinner("Loading models & heavy resources..."):
    
    from data_analyzer import CommodityDataAnalyzer
    from sentiment_classifier import SentimentClassifier

    @st.cache_resource
    def load_data():
        return CommodityDataAnalyzer()

    @st.cache_resource
    def load_classifier():
        return SentimentClassifier(model_dir="model")


analyzer = load_data()
classifier = load_classifier()

st.sidebar.header("Date Range Selection")

stats = analyzer.get_date_range_stats()
min_date = stats["earliest_date"]
max_date = stats["latest_date"]

start_date = st.sidebar.date_input(
    "Start Date",
    value=max_date - timedelta(days=365),
    min_value=min_date,
    max_value=max_date,
)

end_date = st.sidebar.date_input(
    "End Date", value=max_date, min_value=min_date, max_value=max_date
)

if st.sidebar.button("Analyze", type="primary"):
    if start_date > end_date:
        st.error("Start date must be before end date")
    else:
        filtered_df = analyzer.filter_by_date_range(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
        )

        if len(filtered_df) == 0:
            st.warning("No data available for selected date range")
        else:
            with st.spinner("Running sentiment predictions..."):
                headlines = filtered_df["News"].tolist()
                predictions = classifier.predict_batch(headlines)

                filtered_df["predicted_sentiment"] = [
                    p["sentiment"] for p in predictions
                ]
                filtered_df["confidence"] = [p["confidence"] for p in predictions]
                filtered_df["match"] = (
                    filtered_df["Price Sentiment"] == filtered_df["predicted_sentiment"]
                )

            st.session_state.filtered_df = filtered_df
            st.session_state.has_results = True

if st.session_state.get("has_results", False):
    filtered_df = st.session_state.filtered_df

    accuracy = accuracy_score(
        filtered_df["Price Sentiment"], filtered_df["predicted_sentiment"]
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Headlines", len(filtered_df))
    with col2:
        st.metric("Model Accuracy", f"{accuracy*100:.1f}%")
    with col3:
        st.metric("Correct Predictions", filtered_df["match"].sum())
    with col4:
        st.metric("Avg Confidence", f"{filtered_df['confidence'].mean()*100:.1f}%")

    labels = ["negative", "neutral", "positive"]
    sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
    filtered_df["actual_score"] = filtered_df["Price Sentiment"].map(sentiment_map)
    filtered_df["predicted_score"] = filtered_df["predicted_sentiment"].map(
        sentiment_map
    )

    daily_actual = filtered_df.groupby("Dates")["actual_score"].mean().reset_index()
    daily_actual.columns = ["Date", "avg_sentiment"]

    daily_predicted = (
        filtered_df.groupby("Dates")["predicted_score"].mean().reset_index()
    )
    daily_predicted.columns = ["Date", "avg_sentiment"]

    st.subheader("Actual vs predicted trend") 
    smooth = st.toggle("Show averages", value=False)
    add_line_chart(daily_actual, daily_predicted, smooth)

    add_pie_charts(filtered_df)
    add_results_table(filtered_df)
    add_confusion_matrix(filtered_df, labels)

else:
    st.info(
        "Select a date range and press 'Analyze' to view the sentiments for that time range"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dataset Range", f"{min_date} to {max_date}")
    with col2:
        st.metric("Total Records", stats["total_rows"])
    with col3:
        st.metric("Days Covered", stats["date_span_days"])
