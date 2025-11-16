import streamlit as st
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from config import page_title, use_llm, llm_model_name


st.set_page_config(page_title=page_title, layout="wide")
st.title(page_title)


def add_line_chart(daily_actual, daily_predicted, smooth):
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
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Actual sentiment distribution")
        actual_counts = filtered_df["Price Sentiment"].value_counts()
        fig_pie_actual = go.Figure(
            data=[
                go.Pie(
                    labels=actual_counts.index,
                    values=actual_counts.values,
                    marker=dict(colors=["#06D6A0", "#EF476F", "#FFD166"]),
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
                    marker=dict(colors=["#06D6A0", "#EF476F", "#FFD166"]),
                    hole=0.4,
                )
            ]
        )
        fig_pie_pred.update_layout(height=350)
        st.plotly_chart(fig_pie_pred, use_container_width=True)


def add_results_table(filtered_df):
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


with st.spinner("Loading sentiment classifier..."):
    from data_analyzer import CommodityDataAnalyzer
    from sentiment_classifier import SentimentClassifier

    @st.cache_resource
    def load_data():
        return CommodityDataAnalyzer()

    @st.cache_resource
    def load_classifier():
        return SentimentClassifier()


if use_llm:
    with st.spinner(
        f"Loading LLM {llm_model_name} (will download if not already installed)..."
    ):
        from llm_summarizer import LLMSummarizer

        @st.cache_resource
        def load_llm():
            try:
                return LLMSummarizer()
            except Exception as e:
                st.warning(f"Unable to load LLM: {str(e)}")
                return None


analyzer = load_data()
classifier = load_classifier()
if use_llm:
    llm = load_llm()

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

if st.sidebar.button("Analyze Sentiment", type="primary"):
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
            st.session_state.needs_summary = use_llm and llm
            st.session_state.cached_summary = None


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
    smooth = st.toggle("Group dates", value=False, key="smooth_toggle")
    add_line_chart(daily_actual, daily_predicted, smooth)

    st.subheader("AI-Generated Summary")

    if st.session_state.get("needs_summary"):
        st.session_state.needs_summary = False
        total_headlines = len(filtered_df)
        date_range = f"{filtered_df['Dates'].min():%Y-%m-%d} to {filtered_df['Dates'].max():%Y-%m-%d}"

        sentiment_counts = filtered_df["Price Sentiment"].value_counts()
        positive_pct = (sentiment_counts.get("positive", 0) / total_headlines) * 100
        negative_pct = (sentiment_counts.get("negative", 0) / total_headlines) * 100
        neutral_pct = (sentiment_counts.get("neutral", 0) / total_headlines) * 100

        filtered_df_sorted = filtered_df.sort_values("Dates").reset_index(drop=True)
        total_rows = len(filtered_df_sorted)

        indices = [int(i * (total_rows - 1) / 5) for i in range(6)]
        samples = []

        for idx in indices:
            row = filtered_df_sorted.iloc[idx]
            date_str = row["Dates"].strftime("%b %Y")
            sentiment = row["Price Sentiment"]
            headline = row["News"][:80]
            samples.append(f"{date_str} ({sentiment}): {headline}")

        sample_text = "\n".join(samples)

        prompt_text = f"""
Summarize gold market sentiment from {date_range}.

Data: {total_headlines} headlines - {positive_pct:.0f}% positive, {negative_pct:.0f}% negative, {neutral_pct:.0f}% neutral

Key headlines:
{sample_text}

Write 2-3 sentences describing the overall trend."""

        summary_placeholder = st.empty()
        full_summary = ""
        sentence_count = 0

        for token in llm.summarize_stream(
            prompt=prompt_text,
            max_tokens=128,
            temperature=0.6,
            repetition_penalty=1.15,
        ):
            full_summary += token
            summary_placeholder.markdown(full_summary + "▌")

        stop_phrases = [
            "You are",
            "I am",
            "As an",
            "This summary",
            "In summary",
            "The summary",
            "Here is",
            "Based on",
        ]

        for phrase in stop_phrases:
            if phrase in full_summary:
                full_summary = full_summary.split(phrase)[0].strip()
                break

        if full_summary and not full_summary[-1] in [".", "!", "?"]:
            last_period = full_summary.rfind(".")
            last_exclamation = full_summary.rfind("!")
            last_question = full_summary.rfind("?")
            last_punct = max(last_period, last_exclamation, last_question)

            if last_punct > 0:
                full_summary = full_summary[: last_punct + 1]
            else:
                full_summary += "."

        summary_placeholder.markdown(full_summary)
        st.session_state.cached_summary = full_summary
        st.session_state.needs_summary = False

    elif st.session_state.get("cached_summary"):
        st.markdown(st.session_state.cached_summary)

    add_pie_charts(filtered_df)
    add_results_table(filtered_df)
    add_confusion_matrix(filtered_df, labels)

else:
    st.info("Select a date range and press 'Analyze' to view the sentiments")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dataset Range", f"{min_date} to {max_date}")
    with col2:
        st.metric("Total Records", stats["total_rows"])
    with col3:
        st.metric("Days Covered", stats["date_span_days"])
