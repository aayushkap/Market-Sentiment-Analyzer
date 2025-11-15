from datasets import load_dataset
import pandas as pd
from datetime import datetime, date
from typing import Optional, Tuple
from pathlib import Path
import os


class CommodityDataAnalyzer:
    def __init__(
        self,
        dataset_name: str = "SaguaroCapital/sentiment-analysis-in-commodity-market-gold",
        save_dir: str = "data/commodity_data.csv",
    ):
        self.dataset_name = dataset_name
        self.save_dir = Path(save_dir)
        self.df = None

        self.save_dir.parent.mkdir(parents=True, exist_ok=True)
        self.load_dataset()

    def load_dataset(self):
        if self.save_dir.exists():
            self.df = pd.read_csv(self.save_dir)
            self.df["Dates"] = pd.to_datetime(self.df["Dates"])
        else:
            dataset = load_dataset(self.dataset_name)

            train_df = pd.DataFrame(dataset["train"])
            test_df = pd.DataFrame(dataset["test"])
            self.df = pd.concat([train_df, test_df], ignore_index=True)

            self.df["Dates"] = pd.to_datetime(
                self.df["Dates"], format="mixed", errors="coerce"
            )

            self.df = self.df.dropna(subset=["Dates", "Price Sentiment"])
            self.df = self.df[self.df["Price Sentiment"] != "none"]
            self.df.to_csv(self.save_dir, index=False)

            return self.df

    def filter_by_date_range(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ):
        """
        filter dataset by date range
        """
        if self.df is None:
            raise ValueError("No data loaded available.")

        # Handle default dates
        if start_date is None:
            start_date = self.df["Dates"].min()
        else:
            start_date = pd.to_datetime(start_date)

        if end_date is None:
            end_date = pd.Timestamp(datetime.now().date())
        else:
            end_date = pd.to_datetime(end_date)

        filtered_df = self.df[
            (self.df["Dates"] >= start_date) & (self.df["Dates"] <= end_date)
        ].copy()

        return filtered_df

    def get_date_range_stats(self) -> dict:
        """
        basic statistics about the date range in the dataset
        """
        if self.df is None:
            raise ValueError("No data loaded available.")

        return {
            "earliest_date": self.df["Dates"].min().date(),
            "latest_date": self.df["Dates"].max().date(),
            "total_rows": len(self.df),
            "date_span_days": (self.df["Dates"].max() - self.df["Dates"].min()).days,
        }

    def get_sentiment_distribution(self, df: Optional[pd.DataFrame] = None) -> dict:
        """
        sentiment distribution from the dataset
        """
        if df is None:
            df = self.df

        if df is None:
            raise ValueError("No data loaded available.")

        return df["Price Sentiment"].value_counts().to_dict()

    def __repr__(self):
        rows = len(self.df) if self.df is not None else 0
        return f"CommodityDataAnalyzer(dataset='{self.dataset_name}', rows={rows})"
