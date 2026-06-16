"""
Unit tests for data loading and preprocessing.
"""

import unittest
import pandas as pd
from src.data.loader import clean_text, prepare_data, split_data, LABEL_MAP


class TestCleanText(unittest.TestCase):
    def test_removes_urls(self):
        result = clean_text("Check this http://bit.ly/abc cool")
        self.assertNotIn("http", result)

    def test_removes_mentions(self):
        result = clean_text("hello @username how are you")
        self.assertNotIn("@username", result)

    def test_lowercases(self):
        result = clean_text("HELLO WORLD")
        self.assertEqual(result, "hello world")

    def test_removes_special_chars(self):
        result = clean_text("hello!!! #awesome?")
        self.assertEqual(result, "hello awesome")

    def test_collapses_spaces(self):
        result = clean_text("hello    world")
        self.assertEqual(result, "hello world")

    def test_empty_string(self):
        self.assertEqual(clean_text(""), "")

    def test_non_string(self):
        self.assertEqual(clean_text(None), "")


class TestPrepareData(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "text": ["Great flight!", "Terrible delay"],
            "airline_sentiment": ["positive", "negative"],
        })

    def test_adds_clean_text(self):
        result = prepare_data(self.df)
        self.assertIn("clean_text", result.columns)

    def test_adds_label_column(self):
        result = prepare_data(self.df)
        self.assertIn("label", result.columns)

    def test_label_mapping(self):
        result = prepare_data(self.df)
        self.assertEqual(result["label"].tolist(), [2, 0])

    def test_drops_invalid_labels(self):
        df = pd.DataFrame({
            "text": ["ok"],
            "airline_sentiment": ["invalid"],
        })
        result = prepare_data(df)
        self.assertEqual(len(result), 0)


class TestSplitData(unittest.TestCase):
    def test_split_ratio(self):
        texts = ["a"] * 100
        labels = [0] * 50 + [1] * 50
        train_t, val_t, train_l, val_l = split_data(
            pd.DataFrame({"clean_text": texts, "label": labels}),
            test_size=0.2,
            random_state=42,
        )
        self.assertEqual(len(train_t), 80)
        self.assertEqual(len(val_t), 20)


if __name__ == "__main__":
    unittest.main()
