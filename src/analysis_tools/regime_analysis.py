import argparse
import json
import math
import os
import re
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from matplotlib import font_manager, rcParams


DATE_FMT = "%Y-%m-%d"
COMPACT_DATE_FMT = "%Y%m%d"
STOPWORDS = set()


def _read_text_fallback(path: str) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _read_json_fallback(path: str) -> dict:
    raw = _read_text_fallback(path)
    return json.loads(raw)


def _read_csv_fallback(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="utf-8", errors="replace")


def _parse_date_str(s: str) -> datetime:
    s = str(s).strip()
    if len(s) == 8:
        return datetime.strptime(s, COMPACT_DATE_FMT)
    return datetime.strptime(s, DATE_FMT)


def _daterange(start: datetime, end: datetime):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _tokenize_ko(text: str):
    if not isinstance(text, str):
        return []
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+|[가-힣]+", text)
    cleaned = []
    for t in tokens:
        if len(t) < 2:
            continue
        if t.isdigit():
            continue
        if t in STOPWORDS:
            continue
        cleaned.append(t)
    return cleaned


def load_index_and_items(data_dir: str, stock_code: str) -> pd.DataFrame:
    index_path = os.path.join(data_dir, "news", stock_code, "index.csv")
    items_dir = os.path.join(data_dir, "news", stock_code, "items")

    index_df = _read_csv_fallback(index_path)
    index_df["content_id"] = index_df["content_id"].astype(str)
    index_df["published_dt"] = index_df["published_dt"].astype(str)
    index_df["published_date"] = index_df["published_dt"].apply(_parse_date_str)

    items = []
    for cid in index_df["content_id"].unique().tolist():
        item_path = os.path.join(items_dir, f"{cid}.json")
        if not os.path.exists(item_path):
            continue
        try:
            item = _read_json_fallback(item_path)
            item["content_id"] = str(item.get("cntt_usiq_srno", cid))
            items.append(item)
        except Exception:
            continue

    if items:
        items_df = pd.DataFrame(items)
        items_df["content_id"] = items_df["content_id"].astype(str)
        merged = index_df.merge(items_df, on="content_id", how="left")
    else:
        merged = index_df.copy()

    merged["title"] = merged["title"].fillna("")
    if "hts_pbnt_titl_cntt" in merged.columns:
        merged["title"] = merged["hts_pbnt_titl_cntt"].fillna("") + " " + merged["title"]
    merged["text"] = merged["title"].str.strip()

    return merged


def apply_regime_labels(df: pd.DataFrame) -> pd.DataFrame:
    bull_ranges = [
        ("2025-03-04", "2025-03-25"),
        ("2025-05-23", "2025-06-25"),
        ("2025-09-29", "2025-11-03"),
        ("2025-12-02", "2025-12-08"),
        ("2025-12-18", "2026-01-22"),
    ]
    bear_ranges = [
        ("2025-01-13", "2025-03-04"),
        ("2025-03-25", "2025-04-11"),
        ("2025-11-03", "2025-12-01"),
        ("2025-12-08", "2025-12-18"),
    ]
    turning_points = [
        "2025-04-11",
        "2025-11-03",
        "2025-12-01",
        "2025-12-08",
        "2025-12-18",
    ]

    bull_dates = set()
    for s, e in bull_ranges:
        for d in _daterange(_parse_date_str(s), _parse_date_str(e)):
            bull_dates.add(d.date())

    bear_dates = set()
    for s, e in bear_ranges:
        for d in _daterange(_parse_date_str(s), _parse_date_str(e)):
            bear_dates.add(d.date())

    tp_dates = { _parse_date_str(d).date() for d in turning_points }

    def _label(dt: datetime):
        if dt.date() in tp_dates:
            return "turning"
        if dt.date() in bull_dates:
            return "bull"
        if dt.date() in bear_dates:
            return "bear"
        return "other"

    df = df.copy()
    df["regime"] = df["published_date"].apply(_label)
    return df


def load_sentiment_lexicon(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    lex = _read_csv_fallback(path)
    lex = lex.dropna(subset=["word", "score"])
    return {str(r["word"]).strip(): float(r["score"]) for _, r in lex.iterrows()}


def load_stopwords(path: str) -> set:
    if not os.path.exists(path):
        return set()
    text = _read_text_fallback(path)
    words = []
    for line in text.splitlines():
        w = line.strip()
        if not w or w.startswith("#"):
            continue
        words.append(w)
    return set(words)


def setup_korean_font():
    candidates = [
        "Malgun Gothic",
        "AppleGothic",
        "NanumGothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
    ]
    found = None
    for name in candidates:
        try:
            path = font_manager.findfont(name, fallback_to_default=False)
            if path:
                found = name
                break
        except Exception:
            continue

    if found:
        rcParams["font.family"] = found
    rcParams["axes.unicode_minus"] = False


def compute_sentiment_scores(texts: pd.Series, lexicon: dict) -> pd.Series:
    if not lexicon:
        return pd.Series([math.nan] * len(texts), index=texts.index)
    scores = []
    for t in texts:
        toks = _tokenize_ko(t)
        if not toks:
            scores.append(0.0)
            continue
        s = sum(lexicon.get(tok, 0.0) for tok in toks)
        scores.append(s / max(len(toks), 1))
    return pd.Series(scores, index=texts.index)


def top_terms_by_regime(df: pd.DataFrame, min_df: int = 5, top_k: int = 20) -> dict:
    results = {}
    for regime, g in df.groupby("regime"):
        vec = TfidfVectorizer(tokenizer=_tokenize_ko, min_df=min_df, max_df=0.9)
        X = vec.fit_transform(g["text"])
        scores = X.mean(axis=0).A1
        top_idx = scores.argsort()[::-1][:top_k]
        terms = [vec.get_feature_names_out()[i] for i in top_idx]
        results[regime] = terms
    return results


def top_ngrams_by_regime(df: pd.DataFrame, min_df: int = 5, top_k: int = 20) -> dict:
    results = {}
    for regime, g in df.groupby("regime"):
        vec = TfidfVectorizer(
            tokenizer=_tokenize_ko,
            min_df=min_df,
            max_df=0.9,
            ngram_range=(1, 2),
        )
        X = vec.fit_transform(g["text"])
        scores = X.mean(axis=0).A1
        top_idx = scores.argsort()[::-1][:top_k]
        terms = [vec.get_feature_names_out()[i] for i in top_idx]
        results[regime] = terms
    return results


def distinctive_terms_log_odds(df: pd.DataFrame, min_df: int = 5, top_k: int = 20) -> dict:
    vec = CountVectorizer(tokenizer=_tokenize_ko, min_df=min_df, max_df=0.9)
    X = vec.fit_transform(df["text"])
    vocab = vec.get_feature_names_out()
    counts = pd.DataFrame(X.toarray(), columns=vocab)
    counts["regime"] = df["regime"].values

    results = {}
    for regime, g in counts.groupby("regime"):
        other = counts[counts["regime"] != regime]
        reg_counts = g.drop(columns=["regime"]).sum(axis=0)
        other_counts = other.drop(columns=["regime"]).sum(axis=0)

        alpha = 0.01
        reg_total = reg_counts.sum() + alpha * len(vocab)
        other_total = other_counts.sum() + alpha * len(vocab)

        reg_probs = (reg_counts + alpha) / reg_total
        other_probs = (other_counts + alpha) / other_total
        log_odds = (reg_probs / (1 - reg_probs)).apply(math.log) - (
            other_probs / (1 - other_probs)
        ).apply(math.log)

        top_terms = log_odds.sort_values(ascending=False).head(top_k).index.tolist()
        results[regime] = top_terms
    return results


def run_topic_model(df: pd.DataFrame, n_topics: int = 5, min_df: int = 5):
    vec = CountVectorizer(tokenizer=_tokenize_ko, min_df=min_df, max_df=0.9)
    X = vec.fit_transform(df["text"])
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)

    topics = []
    feature_names = vec.get_feature_names_out()
    for t_idx, comp in enumerate(lda.components_):
        top_idx = comp.argsort()[::-1][:10]
        topics.append((t_idx, [feature_names[i] for i in top_idx]))
    return topics


def plot_wordclouds(df: pd.DataFrame, out_dir: str):
    try:
        from wordcloud import WordCloud
    except Exception:
        print("wordcloud package not available, skip wordclouds")
        return

    os.makedirs(out_dir, exist_ok=True)
    for regime, g in df.groupby("regime"):
        text = " ".join(g["text"].astype(str).tolist())
        wc = WordCloud(width=900, height=600, background_color="white")
        img = wc.generate(text)
        out_path = os.path.join(out_dir, f"wordcloud_{regime}.png")
        img.to_file(out_path)


def plot_sentiment_distribution(df: pd.DataFrame, out_dir: str):
    if df["sentiment"].isna().all():
        print("sentiment scores missing, skip sentiment plots")
        return
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8, 4))
    sns.kdeplot(data=df, x="sentiment", hue="regime", fill=True, common_norm=False)
    plt.title("Sentiment distribution by regime")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sentiment_distribution.png"))
    plt.close()

    plt.figure(figsize=(10, 4))
    daily = df.groupby(["published_date", "regime"])["sentiment"].mean().reset_index()
    sns.lineplot(data=daily, x="published_date", y="sentiment", hue="regime")
    plt.title("Daily mean sentiment")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sentiment_timeseries.png"))
    plt.close()


def plot_news_volume(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    daily = df.groupby(["published_date", "regime"]).size().reset_index(name="count")
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=daily, x="published_date", y="count", hue="regime")
    plt.title("Daily news volume")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "news_volume.png"))
    plt.close()


def train_regime_classifier(df: pd.DataFrame):
    df = df[df["regime"].isin(["bull", "bear", "turning"])].copy()
    df = df[df["text"].str.len() > 0]

    X = df["text"]
    y = df["regime"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vec = TfidfVectorizer(tokenizer=_tokenize_ko, min_df=3, max_df=0.9)
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)

    clf = LogisticRegression(max_iter=200, class_weight="balanced")
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)

    report = classification_report(y_test, y_pred, output_dict=False)
    cm = confusion_matrix(y_test, y_pred)
    return report, cm, clf, vec


def add_turning_window_label(df: pd.DataFrame, days_before: int = 5):
    turning_dates = [
        _parse_date_str("2025-04-11"),
        _parse_date_str("2025-11-03"),
        _parse_date_str("2025-12-01"),
        _parse_date_str("2025-12-08"),
        _parse_date_str("2025-12-18"),
    ]
    turning_set = {d.date() for d in turning_dates}

    def _label_turning(dt: datetime):
        if dt.date() in turning_set:
            return 1
        for t in turning_dates:
            if 0 < (t.date() - dt.date()).days <= days_before:
                return 1
        return 0

    df = df.copy()
    df["turning_window"] = df["published_date"].apply(_label_turning)
    return df


def train_turning_classifier(df: pd.DataFrame):
    df = df[df["text"].str.len() > 0].copy()
    X = df["text"]
    y = df["turning_window"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vec = TfidfVectorizer(tokenizer=_tokenize_ko, min_df=3, max_df=0.9)
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)

    clf = LogisticRegression(max_iter=200, class_weight="balanced")
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)

    report = classification_report(y_test, y_pred, output_dict=False)
    cm = confusion_matrix(y_test, y_pred)
    return report, cm, clf, vec


def add_low_window_label(df: pd.DataFrame, days_before: int = 5, days_after: int = 2):
    low_dates = [
        _parse_date_str("2025-04-11"),
        _parse_date_str("2025-12-01"),
        _parse_date_str("2025-12-18"),
    ]
    low_set = {d.date() for d in low_dates}

    def _label_low(dt: datetime):
        if dt.date() in low_set:
            return 1
        for t in low_dates:
            delta = (dt.date() - t.date()).days
            if -days_before <= delta <= days_after:
                return 1
        return 0

    df = df.copy()
    df["low_window"] = df["published_date"].apply(_label_low)
    return df


def low_window_distinctive_terms(df: pd.DataFrame, min_df: int = 3, top_k: int = 30):
    vec = CountVectorizer(tokenizer=_tokenize_ko, min_df=min_df, max_df=0.7)
    X = vec.fit_transform(df["text"])
    vocab = vec.get_feature_names_out()
    counts = pd.DataFrame(X.toarray(), columns=vocab)
    counts["low_window"] = df["low_window"].values

    low = counts[counts["low_window"] == 1]
    other = counts[counts["low_window"] == 0]

    low_counts = low.drop(columns=["low_window"]).sum(axis=0)
    other_counts = other.drop(columns=["low_window"]).sum(axis=0)

    alpha = 0.01
    low_total = low_counts.sum() + alpha * len(vocab)
    other_total = other_counts.sum() + alpha * len(vocab)

    low_probs = (low_counts + alpha) / low_total
    other_probs = (other_counts + alpha) / other_total
    log_odds = (low_probs / (1 - low_probs)).apply(math.log) - (
        other_probs / (1 - other_probs)
    ).apply(math.log)

    return log_odds.sort_values(ascending=False).head(top_k).index.tolist()


def low_window_summary(df: pd.DataFrame) -> dict:
    low_df = df[df["low_window"] == 1].copy()
    other_df = df[df["low_window"] == 0].copy()
    return {
        "low_window_count": int(len(low_df)),
        "other_count": int(len(other_df)),
        "low_window_date_range": [
            str(low_df["published_date"].min().date())
            if len(low_df) else None,
            str(low_df["published_date"].max().date())
            if len(low_df) else None,
        ],
    }


def low_window_top_terms(df: pd.DataFrame, top_k: int = 50) -> dict:
    low_df = df[df["low_window"] == 1].copy()
    if low_df.empty:
        return {}

    all_tokens = []
    for t in low_df["text"].astype(str).tolist():
        all_tokens.extend(_tokenize_ko(t))

    counts = pd.Series(all_tokens).value_counts().head(top_k)
    return {k: int(v) for k, v in counts.items()}


def low_window_source_share(df: pd.DataFrame, top_k: int = 15) -> dict:
    low_df = df[df["low_window"] == 1].copy()
    if low_df.empty:
        return {}

    source_col = "source_name" if "source_name" in low_df.columns else None
    if source_col is None:
        return {}

    counts = low_df[source_col].fillna("UNKNOWN").value_counts()
    total = counts.sum()
    top = counts.head(top_k)

    result = []
    for name, cnt in top.items():
        share = float(cnt) / float(total) if total else 0.0
        result.append({"source": str(name), "count": int(cnt), "share": round(share, 4)})

    return {
        "total_low_window": int(total),
        "top_sources": result,
    }


def main():
    parser = argparse.ArgumentParser(description="Market regime news analysis")
    parser.add_argument("--data-dir", default="data", help="base data dir (default: data)")
    parser.add_argument("--stock", default="005380", help="stock code (default: 005380)")
    parser.add_argument("--out-dir", default="outputs/regime", help="output directory")
    parser.add_argument("--sentiment-lexicon", default="data/sentiment_lexicon.csv")
    parser.add_argument("--stopwords", default="data/stopwords_ko.txt")
    parser.add_argument("--turning-window-days", type=int, default=5)
    parser.add_argument("--low-window-days-before", type=int, default=5)
    parser.add_argument("--low-window-days-after", type=int, default=2)
    parser.add_argument(
        "--run-model",
        action="store_true",
        help="run classification models (default: off)",
    )
    args = parser.parse_args()

    global STOPWORDS
    STOPWORDS = load_stopwords(args.stopwords)
    STOPWORDS.update({"현대차", "현대", "기아", "그룹", "주가", "증시"})

    setup_korean_font()

    df = load_index_and_items(args.data_dir, args.stock)
    df = apply_regime_labels(df)

    lexicon = load_sentiment_lexicon(args.sentiment_lexicon)
    df["sentiment"] = compute_sentiment_scores(df["text"], lexicon)

    os.makedirs(args.out_dir, exist_ok=True)

    top_terms = top_terms_by_regime(df)
    top_terms_path = os.path.join(args.out_dir, "top_terms_by_regime.json")
    with open(top_terms_path, "w", encoding="utf-8") as f:
        json.dump(top_terms, f, ensure_ascii=False, indent=2)

    top_ngrams = top_ngrams_by_regime(df)
    top_ngrams_path = os.path.join(args.out_dir, "top_ngrams_by_regime.json")
    with open(top_ngrams_path, "w", encoding="utf-8") as f:
        json.dump(top_ngrams, f, ensure_ascii=False, indent=2)

    distinctive_terms = distinctive_terms_log_odds(df)
    distinct_path = os.path.join(args.out_dir, "distinctive_terms_log_odds.json")
    with open(distinct_path, "w", encoding="utf-8") as f:
        json.dump(distinctive_terms, f, ensure_ascii=False, indent=2)

    topics = run_topic_model(df)
    topics_path = os.path.join(args.out_dir, "topics.json")
    with open(topics_path, "w", encoding="utf-8") as f:
        json.dump({"topics": topics}, f, ensure_ascii=False, indent=2)

    plot_wordclouds(df, args.out_dir)
    plot_sentiment_distribution(df, args.out_dir)
    plot_news_volume(df, args.out_dir)

    df_low = add_low_window_label(
        df,
        days_before=args.low_window_days_before,
        days_after=args.low_window_days_after,
    )
    low_terms = low_window_distinctive_terms(df_low)
    low_terms_path = os.path.join(args.out_dir, "low_window_distinctive_terms.json")
    with open(low_terms_path, "w", encoding="utf-8") as f:
        json.dump({"low_window_terms": low_terms}, f, ensure_ascii=False, indent=2)

    low_top_terms = low_window_top_terms(df_low)
    low_top_terms_path = os.path.join(args.out_dir, "low_window_top_terms.json")
    with open(low_top_terms_path, "w", encoding="utf-8") as f:
        json.dump({"low_window_top_terms": low_top_terms}, f, ensure_ascii=False, indent=2)

    low_sources = low_window_source_share(df_low)
    low_sources_path = os.path.join(args.out_dir, "low_window_source_share.json")
    with open(low_sources_path, "w", encoding="utf-8") as f:
        json.dump(low_sources, f, ensure_ascii=False, indent=2)

    low_summary = low_window_summary(df_low)
    low_summary_path = os.path.join(args.out_dir, "low_window_summary.json")
    with open(low_summary_path, "w", encoding="utf-8") as f:
        json.dump(low_summary, f, ensure_ascii=False, indent=2)

    if args.run_model:
        report, cm, _, _ = train_regime_classifier(df)
        with open(
            os.path.join(args.out_dir, "regime_report.txt"), "w", encoding="utf-8"
        ) as f:
            f.write(report)
            f.write("\nConfusion matrix:\n")
            f.write(str(cm))

        df_turn = add_turning_window_label(df, days_before=args.turning_window_days)
        t_report, t_cm, _, _ = train_turning_classifier(df_turn)
        with open(
            os.path.join(args.out_dir, "turning_report.txt"), "w", encoding="utf-8"
        ) as f:
            f.write(t_report)
            f.write("\nConfusion matrix:\n")
            f.write(str(t_cm))

    df.to_csv(
        os.path.join(args.out_dir, "labeled_news.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Saved outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
