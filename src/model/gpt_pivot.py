# make_updown_prob_dashboard.py
# 목적
# - 5/10/20일 forward return을 변동성 기반 동적 임계값으로 Up/Down 이벤트 라벨링(이진 2개)
# - 종목별 time split + embargo(H) 적용
# - Up/Down 확률을 pred_with_probs.csv + model_report.txt + HTML 대시보드로 저장
#
# 실행
#   python make_updown_prob_dashboard.py
#
# 입력/출력 경로(요청대로 data/ 아래)
#   data/features_lgbm.csv
#   data/pred_with_probs.csv
#   data/model_report.txt
#   data/updown_prob_dashboard.html

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


# =========================
# Config
# =========================
INPUT = Path("data/features_lgbm.csv")

OUT_PRED_CSV = Path("data/pred_with_probs.csv")
OUT_REPORT_TXT = Path("data/model_report.txt")
OUT_HTML = Path("data/updown_prob_dashboard.html")

HORIZONS = [5, 10, 20]
SPLIT_RATIO = 0.7
PROB_THR = 0.8  # 신호 임계값 (p>=0.8)

# 동적 임계값: thr_h = K * vol20 * sqrt(h)
# 이전 결과가 너무 빡셌으니 기본값은 낮게 시작
K_BY_H = {5: 0.6, 10: 0.8, 20: 1.0}

# K 자동 튜닝(라벨 희귀하면 K를 더 낮춰서 pos 샘플을 늘림)
AUTO_TUNE_K = True
K_GRID = [0.25, 0.35, 0.5, 0.6, 0.75, 0.9, 1.0, 1.2, 1.5]

# 목표 pos 비율 범위(Train 기준) - 너무 희귀/너무 흔함을 피함
TARGET_POS_RATE_LO = 0.05
TARGET_POS_RATE_HI = 0.20

MODEL_KWARGS = dict(
    max_depth=3,
    learning_rate=0.06,
    max_iter=450,
    random_state=42,
)


# =========================
# Helpers
# =========================
def pick_col(cols: list[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in cols:
            return c
    lower_map = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def to_datetime_safe(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, errors="coerce")
    if out.isna().mean() > 0.5:
        out = pd.to_datetime(s.astype(str), format="%Y%m%d", errors="coerce")
    return out


def ensure_vol20(df: pd.DataFrame, stock_col: str, close_col: str) -> tuple[pd.DataFrame, str]:
    if "volatility_20" in df.columns:
        return df, "volatility_20"

    # fallback: 20일 수익률 표준편차
    ret1 = df.groupby(stock_col)[close_col].pct_change()
    vol20 = ret1.groupby(df[stock_col]).rolling(20, min_periods=20).std(ddof=0).reset_index(level=0, drop=True)
    df = df.copy()
    df["volatility_20_fallback"] = vol20
    return df, "volatility_20_fallback"


def build_feature_cols(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    banned_kw = ["fwd_ret_", "thr_", "y_up_", "y_down_", "p_up_", "p_down_"]
    feat = []
    for c in num_cols:
        if c in exclude:
            continue
        lc = c.lower()
        if any(k in lc for k in banned_kw):
            continue
        feat.append(c)
    return feat


def per_stock_split_indices(data: pd.DataFrame, stock_col: str, horizon: int, split_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """
    data: (stock,date) 정렬된 단일 데이터프레임
    종목별로 split 후 embargo(H) 적용
      train: [0 : cut-h]
      test : [cut+h : n]
    """
    tr_all = []
    te_all = []
    for _, g in data.groupby(stock_col, sort=False):
        idx = g.index.to_numpy()
        n = len(idx)
        if n < 250:
            continue
        cut = int(n * split_ratio)
        tr_end = max(0, cut - horizon)
        te_start = min(n, cut + horizon)
        if tr_end < 100 or (n - te_start) < 60:
            continue
        tr_all.append(idx[:tr_end])
        te_all.append(idx[te_start:])
    if not tr_all or not te_all:
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.concatenate(tr_all), np.concatenate(te_all)


def precision_at_threshold(y_true: np.ndarray, p: np.ndarray, thr: float) -> tuple[float, int]:
    m = p >= thr
    if m.sum() == 0:
        return float("nan"), 0
    return float((y_true[m] == 1).mean()), int(m.sum())


def choose_k_for_h(
    base: pd.DataFrame,
    stock_col: str,
    close_col: str,
    vol_col: str,
    h: int,
    k_default: float,
) -> float:
    # Train 구간에서 pos 비율이 목표 범위에 들어오도록 K 선택
    # base는 원본 df (정렬됨)
    parts = []
    for _, g in base.groupby(stock_col, sort=False):
        gg = g.sort_values("___date").copy()
        n = len(gg)
        if n < 250:
            continue
        cut = int(n * SPLIT_RATIO)
        tr_end = max(0, cut - h)
        if tr_end < 120:
            continue
        parts.append(gg.iloc[:tr_end])
    if not parts:
        return k_default

    train = pd.concat(parts, axis=0, ignore_index=True)
    c = train[close_col].astype(float)
    fwd = train.groupby(stock_col)[close_col].shift(-h) / c - 1.0
    sigma_h = train[vol_col] * np.sqrt(h)

    best = None
    for k in K_GRID:
        thr_h = k * sigma_h
        y_up = (fwd >= thr_h) & fwd.notna() & thr_h.notna()
        y_dn = (fwd <= -thr_h) & fwd.notna() & thr_h.notna()

        up_rate = float(y_up.mean())
        dn_rate = float(y_dn.mean())

        # Up/Down 둘 다 목표 범위에 근접하면 가산점
        score = 0.0
        for r in [up_rate, dn_rate]:
            if TARGET_POS_RATE_LO <= r <= TARGET_POS_RATE_HI:
                score += 2.0
            # 너무 희귀하면 큰 패널티
            score -= abs(r - 0.10)

        if best is None or score > best[0]:
            best = (score, k, up_rate, dn_rate)

    if best is None:
        return k_default

    return float(best[1])


# =========================
# Main
# =========================
def main():
    df = pd.read_csv(INPUT)
    cols = df.columns.tolist()

    stock_col = pick_col(cols, ["stock_code", "종목코드"])
    date_col = pick_col(cols, ["date", "날짜"])
    close_col = pick_col(cols, ["종가", "close"])
    if stock_col is None or date_col is None or close_col is None:
        raise ValueError(f"필수 컬럼 누락 stock={stock_col}, date={date_col}, close={close_col}")

    df[date_col] = to_datetime_safe(df[date_col])
    df = df.dropna(subset=[stock_col, date_col, close_col]).copy()
    df = df.sort_values([stock_col, date_col]).reset_index(drop=True)

    # 내부 정렬용 컬럼 (가독성용)
    df = df.rename(columns={date_col: "___date"})
    date_col = "___date"

    # vol20 확보
    df, vol_col = ensure_vol20(df, stock_col=stock_col, close_col=close_col)

    # features
    exclude = {close_col, "최고가", "최저가", "종목코드", "날짜", "stock_code", "date", "___date"}
    feat_cols = build_feature_cols(df, exclude=exclude)
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan)

    # pred base
    pred = df[[stock_col, date_col, close_col]].copy()

    report = []
    report.append("=== Up/Down Binary Models (vol-based thresholds) ===")
    report.append(f"INPUT: {INPUT}")
    report.append(f"split: per-stock {SPLIT_RATIO}, embargo=H")
    report.append(f"prob_thr: {PROB_THR}")
    report.append(f"vol_col: {vol_col}")
    report.append("")

    # horizon loop
    for h in HORIZONS:
        k = K_BY_H.get(h, 1.0)
        if AUTO_TUNE_K:
            k = choose_k_for_h(df, stock_col, close_col, vol_col, h, k)

        # compute forward ret + threshold (전체)
        c = df[close_col].astype(float)
        fwd_ret = df.groupby(stock_col)[close_col].shift(-h) / c - 1.0
        sigma_h = df[vol_col] * np.sqrt(h)
        thr_h = k * sigma_h

        # labels (binary)
        y_up = np.where((fwd_ret.notna()) & (thr_h.notna()) & (fwd_ret >= thr_h), 1, 0).astype("float")
        y_dn = np.where((fwd_ret.notna()) & (thr_h.notna()) & (fwd_ret <= -thr_h), 1, 0).astype("float")

        # invalid tail -> NaN (미래 없어서 라벨 불가)
        y_up[(fwd_ret.isna()) | (thr_h.isna())] = np.nan
        y_dn[(fwd_ret.isna()) | (thr_h.isna())] = np.nan

        df[f"fwd_ret_{h}"] = fwd_ret
        df[f"thr_{h}"] = thr_h
        df[f"y_up_{h}"] = y_up
        df[f"y_down_{h}"] = y_dn

        # dataset for this horizon (라벨 있는 행)
        base_mask = df[f"y_up_{h}"].notna() & df[feat_cols].notna().any(axis=1)
        data_h = df.loc[
            base_mask,
            [stock_col, date_col, close_col, f"thr_{h}", *feat_cols, f"y_up_{h}", f"y_down_{h}"]
        ].copy()
        data_h = data_h.sort_values([stock_col, date_col]).reset_index(drop=True)

        tr_idx, te_idx = per_stock_split_indices(data_h, stock_col, horizon=h, split_ratio=SPLIT_RATIO)
        if tr_idx.size == 0 or te_idx.size == 0:
            report.append(f"===== Horizon {h}d =====")
            report.append("no valid train/test split")
            report.append("")
            continue

        # fill NaNs by train median
        train_med = data_h.loc[tr_idx, feat_cols].median(numeric_only=True)
        X = data_h[feat_cols].fillna(train_med).to_numpy(float)

        # ---------- UP model ----------
        y_up_arr = data_h[f"y_up_{h}"].astype(int).to_numpy()
        if len(np.unique(y_up_arr[tr_idx])) < 2:
            p_up = np.full(len(te_idx), np.nan)
            report_up = "UP: train has single class"
        else:
            clf_up = HistGradientBoostingClassifier(**MODEL_KWARGS)
            clf_up.fit(X[tr_idx], y_up_arr[tr_idx])
            p_up = clf_up.predict_proba(X[te_idx])[:, 1]
            report_up = None

        # ---------- DOWN model ----------
        y_dn_arr = data_h[f"y_down_{h}"].astype(int).to_numpy()
        if len(np.unique(y_dn_arr[tr_idx])) < 2:
            p_dn = np.full(len(te_idx), np.nan)
            report_dn = "DOWN: train has single class"
        else:
            clf_dn = HistGradientBoostingClassifier(**MODEL_KWARGS)
            clf_dn.fit(X[tr_idx], y_dn_arr[tr_idx])
            p_dn = clf_dn.predict_proba(X[te_idx])[:, 1]
            report_dn = None

        # write predictions back to pred via merge on (stock,date)
        out = data_h.loc[te_idx, [stock_col, date_col]].copy()
        out[f"p_up_{h}"] = p_up
        out[f"p_down_{h}"] = p_dn
        out[f"thr_{h}"] = data_h.loc[te_idx, f"thr_{h}"].to_numpy()
        out[f"fwd_ret_{h}"] = data_h.loc[te_idx, close_col].to_numpy(dtype=float)  # placeholder, keep schema stable

        pred = pred.merge(out, on=[stock_col, date_col], how="left")

        # report metrics on TEST
        report.append(f"===== Horizon {h}d =====")
        report.append(f"threshold: thr_h = {k:.3f} * {vol_col} * sqrt({h})")
        # label base rates (test)
        up_base = float(y_up_arr[te_idx].mean())
        dn_base = float(y_dn_arr[te_idx].mean())
        report.append(f"base_rate_up(test): {up_base:.4f}")
        report.append(f"base_rate_down(test): {dn_base:.4f}")

        if report_up:
            report.append(report_up)
        else:
            prec_up, cov_up = precision_at_threshold(y_up_arr[te_idx], p_up, PROB_THR)
            try:
                auc_up = roc_auc_score(y_up_arr[te_idx], p_up)
            except Exception:
                auc_up = float("nan")
            report.append(f"UP  precision@{PROB_THR}: {prec_up:.3f} | coverage: {cov_up} | AUC: {auc_up:.3f}")

        if report_dn:
            report.append(report_dn)
        else:
            prec_dn, cov_dn = precision_at_threshold(y_dn_arr[te_idx], p_dn, PROB_THR)
            try:
                auc_dn = roc_auc_score(y_dn_arr[te_idx], p_dn)
            except Exception:
                auc_dn = float("nan")
            report.append(f"DOWN precision@{PROB_THR}: {prec_dn:.3f} | coverage: {cov_dn} | AUC: {auc_dn:.3f}")

        report.append("")

    # save pred/report
    OUT_PRED_CSV.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(OUT_PRED_CSV, index=False, encoding="utf-8-sig")
    OUT_REPORT_TXT.write_text("\n".join(report), encoding="utf-8")

    # =========================
    # Dashboard (Plotly, stock dropdown, one chart per horizon)
    # =========================
    import plotly.graph_objects as go
    from plotly.offline import plot

    stocks = sorted(pred[stock_col].dropna().unique().tolist())
    html_blocks = []

    for h in HORIZONS:
        need = [f"p_up_{h}", f"p_down_{h}"]
        if not all(c in pred.columns for c in need):
            continue

        d = pred.dropna(subset=need).sort_values([stock_col, date_col]).copy()
        if d.empty:
            continue

        # 실제로 존재하는 종목만
        stocks_h = sorted(d[stock_col].unique().tolist())

        fig = go.Figure()
        n_per = 5  # price + p_up + p_down + buy + sell

        for s in stocks_h:
            ds = d[d[stock_col] == s]
            fig.add_trace(go.Scatter(x=ds[date_col], y=ds[close_col], mode="lines", name=f"{s} 종가", yaxis="y1"))
            fig.add_trace(go.Scatter(x=ds[date_col], y=ds[f"p_up_{h}"], mode="lines", name=f"{s} P(Up)", yaxis="y2"))
            fig.add_trace(go.Scatter(x=ds[date_col], y=ds[f"p_down_{h}"], mode="lines", name=f"{s} P(Down)", yaxis="y2"))

            buy = ds[ds[f"p_up_{h}"] >= PROB_THR]
            sell = ds[ds[f"p_down_{h}"] >= PROB_THR]
            fig.add_trace(go.Scatter(x=buy[date_col], y=buy[close_col], mode="markers", name=f"{s} BUY(pUp≥{PROB_THR})", yaxis="y1"))
            fig.add_trace(go.Scatter(x=sell[date_col], y=sell[close_col], mode="markers", name=f"{s} SELL(pDn≥{PROB_THR})", yaxis="y1"))

        masks = []
        for i in range(len(stocks_h)):
            mask = [False] * (len(stocks_h) * n_per)
            for j in range(n_per):
                mask[i * n_per + j] = True
            masks.append(mask)

        if stocks_h:
            for k in range(len(stocks_h) * n_per):
                fig.data[k].visible = masks[0][k]

        buttons = [
            dict(
                label=str(stocks_h[i]),
                method="update",
                args=[
                    {"visible": masks[i]},
                    {"title": f"{stocks_h[i]} | Horizon {h}d | P(Up)/P(Down)"},
                ],
            )
            for i in range(len(stocks_h))
        ]

        fig.update_layout(
            template="simple_white",
            title=f"{stocks_h[0] if stocks_h else ''} | Horizon {h}d | P(Up)/P(Down)",
            hovermode="x unified",
            updatemenus=[dict(buttons=buttons, direction="down", x=1.02, xanchor="left", y=1.0, yanchor="top")],
            margin=dict(l=50, r=240, t=70, b=50),
            yaxis=dict(title="가격", side="left"),
            yaxis2=dict(title="확률", overlaying="y", side="right", range=[0, 1]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )

        html_blocks.append(
            "<div style='border:1px solid #e5e7eb;border-radius:14px;padding:12px;margin-bottom:14px;background:#fff'>"
            + plot(fig, include_plotlyjs="cdn" if not html_blocks else False, output_type="div")
            + "</div>"
        )

    page = "\n".join([
        "<!doctype html><html lang='ko'><head><meta charset='utf-8'/>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'/>",
        "<title>Up/Down 확률 대시보드</title>",
        "<style>",
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Noto Sans KR',Arial;margin:0}",
        "header{padding:16px 20px;border-bottom:1px solid #e5e7eb}",
        ".wrap{padding:16px 20px;max-width:1500px;margin:0 auto}",
        ".hint{color:#6b7280;font-size:13px;line-height:1.4;margin-top:6px}",
        "</style></head><body>",
        "<header><div style='max-width:1500px;margin:0 auto'>",
        "<div style='font-size:18px;font-weight:800'>Up/Down 확률 대시보드</div>",
        f"<div class='hint'>라벨: thr_h = K*vol20*sqrt(h) | K(default)={K_BY_H} | auto_tune_k={AUTO_TUNE_K} | split={SPLIT_RATIO} (종목별) | embargo=H | 신호: pUp/pDown ≥ {PROB_THR}</div>",
        f"<div class='hint'>저장: {OUT_PRED_CSV.name}, {OUT_REPORT_TXT.name}</div>",
        "</div></header>",
        "<div class='wrap'>",
        *html_blocks,
        "</div></body></html>",
    ])

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(page, encoding="utf-8")

    print("saved:", OUT_PRED_CSV)
    print("saved:", OUT_REPORT_TXT)
    print("saved:", OUT_HTML)


if __name__ == "__main__":
    main()