"""Streamlit front end: predict, explore thresholds, watch drift."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from client import get, predict_batch, prepare_rows

st.set_page_config(page_title="Fraud Detection", layout="wide")
SAMPLES = pd.read_csv(Path(__file__).parent / "samples.csv")
TEST = SAMPLES[SAMPLES.source == "test"]  # unseen period: honest scoring
LEGIT = TEST[TEST.Class == 0]
TRAIN_LEGIT = SAMPLES[SAMPLES.source == "train"]  # all hours: matches the drift reference
COLORS = {"ok": "#2e7d32", "watch": "#f9a825", "drift": "#c62828"}

try:
    health = get("/health")
except requests.RequestException:
    st.error("API unreachable. Is the api service running?")
    st.stop()

st.title("Credit card fraud detection")
st.caption(f"Model version {health['model_version']}")
policy = st.sidebar.radio(
    "Threshold policy",
    ["cost", "recall", "precision"],
    help="cost: cheapest on validation (FN €120, FP €5). recall/precision: hit ≥ 90% on validation.",
)

predict_tab, threshold_tab, drift_tab = st.tabs(["Predict", "Threshold", "Drift"])

with predict_tab:
    c1, c2 = st.columns(2)
    pick = None
    if c1.button("Random legit transaction"):
        pick = LEGIT.sample(1)
    if c2.button("Random fraud transaction"):
        pick = TEST[TEST.Class == 1].sample(1)
    if pick is not None:
        res = predict_batch(prepare_rows(pick), policy)[0]
        actual = "fraud" if pick.Class.iloc[0] == 1 else "legit"
        st.metric("Fraud probability", f"{res['fraud_probability']:.2%}")
        verdict = "FRAUD" if res["is_fraud"] else "legit"
        show = st.error if res["is_fraud"] else st.success
        show(
            f"Decision: {verdict} (threshold {res['threshold']:.3f}, policy {res['policy']}). "
            f"Actual label: {actual}"
        )
        st.dataframe(pick.drop(columns=["Class", "source"]))

    upload = st.file_uploader("Or score a CSV (columns V1–V28, Amount)", type="csv")
    if upload:
        df = pd.read_csv(upload)
        try:
            rows = prepare_rows(df)
        except ValueError as e:
            st.error(str(e))
        else:
            results = pd.DataFrame(predict_batch(rows, policy))
            st.write(f"{int(results.is_fraud.sum())} of {len(results)} flagged")
            st.dataframe(
                pd.concat(
                    [df.reset_index(drop=True), results[["fraud_probability", "is_fraud"]]], axis=1
                )
            )

with threshold_tab:
    curves = get("/curves")
    sweep = pd.DataFrame(curves["sweep"])
    t = st.slider("Threshold", 0.001, 0.999, 0.05, 0.001, format="%.3f")
    row = sweep.loc[(sweep.threshold - t).abs().idxmin()]
    m = st.columns(4)
    m[0].metric("Frauds caught", f"{int(row.tp)} / {int(row.tp + row.fn)}")
    m[1].metric("False alarms", int(row.fp))
    m[2].metric("Precision", f"{row.precision:.1%}")
    m[3].metric("Cost (validation)", f"€{row.cost:,.0f}")

    pr = go.Figure(go.Scatter(x=curves["pr"]["recall"], y=curves["pr"]["precision"], name="PR"))
    pr.add_trace(go.Scatter(x=[row.recall], y=[row.precision], mode="markers", name="current"))
    pr.update_layout(
        title="Precision vs recall (validation)", xaxis_title="Recall", yaxis_title="Precision"
    )
    cost = go.Figure(go.Scatter(x=sweep.threshold, y=sweep.cost, name="cost"))
    cost.add_vline(x=t, line_dash="dash")
    cost.update_layout(
        title="Total cost vs threshold (validation)", xaxis_title="Threshold", yaxis_title="€"
    )
    g1, g2 = st.columns(2)
    g1.plotly_chart(pr, use_container_width=True)
    g2.plotly_chart(cost, use_container_width=True)

with drift_tab:
    st.caption(
        "Each button sends 500 legit transactions; the report covers the last 500 predictions."
    )
    c1, c2, c3 = st.columns(3)
    if c1.button("Send normal traffic"):
        predict_batch(prepare_rows(TRAIN_LEGIT.sample(500, replace=True)), policy)
    if c2.button("Send drifted traffic (V14 −3, Amount ×5)"):
        d = TRAIN_LEGIT.sample(500, replace=True).copy()
        d["V14"] -= 3
        d["Amount"] *= 5
        predict_batch(prepare_rows(d), policy)
    if c3.button("Send night-time traffic (18:00–24:00)"):
        predict_batch(prepare_rows(LEGIT.sample(500, replace=True)), policy)
    with st.expander("Why does real night-time traffic show drift?"):
        st.markdown(
            "V1, V3 and V28 vary with **time of day**. The reference pools all 48 hours of "
            "training data, so any few-hour window looks different, even a slice of the "
            "training data itself (V1 PSI ≈ 1.0). The model copes (score PSI stays low), but "
            "naive PSI raises a **seasonality false alarm**. Production fix: compare against a "
            "reference from the same hour or weekday, or use longer windows."
        )

    report = get("/drift", n=500)
    if report["status"] == "insufficient_data":
        st.info(f"Need at least 50 logged predictions; have {report['n']}. Send some traffic.")
    else:
        st.subheader(f"Overall: {report['status'].upper()} (last {report['n']} predictions)")
        feats = pd.DataFrame(report["features"]).T.reset_index(names="feature")
        feats["psi"] = feats.psi.astype(float)
        fig = go.Figure(go.Bar(x=feats.feature, y=feats.psi, marker_color=feats.status.map(COLORS)))
        fig.add_hline(y=0.1, line_dash="dot")
        fig.add_hline(y=0.25, line_dash="dash")
        fig.update_layout(title="PSI per feature", yaxis_title="PSI")
        st.plotly_chart(fig, use_container_width=True)
        m = st.columns(3)
        m[0].metric("Score PSI", f"{report['score']['psi']:.3f}", report["score"]["status"])
        m[1].metric("Live flag rate", f"{report['flag_rate']['live']:.2%}")
        m[2].metric("Expected flag rate", f"{report['flag_rate']['expected']:.2%}")
