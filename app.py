"""AGRIRAIN AI - Flask app. Routes only orchestrate; analysis lives in other modules."""
import math
import os
from functools import lru_cache

import numpy as np
from flask import Flask, jsonify, redirect, render_template, request, url_for

import charts as ch
import model as ml
from data_processing import (CORR_COLS, DataError, correlations, filter_df, key_findings,
                             load_data, overview_stats, quality_report, trend_slope, yearly)

app = Flask(__name__)
DATA_PATH = os.path.join(app.root_path, "data", "crop_yield.csv")

PAGES = [("overview", "Overview", "Project summary and dataset at a glance"),
         ("data_quality", "Data Quality", "Structure, missing values and duplicates"),
         ("eda", "EDA", "Exploratory analysis of crops, states, seasons and rainfall"),
         ("time_analysis", "Time Analysis", "How rainfall, yield and production change by year"),
         ("relationships", "Relationships", "Correlation between rainfall and agricultural factors"),
         ("gradient_boosting", "Gradient Boosting", "Model training, evaluation and feature importance"),
         ("forecast", "Forecast", "Yield Forecast Simulator using the trained model"),
         ("findings", "Findings", "Conclusions derived from the data and the model")]


# ---------- cached resources (loaded / trained once) ----------
@lru_cache(maxsize=1)
def get_df():
    return load_data(DATA_PATH)


@lru_cache(maxsize=1)
def get_model():
    return ml.train_all(get_df())


# ---------- template helpers ----------
@app.template_filter("num")
def num(v, d=2):
    return f"{v:,.{d}f}"


@app.context_processor
def nav():
    ep = request.endpoint
    idx = next((i for i, p in enumerate(PAGES) if p[0] == ep), None)
    prev = PAGES[idx - 1] if idx else None
    nxt = PAGES[idx + 1] if idx is not None and idx < len(PAGES) - 1 else None
    return dict(pages=PAGES, active=ep, page_no=(idx + 1 if idx is not None else 0),
                page=PAGES[idx] if idx is not None else None, prev_page=prev, next_page=nxt)


@app.errorhandler(DataError)
def data_error(e):
    return render_template("error.html", message=str(e)), 500


def page(template, charts=None, **ctx):
    return render_template(template, charts=charts or {}, **ctx)


# ---------- routes ----------
@app.route("/")
def home():
    return redirect(url_for("overview"))


@app.route("/overview")
def overview():
    df = get_df()
    s = overview_stats(df)
    cy = yearly(df)
    top = df.groupby("State").size().sort_values(ascending=False).head(10)
    charts = {"c1": ch.line({"Avg rainfall (mm)": (cy.Crop_Year, cy.rain)},
                            "Average Annual Rainfall by Year", "Year", "mm"),
              "c2": ch.bar(top.index, top.values, "Top 10 States by Number of Records",
                           yt="Records", horizontal=True)}
    return page("overview.html", charts, s=s)


@app.route("/data-quality")
def data_quality():
    return page("data_quality.html", q=quality_report(get_df()), s=overview_stats(get_df()))


@app.route("/eda")
def eda():
    df = get_df()
    vc = df.Crop.value_counts()
    sc = df.State.value_counts()
    se = df.Season.value_counts()
    ya = df.Yield[df.Yield > 0]
    ca = df.groupby("Crop").Yield.mean().sort_values(ascending=False).head(15)
    sa = df.groupby("State").Yield.mean().sort_values(ascending=False)
    pr = df.groupby("Crop").Production.sum().sort_values(ascending=False).head(15)
    rs = df.groupby("State").Annual_Rainfall.mean().sort_values(ascending=False)
    charts = {
        "crop_dist": ch.bar(vc.head(20).index, vc.head(20).values, "Top 20 Crops by Record Count", yt="Records", horizontal=True),
        "state_dist": ch.bar(sc.index, sc.values, "Records per State", yt="Records", horizontal=True),
        "season_dist": ch.bar(se.index, se.values, "Records per Season", "Season", "Records"),
        "rain_hist": ch.hist(df.Annual_Rainfall, "Annual Rainfall Distribution", "Annual rainfall (mm)"),
        "yield_hist": ch.hist(np.log10(ya), "Yield Distribution (log10 scale)", "log10(Yield)"),
        "crop_yield": ch.bar(ca.index, ca.values, "Top 15 Crops by Average Yield", yt="Avg yield", horizontal=True),
        "state_yield": ch.bar(sa.index, sa.values, "Average Yield by State", yt="Avg yield", horizontal=True),
        "crop_prod": ch.bar(pr.index, pr.values, "Top 15 Crops by Total Production", yt="Total production", horizontal=True),
        "state_rain": ch.bar(rs.index, rs.values, "Average Annual Rainfall by State", yt="mm", horizontal=True),
    }
    notes = {
        "crop": f"{vc.index[0]} has the most records ({vc.iloc[0]:,}, {vc.iloc[0]/len(df):.1%} of the data); {vc.index[-1]} has the fewest ({vc.iloc[-1]}).",
        "state": f"{sc.index[0]} contributes the most records ({sc.iloc[0]:,}); {sc.index[-1]} the fewest ({sc.iloc[-1]}).",
        "season": f"{se.index[0]} is the most frequent season ({se.iloc[0]:,} records).",
        "rain": f"Rainfall ranges from {df.Annual_Rainfall.min():,.1f} to {df.Annual_Rainfall.max():,.1f} mm; median {df.Annual_Rainfall.median():,.1f} mm, mean {df.Annual_Rainfall.mean():,.1f} mm.",
        "yield": f"Yield is strongly right-skewed: median {df.Yield.median():.2f} vs mean {df.Yield.mean():.2f}, max {df.Yield.max():,.1f}. The chart uses a log scale and excludes {int((df.Yield == 0).sum())} zero-yield records.",
        "crop_yield": f"{ca.index[0]} has the highest average yield ({ca.iloc[0]:,.1f}). Crops such as coconut appear to be recorded in different units, so yields are not directly comparable across all crops.",
        "state_yield": f"{sa.index[0]} has the highest average yield ({sa.iloc[0]:,.1f}); {sa.index[-1]} the lowest ({sa.iloc[-1]:,.2f}).",
        "prod": f"{pr.index[0]} has the highest total production ({pr.iloc[0]:,.0f}).",
        "srain": f"{rs.index[0]} has the highest average rainfall ({rs.iloc[0]:,.0f} mm); {rs.index[-1]} the lowest ({rs.iloc[-1]:,.0f} mm).",
    }
    return page("eda.html", charts, notes=notes)


def time_charts(df):
    """Build the five time-analysis charts for an (optionally filtered) frame."""
    cy = yearly(df)
    seas = {s: (g.groupby("Crop_Year").Yield.mean().index, g.groupby("Crop_Year").Yield.mean().values)
            for s, g in df.groupby("Season")}
    charts = {
        "rain": ch.line({"Avg rainfall": (cy.Crop_Year, cy.rain)}, "Year vs Average Annual Rainfall", "Year", "mm"),
        "yield": ch.line({"Avg yield": (cy.Crop_Year, cy["yield"])}, "Year vs Average Yield", "Year", "Yield"),
        "prod": ch.line({"Total production": (cy.Crop_Year, cy["prod"])}, "Year vs Total Production", "Year", "Production"),
        "activity": ch.line({"Distinct crops": (cy.Crop_Year, cy.crops), "Records": (cy.Crop_Year, cy.records)},
                            "Year-wise Crop Activity", "Year", "Count"),
        "season": ch.line(seas, "Seasonal Average Yield Trend", "Year", "Avg yield"),
    }
    notes = []
    if len(cy) > 1:
        yi, ri = cy.loc[cy["yield"].idxmax()], cy.loc[cy.rain.idxmax()]
        notes = [f"Records analysed: {len(df):,} across {len(cy)} years ({int(cy.Crop_Year.min())}-{int(cy.Crop_Year.max())}).",
                 f"Average rainfall trend: {trend_slope(cy.Crop_Year, cy.rain):+.2f} mm per year (linear fit); wettest year {int(ri.Crop_Year)} ({ri.rain:,.0f} mm).",
                 f"Average yield trend: {trend_slope(cy.Crop_Year, cy['yield']):+.2f} per year (linear fit); highest in {int(yi.Crop_Year)} ({yi['yield']:,.1f}).",
                 f"Total production peaked in {int(cy.loc[cy['prod'].idxmax()].Crop_Year)}."]
    elif len(cy) == 1:
        notes = ["Only one year of data matches these filters - no trend can be computed."]
    else:
        notes = ["No records match these filters."]
    return charts, notes


@app.route("/time-analysis")
def time_analysis():
    df = get_df()
    charts, notes = time_charts(df)
    return page("time_analysis.html", charts, notes=notes, states=sorted(df.State.unique()),
                crops=sorted(df.Crop.unique()), seasons=sorted(df.Season.unique()))


@app.route("/api/time-series")
def api_time_series():
    df = filter_df(get_df(), request.args.get("state", ""), request.args.get("crop", ""),
                   request.args.get("season", ""))
    charts, notes = time_charts(df)
    return jsonify(charts=charts, notes=notes, count=len(df))


@app.route("/relationships")
def relationships():
    df = get_df()
    pear, spear = correlations(df)
    samp = df.sample(min(4000, len(df)), random_state=42)
    pairs = ["Annual_Rainfall", "Area", "Fertilizer", "Pesticide", "Production"]
    charts = {"heat": ch.heatmap(pear.values, CORR_COLS, CORR_COLS, "Pearson Correlation Heatmap", height=480)}
    for c in pairs:
        charts[c] = ch.scatter(samp[c], samp.Yield, f"{c} vs Yield", c, "Yield", log=(c != "Annual_Rainfall"))
    rows = [{"var": c, "pearson": pear.loc[c, "Yield"], "spearman": spear.loc[c, "Yield"]} for c in pairs]
    sp = max(rows, key=lambda r: abs(r["spearman"]))
    pe = max(rows, key=lambda r: abs(r["pearson"]))
    rain = next(r for r in rows if r["var"] == "Annual_Rainfall")
    insights = [
        f"Strongest linear (Pearson) correlation with Yield: {pe['var']} (r = {pe['pearson']:.3f}).",
        f"Strongest rank (Spearman) correlation with Yield: {sp['var']} (rho = {sp['spearman']:.3f}); Spearman is less affected by the extreme outliers in this data.",
        f"Annual_Rainfall vs Yield: Pearson r = {rain['pearson']:.3f}, Spearman rho = {rain['spearman']:.3f} - rainfall alone is a {'weak' if abs(rain['spearman']) < 0.3 else 'moderate'} predictor across all crops pooled together.",
        "Correlation does not imply causation; yield depends strongly on crop type, which is not captured by a single numeric coefficient.",
    ]
    return page("relationships.html", charts, rows=rows, insights=insights, sample_n=len(samp))


@app.route("/gradient-boosting")
def gradient_boosting():
    a = get_model()
    yt, yp = a["y_test"], a["y_pred"]
    lim = [float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))]
    res = yt - yp
    lo, hi = np.percentile(res, [1, 99])
    cm = np.array(a["cls"]["cm"])
    L = a["cls"]["labels"]
    charts = {
        "avp": ch.scatter(yt, yp, "Actual vs Predicted Yield (Test Set)", "Actual yield", "Predicted yield", line_xy=(lim, lim)),
        "resid": ch.hist(res[(res >= lo) & (res <= hi)], "Residual Distribution (middle 98%)", "Actual - Predicted", 50),
        "imp": ch.bar([k for k, _ in a["imp_agg"]], [v for _, v in a["imp_agg"]], "Feature Importance (grouped by original feature)", yt="Importance", horizontal=True),
        "imp_d": ch.bar([k for k, _ in a["imp_detail"]], [v for _, v in a["imp_detail"]], "Top 12 Encoded Features", yt="Importance", horizontal=True),
        "cm": ch.heatmap(cm, [f"Pred {l}" for l in L], [f"Actual {l}" for l in L], "Confusion Matrix (Yield Class)", fmt=",.0f", scale="Greens", zmin=None, zmax=None, height=380),
    }
    return page("gradient_boosting.html", charts, a=a, params=ml.GB_PARAMS, feats=ml.FEATURES,
                test_size=ml.TEST_SIZE, rs=ml.RANDOM_STATE, top=a["imp_agg"][0])


def parse_form(form, df):
    """Validate forecast inputs. Returns (row, errors)."""
    row, err = {}, []
    for k, col in (("state", "State"), ("crop", "Crop"), ("season", "Season")):
        v = form.get(k, "")
        if v not in set(df[col]): err.append(f"Select a valid {col}.")
        row[col] = v
    ymin, ymax = int(df.Crop_Year.min()), int(df.Crop_Year.max())
    for k, col, lo, hi in (("year", "Crop_Year", ymin, ymax), ("area", "Area", 0.01, None),
                           ("production", "Production", 0, None), ("rain", "Annual_Rainfall", 0, None),
                           ("fert", "Fertilizer", 0, None), ("pest", "Pesticide", 0, None)):
        try:
            v = float(form.get(k, ""))
            if not math.isfinite(v) or v < lo or (hi is not None and v > hi): raise ValueError
            row[col] = v
        except ValueError:
            rng = f" between {lo:g} and {hi:g}" if hi is not None else f" of at least {lo:g}"
            err.append(f"{col} must be a number{rng}.")
    return row, err


@app.route("/forecast", methods=["GET", "POST"])
def forecast():
    df = get_df()
    med = df[["Area", "Production", "Annual_Rainfall", "Fertilizer", "Pesticide"]].median()
    defaults = {"state": "Karnataka", "crop": "Rice", "season": "Kharif", "year": int(df.Crop_Year.max()),
                "area": med.Area, "production": med.Production, "rain": med.Annual_Rainfall,
                "fert": med.Fertilizer, "pest": med.Pesticide}
    vals, result, errors = dict(defaults), None, []
    if request.method == "POST":
        vals = {k: request.form.get(k, "") for k in defaults}
        row, errors = parse_form(request.form, df)
        if not errors:
            p = ml.predict_yield(get_model(), row)
            hist = df[(df.Crop == row["Crop"]) & (df.State == row["State"])].Yield
            result = {"pred": max(p, 0.0), "raw": p, "row": row,
                      "crop_median": df[df.Crop == row["Crop"]].Yield.median(),
                      "pair_median": hist.median() if len(hist) else None, "pair_n": len(hist)}
    return page("forecast.html", defaults=defaults, vals=vals, result=result, errors=errors,
                states=sorted(df.State.unique()), crops=sorted(df.Crop.unique()),
                seasons=sorted(df.Season.unique()), ymin=int(df.Crop_Year.min()), ymax=int(df.Crop_Year.max()),
                m=get_model()["metrics"])


@app.route("/findings")
def findings():
    return page("findings.html", f=key_findings(get_df()), m=get_model()["metrics"],
                mnp=get_model()["metrics_no_prod"], s=overview_stats(get_df()), top=get_model()["imp_agg"][:3])


if __name__ == "__main__":
    get_df(); get_model()  # warm caches so the first page loads quickly
    app.run(debug=False, port=5000)
