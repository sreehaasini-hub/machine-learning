"""Builds Plotly figure specs (plain dicts) so no server-side plotly install is needed."""
import numpy as np

GREEN = "#2e8b57"


def _clean(a):
    return [None if (isinstance(v, float) and not np.isfinite(v)) else v for v in np.asarray(a).tolist()]


def _layout(title, xt="", yt="", **kw):
    d = dict(title=dict(text=title, font=dict(size=15)), xaxis=dict(title=xt, automargin=True),
             yaxis=dict(title=yt, automargin=True), margin=dict(l=60, r=20, t=50, b=50),
             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
             font=dict(family="Inter, Segoe UI, sans-serif", color="#1f3d2e"))
    d.update(kw)
    return d


def bar(x, y, title, xt="", yt="", horizontal=False, color=GREEN):
    x, y = _clean(x), _clean(y)
    if horizontal:
        t = dict(type="bar", x=y, y=x, orientation="h", marker=dict(color=color))
        lay = _layout(title, yt, xt, yaxis=dict(autorange="reversed", automargin=True))
    else:
        t = dict(type="bar", x=x, y=y, marker=dict(color=color))
        lay = _layout(title, xt, yt)
    return {"data": [t], "layout": lay}


def hist(values, title, xt, nbins=40, color=GREEN):
    return {"data": [dict(type="histogram", x=_clean(values), nbinsx=nbins, marker=dict(color=color))],
            "layout": _layout(title, xt, "Count", bargap=0.05)}


def line(series, title, xt, yt):
    data = [dict(type="scatter", mode="lines+markers", name=n, x=_clean(x), y=_clean(y))
            for n, (x, y) in series.items()]
    return {"data": data, "layout": _layout(title, xt, yt, legend=dict(orientation="h", y=-0.25))}


def scatter(x, y, title, xt, yt, log=False, line_xy=None, color=GREEN):
    data = [dict(type="scattergl", mode="markers", x=_clean(x), y=_clean(y),
                 marker=dict(size=4, opacity=0.45, color=color), name="Records")]
    if line_xy is not None:
        data.append(dict(type="scatter", mode="lines", x=line_xy[0], y=line_xy[1],
                         line=dict(color="#c0392b", dash="dash"), name="Perfect prediction"))
    lay = _layout(title, xt, yt)
    if log:
        lay["xaxis"]["type"] = "log"; lay["yaxis"]["type"] = "log"
    return {"data": data, "layout": lay}


def heatmap(z, xl, yl, title, fmt=".2f", scale="RdYlGn", zmin=-1, zmax=1, height=None):
    z = np.round(np.asarray(z, dtype=float), 3)
    txt = [[format(v, fmt) for v in row] for row in z]
    lay = _layout(title, margin=dict(l=110, r=20, t=50, b=90))
    lay["yaxis"] = dict(autorange="reversed", automargin=True)
    if height: lay["height"] = height
    t = dict(type="heatmap", z=z.tolist(), x=xl, y=yl, text=txt, texttemplate="%{text}",
             colorscale=scale, colorbar=dict(thickness=12))
    if zmin is not None: t.update(zmin=zmin, zmax=zmax)
    return {"data": [t], "layout": lay}
