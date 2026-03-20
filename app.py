import math
from datetime import datetime, timedelta
from io import StringIO

import folium
import numpy as np
import pandas as pd
import streamlit as st
from scipy.ndimage import gaussian_filter1d
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Dump Truck GPS Tracker",
    page_icon="🚛",
    layout="wide",
)

# ── Smoothing algorithms ──────────────────────────────────────────────────────

def smooth_moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(arr)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def smooth_gaussian(arr: np.ndarray, window: int) -> np.ndarray:
    return gaussian_filter1d(arr.astype(float), sigma=window / 6, mode="nearest")


def smooth_median(arr: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(arr)
        .rolling(window=window, center=True, min_periods=1)
        .median()
        .to_numpy()
    )


ALGORITHMS: dict[str, callable] = {
    "Gaussian Weighted": smooth_gaussian,
    "Moving Average": smooth_moving_average,
    "Median Filter": smooth_median,
}


def apply_smoothing(df: pd.DataFrame, algorithm: str, window: int) -> pd.DataFrame:
    fn = ALGORITHMS[algorithm]
    result = df.copy()
    result["lat"] = fn(df["lat"].to_numpy(), window)
    result["lng"] = fn(df["lng"].to_numpy(), window)
    return result


# ── Sample data ───────────────────────────────────────────────────────────────

@st.cache_data
def generate_sample_data() -> pd.DataFrame:
    """
    Simulates 3 haul cycles for a mining truck:
      load zone  →  haul road  →  dump zone  →  return road
    GPS noise is large at the zones (maneuvering) and small on the haul road.
    Uses a deterministic LCG so the output is reproducible.
    """
    state = [42]

    def rand() -> float:
        state[0] = (state[0] * 1664525 + 1013904223) & 0xFFFFFFFF
        return state[0] / 0xFFFFFFFF

    def noise(v: float, mag: float) -> float:
        return v + (rand() - 0.5) * 2 * mag

    def lerp(a: tuple, b: tuple, t: float) -> tuple:
        return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)

    load_zone = (-27.5012, 151.0003)
    dump_zone = (-27.4798, 151.0312)
    haul_road = [
        (-27.5012, 151.0003),
        (-27.4990, 151.0045),
        (-27.4965, 151.0090),
        (-27.4940, 151.0135),
        (-27.4915, 151.0180),
        (-27.4890, 151.0224),
        (-27.4863, 151.0268),
        (-27.4835, 151.0298),
        (-27.4798, 151.0312),
    ]

    rows: list[dict] = []
    t = datetime(2024, 3, 15, 7, 0, 0)

    for _ in range(3):
        # Loading zone — large noise (truck manoeuvring, multi-point turns)
        for _ in range(50):
            rows.append(
                {
                    "timestamp": t,
                    "lat": noise(load_zone[0], 0.00035),
                    "lng": noise(load_zone[1], 0.00035),
                    "speed_kmh": rand() * 6,
                }
            )
            t += timedelta(seconds=5)

        # Haul road (loaded)
        for seg in range(len(haul_road) - 1):
            for frac in np.linspace(0, 1, 11):
                la, lo = lerp(haul_road[seg], haul_road[seg + 1], frac)
                rows.append(
                    {
                        "timestamp": t,
                        "lat": noise(la, 0.000075),
                        "lng": noise(lo, 0.000075),
                        "speed_kmh": 22 + rand() * 18,
                    }
                )
                t += timedelta(seconds=3)

        # Dump zone — large noise
        for _ in range(45):
            rows.append(
                {
                    "timestamp": t,
                    "lat": noise(dump_zone[0], 0.00035),
                    "lng": noise(dump_zone[1], 0.00035),
                    "speed_kmh": rand() * 7,
                }
            )
            t += timedelta(seconds=5)

        # Return road (empty, slightly offset lane, faster)
        for seg in range(len(haul_road) - 1, 0, -1):
            for frac in np.linspace(0, 1, 11):
                la, lo = lerp(haul_road[seg], haul_road[seg - 1], frac)
                rows.append(
                    {
                        "timestamp": t,
                        "lat": noise(la + 0.00015, 0.000075),
                        "lng": noise(lo - 0.00015, 0.000075),
                        "speed_kmh": 30 + rand() * 20,
                    }
                )
                t += timedelta(seconds=2.5)

    return pd.DataFrame(rows)


# ── CSV parsing ───────────────────────────────────────────────────────────────

@st.cache_data
def parse_csv(content: str) -> pd.DataFrame:
    df = pd.read_csv(StringIO(content))
    df.columns = df.columns.str.strip().str.lower()

    lat_col = next((c for c in df.columns if c in ("lat", "latitude")), None)
    lng_col = next((c for c in df.columns if c in ("lng", "lon", "longitude")), None)
    if not lat_col or not lng_col:
        raise ValueError(
            f"Could not find lat/lng columns. Found: {list(df.columns)}"
        )

    ts_col = next(
        (c for c in df.columns if c in ("timestamp", "time", "datetime", "date")), None
    )
    spd_col = next(
        (c for c in df.columns if "speed" in c or "velocity" in c), None
    )

    result = pd.DataFrame(
        {"lat": df[lat_col].astype(float), "lng": df[lng_col].astype(float)}
    )
    if ts_col:
        result["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")
    if spd_col:
        result["speed_kmh"] = df[spd_col].astype(float)

    return result.dropna(subset=["lat", "lng"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def haversine_total(df: pd.DataFrame) -> float:
    R = 6_371_000
    lat = np.radians(df["lat"].to_numpy())
    lng = np.radians(df["lng"].to_numpy())
    dlat, dlng = np.diff(lat), np.diff(lng)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlng / 2) ** 2
    return float(np.sum(R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))))


def fmt_dist(metres: float) -> str:
    return f"{metres / 1000:.2f} km" if metres >= 1000 else f"{metres:.0f} m"


def fmt_duration(df: pd.DataFrame) -> str:
    if "timestamp" not in df.columns:
        return "—"
    ts = pd.to_datetime(df["timestamp"])
    secs = int((ts.iloc[-1] - ts.iloc[0]).total_seconds())
    if secs <= 0:
        return "—"
    h, m = divmod(secs // 60, 60)
    return f"{h}h {m}m" if h else f"{m}m"


def detect_zones(
    df: pd.DataFrame, speed_thresh: float = 6.0, min_pts: int = 15
) -> list[tuple[int, int]]:
    """Return (start, end) index pairs for slow/stationary clusters."""
    spd = df["speed_kmh"].to_numpy() if "speed_kmh" in df.columns else np.zeros(len(df))
    zones, in_zone, start = [], False, 0
    for i, s in enumerate(spd):
        if not in_zone and s < speed_thresh:
            in_zone, start = True, i
        elif in_zone and s >= speed_thresh:
            in_zone = False
            if i - start >= min_pts:
                zones.append((start, i - 1))
    if in_zone and len(spd) - start >= min_pts:
        zones.append((start, len(spd) - 1))
    return zones


# ── Map builder ───────────────────────────────────────────────────────────────

def build_map(
    raw: pd.DataFrame,
    smooth: pd.DataFrame,
    show_raw: bool,
    show_smooth: bool,
    show_markers: bool,
) -> folium.Map:
    center = [raw["lat"].mean(), raw["lng"].mean()]
    m = folium.Map(location=center, zoom_start=14, tiles="CartoDB dark_matter")

    if show_raw:
        folium.PolyLine(
            raw[["lat", "lng"]].values.tolist(),
            color="#fc8181",
            weight=2,
            opacity=0.55,
            tooltip="Raw GPS",
        ).add_to(m)

    if show_smooth:
        folium.PolyLine(
            smooth[["lat", "lng"]].values.tolist(),
            color="#68d391",
            weight=3,
            opacity=0.9,
            tooltip="Smoothed",
        ).add_to(m)

    if show_markers:
        # Start marker
        folium.CircleMarker(
            raw[["lat", "lng"]].iloc[0].tolist(),
            radius=7,
            color="#ffffff",
            fill_color="#68d391",
            fill_opacity=1,
            popup="<b>Start</b>",
        ).add_to(m)
        # End marker
        folium.CircleMarker(
            raw[["lat", "lng"]].iloc[-1].tolist(),
            radius=7,
            color="#ffffff",
            fill_color="#fc8181",
            fill_opacity=1,
            popup="<b>End</b>",
        ).add_to(m)

        zone_colors = ["#68d391", "#f6ad55"] * 10
        zone_labels = ["Load Zone", "Dump Zone"] * 10
        for idx, (s, e) in enumerate(detect_zones(raw)):
            centroid = raw.iloc[s : e + 1][["lat", "lng"]].mean()
            folium.CircleMarker(
                [centroid["lat"], centroid["lng"]],
                radius=12,
                color="#ffffff",
                fill_color=zone_colors[idx],
                fill_opacity=0.75,
                popup=f"<b>{zone_labels[idx]}</b><br/>{e - s + 1} points",
            ).add_to(m)

    sw = [raw["lat"].min(), raw["lng"].min()]
    ne = [raw["lat"].max(), raw["lng"].max()]
    m.fit_bounds([sw, ne], padding=(30, 30))
    return m


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🚛 GPS Tracker")

    st.subheader("Data Source")
    uploaded = st.file_uploader("Upload CSV", type="csv", label_visibility="collapsed")
    use_sample = st.button("↺  Use Sample Data", use_container_width=True)

    st.markdown(
        "<small style='color:#718096'>Required columns: `lat`/`latitude`, "
        "`lng`/`lon`/`longitude`<br/>Optional: `timestamp`, `speed`</small>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.subheader("Smoothing")
    algorithm = st.selectbox("Algorithm", list(ALGORITHMS.keys()))
    window = st.slider("Window Size", min_value=3, max_value=31, step=2, value=11)

    st.divider()
    st.subheader("Visibility")
    show_raw     = st.checkbox("Raw GPS path",  value=True)
    show_smooth  = st.checkbox("Smoothed path", value=True)
    show_markers = st.checkbox("Zone markers",  value=True)


# ── Session state ─────────────────────────────────────────────────────────────

if "df" not in st.session_state:
    # Auto-load sample data on first visit
    st.session_state.df = generate_sample_data()
    st.session_state.source = "sample_truck_data.csv (demo)"

if use_sample:
    st.session_state.df = generate_sample_data()
    st.session_state.source = "sample_truck_data.csv (demo)"

if uploaded is not None:
    try:
        st.session_state.df = parse_csv(uploaded.read().decode())
        st.session_state.source = uploaded.name
    except ValueError as exc:
        st.error(str(exc))


# ── Main content ──────────────────────────────────────────────────────────────

st.title("Dump Truck GPS Tracker")

df: pd.DataFrame = st.session_state.df
smooth_df = apply_smoothing(df, algorithm, window)

st.caption(
    f"Source: **{st.session_state.source}** — {len(df):,} GPS points  |  "
    f"Algorithm: **{algorithm}**, window = **{window}**"
)

# Map
m = build_map(df, smooth_df, show_raw, show_smooth, show_markers)
st_folium(m, use_container_width=True, height=560, returned_objects=[])

# Stats row
d_raw = haversine_total(df)
d_sm  = haversine_total(smooth_df)
noise_pct = (d_raw - d_sm) / d_raw * 100 if d_raw > 0 else 0.0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("GPS Points",      f"{len(df):,}")
c2.metric("Trip Duration",   fmt_duration(df))
c3.metric("Raw Distance",    fmt_dist(d_raw))
c4.metric("Smooth Distance", fmt_dist(d_sm))
c5.metric("Noise Removed",   f"{noise_pct:.1f}%",
          help="Reduction in total path length after smoothing")

# GPS coordinates table
st.subheader("GPS Coordinates")

coord_df = pd.DataFrame({
    "timestamp":        df["timestamp"] if "timestamp" in df.columns else range(len(df)),
    "raw_lat":          df["lat"].round(6),
    "raw_lng":          df["lng"].round(6),
    "smoothed_lat":     smooth_df["lat"].round(6),
    "smoothed_lng":     smooth_df["lng"].round(6),
    "lat_delta":        (smooth_df["lat"] - df["lat"]).round(6),
    "lng_delta":        (smooth_df["lng"] - df["lng"]).round(6),
})

def highlight_raw(s):
    return ["color: #fc8181" if s.name in ("raw_lat", "raw_lng") else
            "color: #68d391" if s.name in ("smoothed_lat", "smoothed_lng") else
            "" for _ in s]

st.dataframe(
    coord_df.style.apply(highlight_raw),
    use_container_width=True,
    height=300,
)
