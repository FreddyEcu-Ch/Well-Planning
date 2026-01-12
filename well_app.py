import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

ASSETS_DIR = Path(__file__).parent / "Resources"
LOGO_PATH = ASSETS_DIR / "ESPOL.png"
FICT_PATH = ASSETS_DIR / "FICT.png"
IMG_WELL_PATH = ASSETS_DIR / "well.jpg"
IMG_DD_PATH = ASSETS_DIR / "dd.jpg"


def show_image_if_exists(path: Path, **kwargs):
    if path.exists():
        st.image(str(path), **kwargs)
    else:
        st.warning(f"No encontré el archivo: {path.name} (revisa la carpeta assets/).")


st.set_page_config(page_title="Simulador de Pozos Direccionales 3D", layout="wide")


UNITS = "m"

# -----------------------------
# Defaults
# -----------------------------
DEFAULT_MANUAL = pd.DataFrame(
    [
        {"TVD": 0.0, "DespNS": 0.0, "DespEW": 0.0},
        {"TVD": 500.0, "DespNS": 0.0, "DespEW": 0.0},
        {"TVD": 1000.0, "DespNS": 120.0, "DespEW": 40.0},
    ]
)


# -----------------------------
# Validation & computations
# -----------------------------
def validate_coords(df: pd.DataFrame):
    required = ["TVD", "DespNS", "DespEW"]
    for c in required:
        if c not in df.columns:
            return False, f"Falta la columna '{c}'."
    df = df.copy()
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[required].isna().any().any():
        return False, "Hay celdas vacías o no numéricas."
    if (df["TVD"].values < 0).any():
        return False, "TVD no puede ser negativo."
    tvd = df["TVD"].values
    if np.any(np.diff(tvd) < 0):
        return False, "TVD debe ser NO decreciente (cada estación más profunda o igual)."
    if len(df) < 2:
        return False, "Agrega al menos 2 estaciones (incluida superficie)."
    return True, ""


def compute_from_coords(df: pd.DataFrame):
    """A partir de TVD/NS/EW calcula MD, Inc/Az por segmento, Dogleg y DLS (deg/30m)."""
    df = df.copy()
    for c in ["TVD", "DespNS", "DespEW"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    x = df["DespEW"].to_numpy(float)  # Este (+)
    y = df["DespNS"].to_numpy(float)  # Norte (+)
    z = df["TVD"].to_numpy(float)     # Abajo (+)

    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)

    seg = np.sqrt(dx * dx + dy * dy + dz * dz)
    md = np.concatenate([[0.0], np.cumsum(seg)])

    seg_safe = np.where(seg == 0, np.nan, seg)
    cos_inc = np.clip(dz / seg_safe, -1.0, 1.0)
    inc_seg = np.degrees(np.arccos(cos_inc))  # 0 vertical, 90 horizontal
    az_seg = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0  # 0=N, 90=E

    inc = np.concatenate([[0.0], inc_seg])
    az = np.concatenate([[0.0], az_seg])

    dog = np.zeros(len(df))
    if len(df) >= 3:
        inc1 = np.radians(inc_seg[:-1])
        inc2 = np.radians(inc_seg[1:])
        az1 = np.radians(az_seg[:-1])
        az2 = np.radians(az_seg[1:])
        cos_dog = np.cos(inc1) * np.cos(inc2) + np.sin(inc1) * np.sin(inc2) * np.cos(az2 - az1)
        cos_dog = np.clip(cos_dog, -1.0, 1.0)
        dogleg = np.degrees(np.arccos(cos_dog))
        dog[2:] = dogleg

    dls = np.zeros(len(df))
    if len(df) >= 3:
        dmd = np.diff(md)
        dmd2 = dmd[1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            dls_val = np.where(dmd2 > 0, dog[2:] / dmd2 * 30.0, 0.0)  # deg/30m
        dls[2:] = np.nan_to_num(dls_val)

    out = df.copy()
    out["MD"] = md
    out["Inc_deg"] = inc
    out["Az_deg"] = az
    out["Dogleg_deg"] = dog
    out["DLS_deg_per_30m"] = dls
    return out


# -----------------------------
# Minimum Curvature generator
# -----------------------------
def mc_step(delta_md, inc1_deg, az1_deg, inc2_deg, az2_deg):
    inc1 = math.radians(inc1_deg)
    inc2 = math.radians(inc2_deg)
    az1 = math.radians(az1_deg)
    az2 = math.radians(az2_deg)

    cos_dog = math.cos(inc1) * math.cos(inc2) + math.sin(inc1) * math.sin(inc2) * math.cos(az2 - az1)
    cos_dog = max(-1.0, min(1.0, cos_dog))
    dogleg = math.acos(cos_dog)  # rad

    if dogleg < 1e-10:
        rf = 1.0
    else:
        rf = (2.0 / dogleg) * math.tan(dogleg / 2.0)

    dN = (delta_md / 2.0) * (math.sin(inc1) * math.cos(az1) + math.sin(inc2) * math.cos(az2)) * rf
    dE = (delta_md / 2.0) * (math.sin(inc1) * math.sin(az1) + math.sin(inc2) * math.sin(az2)) * rf
    dTVD = (delta_md / 2.0) * (math.cos(inc1) + math.cos(inc2)) * rf

    return dTVD, dN, dE, math.degrees(dogleg)


def _append_station(stations, delta_md, inc_next, az_next, label):
    cur = stations[-1]
    dTVD, dN, dE, dog = mc_step(delta_md, cur["Inc_deg"], cur["Az_deg"], inc_next, az_next)
    stations.append(
        {
            "MD": cur["MD"] + delta_md,
            "Inc_deg": inc_next,
            "Az_deg": az_next,
            "TVD": cur["TVD"] + dTVD,
            "DespNS": cur["DespNS"] + dN,
            "DespEW": cur["DespEW"] + dE,
            "Segmento": label,
            "Dogleg_deg": dog,
        }
    )


def vertical_to_tvd(stations, tvd_target, step_md, label):
    while stations[-1]["TVD"] < tvd_target - 1e-6:
        cur = stations[-1]
        remaining = tvd_target - cur["TVD"]
        delta = min(step_md, remaining)  # inc=0 => dTVD≈deltaMD
        _append_station(stations, delta, 0.0, cur["Az_deg"], label)
    return stations


def build_to_inc(stations, inc_target, rate_deg_per_30m, step_md, label):
    rate = float(rate_deg_per_30m)
    if abs(rate) < 1e-9:
        return stations

    while True:
        cur = stations[-1]
        inc = cur["Inc_deg"]
        if (rate > 0 and inc >= inc_target - 1e-6) or (rate < 0 and inc <= inc_target + 1e-6):
            break

        delta = step_md
        inc_next = inc + rate * (delta / 30.0)

        if rate > 0:
            inc_next = min(inc_next, inc_target)
        else:
            inc_next = max(inc_next, inc_target)

        _append_station(stations, delta, inc_next, cur["Az_deg"], label)
    return stations


def hold_md(stations, length_md, step_md, label):
    remaining = float(length_md)
    while remaining > 1e-6:
        delta = min(step_md, remaining)
        cur = stations[-1]
        _append_station(stations, delta, cur["Inc_deg"], cur["Az_deg"], label)
        remaining -= delta
    return stations


def compute_dls(df: pd.DataFrame):
    df = df.copy()
    md = df["MD"].to_numpy(float)
    dog = df["Dogleg_deg"].to_numpy(float)
    dls = np.zeros(len(df))
    dmd = np.diff(md)
    dls[1:] = np.where(dmd > 0, dog[1:] / dmd * 30.0, 0.0)
    df["DLS_deg_per_30m"] = dls
    return df


def generate_well(well_type: str,
                  az_deg: float,
                  step_md: float,
                  kop_tvd: float,
                  build_rate: float,
                  drop_rate: float,
                  inc_target: float,
                  inc_max: float,
                  hold_md_len: float,
                  lateral_md_len: float,
                  final_tvd: float | None):
    """
    Genera estaciones usando Minimum Curvature.

    - J: vertical -> build a inc_target -> hold (hasta final_tvd si aplica)
    - Horizontal: vertical -> build a 90° -> lateral (lateral_md_len)
    - S: vertical -> build a inc_max -> hold (hold_md_len) -> drop a 0° -> vertical final hasta final_tvd
    """
    stations = [
        {
            "MD": 0.0,
            "Inc_deg": 0.0,
            "Az_deg": float(az_deg),
            "TVD": 0.0,
            "DespNS": 0.0,
            "DespEW": 0.0,
            "Segmento": "Superficie",
            "Dogleg_deg": 0.0,
        }
    ]

    stations = vertical_to_tvd(stations, kop_tvd, step_md, "Vertical (hasta KOP)")

    if well_type == "J":
        stations = build_to_inc(stations, inc_target, abs(build_rate), step_md, "Build")
        if final_tvd is not None:
            if stations[-1]["Inc_deg"] < 89.9:
                while stations[-1]["TVD"] < final_tvd - 1e-6:
                    cur = stations[-1]
                    remaining_tvd = final_tvd - cur["TVD"]
                    cosi = max(1e-6, math.cos(math.radians(cur["Inc_deg"])))
                    delta = min(step_md, remaining_tvd / cosi)
                    _append_station(stations, delta, cur["Inc_deg"], cur["Az_deg"], "Hold")
    elif well_type == "Horizontal":
        stations = build_to_inc(stations, 90.0, abs(build_rate), step_md, "Build")
        stations = hold_md(stations, lateral_md_len, step_md, "Lateral")
    elif well_type == "S":
        stations = build_to_inc(stations, inc_max, abs(build_rate), step_md, "Build")
        stations = hold_md(stations, hold_md_len, step_md, "Hold")
        stations = build_to_inc(stations, 0.0, -abs(drop_rate), step_md, "Drop")
        if final_tvd is not None:
            stations = vertical_to_tvd(stations, final_tvd, step_md, "Vertical final")
    else:
        raise ValueError("well_type debe ser: J, S o Horizontal")

    df = pd.DataFrame(stations)
    df = compute_dls(df)
    return df


# -----------------------------
# Plot
# -----------------------------
def make_3d_figure(survey: pd.DataFrame, targets: list[dict] | None = None):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=survey["DespEW"],
            y=survey["DespNS"],
            z=survey["TVD"],
            mode="lines+markers",
            name="Pozo",
            line=dict(width=6),
            marker=dict(size=4),
            hovertemplate="TVD=%{z:.1f}<br>NS=%{y:.1f}<br>EW=%{x:.1f}<extra></extra>",
        )
    )

    if targets:
        for t in targets:
            fig.add_trace(
                go.Scatter3d(
                    x=[t["DespEW"]],
                    y=[t["DespNS"]],
                    z=[t["TVD"]],
                    mode="markers",
                    name=t.get("name", "Objetivo"),
                    marker=dict(size=8, symbol="diamond"),
                    hovertemplate=(
                        f"{t.get('name','Objetivo')}<br>"
                        "TVD=%{z:.1f}<br>NS=%{y:.1f}<br>EW=%{x:.1f}<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        height=650,
        margin=dict(l=10, r=10, t=10, b=10),
        scene=dict(
            xaxis_title=f"DespEW ({UNITS})",
            yaxis_title=f"DespNS ({UNITS})",
            zaxis_title=f"TVD ({UNITS})",
            aspectmode="data",
            zaxis=dict(autorange="reversed"),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


# -----------------------------
# Game helpers
# -----------------------------
def random_targets(seed: int, tvd_range, disp_range):
    rng = np.random.default_rng(seed)
    landing = {
        "name": "Landing",
        "TVD": float(rng.uniform(*tvd_range)),
        "DespNS": float(rng.uniform(*disp_range)),
        "DespEW": float(rng.uniform(*disp_range)),
    }
    toe = {
        "name": "Toe",
        "TVD": float(landing["TVD"] + rng.uniform(-50, 50)),
        "DespNS": float(landing["DespNS"] + rng.uniform(300, 1200) * np.sign(rng.normal())),
        "DespEW": float(landing["DespEW"] + rng.uniform(300, 1200) * np.sign(rng.normal())),
    }
    toe["TVD"] = max(0.0, toe["TVD"])
    return landing, toe


def closest_station_distance(survey: pd.DataFrame, target: dict):
    dx = survey["DespEW"].to_numpy(float) - target["DespEW"]
    dy = survey["DespNS"].to_numpy(float) - target["DespNS"]
    dz = survey["TVD"].to_numpy(float) - target["TVD"]
    dist = np.sqrt(dx * dx + dy * dy + dz * dz)
    i = int(np.argmin(dist))
    return float(dist[i]), int(i)


def score_run(survey: pd.DataFrame, landing: dict, toe: dict, tol: float, dls_max: float):
    d_land, _ = closest_station_distance(survey, landing)
    d_toe, _ = closest_station_distance(survey, toe)

    exceed = survey["DLS_deg_per_30m"].to_numpy(float) - dls_max
    exceed = exceed[exceed > 0]
    penalty_dls = float(exceed.sum() * 20.0) if len(exceed) else 0.0

    score = 1000.0
    score -= 3.0 * d_land
    score -= 3.0 * d_toe
    score -= penalty_dls
    score = max(0.0, score)

    win = (d_land <= tol) and (d_toe <= tol) and (penalty_dls == 0.0)
    return {"d_land": d_land, "d_toe": d_toe, "score": score, "penalty_dls": penalty_dls, "win": win}


# -----------------------------
# Session state
# -----------------------------
if "manual_points" not in st.session_state:
    st.session_state.manual_points = DEFAULT_MANUAL.copy()

if "generated_survey" not in st.session_state:
    st.session_state.generated_survey = None

if "game" not in st.session_state:
    st.session_state.game = {"seed": 10, "landing": None, "toe": None}

# -----------------------------
# UI
# -----------------------------
st.title("🛢️ Simulador: Pozos Direccionales 3D (J, S, Horizontal)")

with st.expander("🖼️ Figuras de referencia (perforación direccional)", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        show_image_if_exists(IMG_WELL_PATH, caption="Ejemplo: pozo direccional / BHA", use_container_width=True)
    with c2:
        show_image_if_exists(IMG_DD_PATH, caption="Ejemplo: perforación direccional (visual)", use_container_width=True)


tab1, tab2, tab3 = st.tabs(
    ["✍️ Constructor manual (TVD/NS/EW)", "🧰 Generador (J / S / Horizontal)", "🎮 Modo juego (Landing + Toe)"]
)

# -----------------------------
# Tab 1: Manual
# -----------------------------
with tab1:
    st.markdown("Edita estaciones usando **TVD**, **DespNS** y **DespEW**. Se calcula **MD, Inc, Az y DLS**.")
    edited = st.data_editor(
        st.session_state.manual_points,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "TVD": st.column_config.NumberColumn("TVD", format="%.2f"),
            "DespNS": st.column_config.NumberColumn("DespNS", format="%.2f"),
            "DespEW": st.column_config.NumberColumn("DespEW", format="%.2f"),
        },
        key="manual_editor",
    )
    st.session_state.manual_points = edited.copy()

    ok, msg = validate_coords(st.session_state.manual_points)
    if not ok:
        st.error(msg)
    else:
        survey = compute_from_coords(st.session_state.manual_points)
        c1, c2 = st.columns([1.0, 1.2], gap="large")
        with c1:
            st.subheader("📋 Survey calculado")
            st.dataframe(
                survey[["TVD", "DespNS", "DespEW", "MD", "Inc_deg", "Az_deg", "DLS_deg_per_30m"]],
                use_container_width=True,
                hide_index=True,
            )
            csv = survey.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Descargar CSV", data=csv, file_name="survey_manual.csv", mime="text/csv")
        with c2:
            st.subheader("🧭 3D")
            st.plotly_chart(make_3d_figure(survey), use_container_width=True)

# -----------------------------
# Tab 2: Generator
# -----------------------------
with tab2:
    st.markdown("Genera trayectorias típicas. Luego puedes **copiar estaciones al tab manual**.")

    with st.sidebar:
        # Logo ESPOL en el lateral
        show_image_if_exists(LOGO_PATH, use_container_width=True)
        st.markdown("---")
        show_image_if_exists(FICT_PATH, use_container_width=True)
        st.markdown("---")

        st.header("⚙️ Parámetros del generador")
        well_type = st.radio("Tipo de pozo", ["J", "S", "Horizontal"], index=0)
        az_deg = st.slider("Azimut (°) [0=N, 90=E]", 0.0, 360.0, 45.0, step=1.0)
        step_md = st.select_slider("Paso MD para estaciones", options=[10.0, 15.0, 30.0, 45.0, 60.0], value=30.0)

        kop_tvd = st.number_input("KOP TVD", min_value=0.0, value=500.0, step=50.0)
        build_rate = st.number_input("Build rate (°/30m)", min_value=0.1, value=3.0, step=0.1)

        dls_max = st.number_input("Límite DLS (°/30m) (alerta)", min_value=0.1, value=6.0, step=0.1)

        if well_type == "J":
            inc_target = st.slider("Inclinación objetivo (°)", 5.0, 89.0, 60.0, step=1.0)
            final_tvd = st.number_input("TVD final (termina en Hold)", min_value=kop_tvd, value=2000.0, step=50.0)
            inc_max = 0.0
            hold_md_len = 0.0
            drop_rate = 0.0
            lateral_md_len = 0.0
        elif well_type == "Horizontal":
            inc_target = 90.0
            lateral_md_len = st.number_input("Longitud lateral (MD)", min_value=100.0, value=800.0, step=50.0)
            final_tvd = None
            inc_max = 0.0
            hold_md_len = 0.0
            drop_rate = 0.0
        else:  # S
            inc_max = st.slider("Inclinación máxima (°)", 5.0, 80.0, 45.0, step=1.0)
            hold_md_len = st.number_input("Hold (MD)", min_value=0.0, value=600.0, step=50.0)
            drop_rate = st.number_input("Drop rate (°/30m)", min_value=0.1, value=3.0, step=0.1)
            final_tvd = st.number_input("TVD final", min_value=kop_tvd, value=2500.0, step=50.0)
            inc_target = 0.0
            lateral_md_len = 0.0

        generate = st.button("🧰 Generar trayectoria", use_container_width=True)

    if generate or st.session_state.generated_survey is None:
        st.session_state.generated_survey = generate_well(
            well_type=well_type,
            az_deg=az_deg,
            step_md=step_md,
            kop_tvd=kop_tvd,
            build_rate=build_rate,
            drop_rate=drop_rate,
            inc_target=inc_target,
            inc_max=inc_max,
            hold_md_len=hold_md_len,
            lateral_md_len=lateral_md_len,
            final_tvd=final_tvd,
        )

    survey = st.session_state.generated_survey.copy()

    max_dls = float(np.nanmax(survey["DLS_deg_per_30m"].to_numpy(float)))
    if max_dls > dls_max + 1e-6:
        st.warning(f"⚠️ DLS máximo calculado = {max_dls:.2f} °/30m (supera {dls_max:.2f}).")
    else:
        st.success(f"✅ DLS máximo calculado = {max_dls:.2f} °/30m (dentro de {dls_max:.2f}).")

    c1, c2 = st.columns([1.0, 1.2], gap="large")
    with c1:
        st.subheader("📋 Survey generado")
        st.dataframe(
            survey[["Segmento", "MD", "Inc_deg", "Az_deg", "TVD", "DespNS", "DespEW", "DLS_deg_per_30m"]],
            use_container_width=True,
            hide_index=True,
        )

        csv = survey.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Descargar CSV (generado)", data=csv, file_name=f"survey_{well_type}.csv", mime="text/csv")

        if st.button("➡️ Copiar estaciones al tab manual", use_container_width=True):
            st.session_state.manual_points = survey[["TVD", "DespNS", "DespEW"]].copy()
            st.success("Listo: se copiaron las estaciones al constructor manual.")
    with c2:
        st.subheader("🧭 3D")
        st.plotly_chart(make_3d_figure(survey), use_container_width=True)

# -----------------------------
# Tab 3: Game
# -----------------------------
with tab3:
    st.markdown(
        "Meta: pasar cerca de **dos objetivos** (**Landing** y **Toe**) sin exceder **DLS máximo**."
    )

    gcol1, gcol2 = st.columns([1.0, 1.2], gap="large")

    with gcol1:
        st.subheader("🎯 Objetivos")
        seed = st.number_input("Seed", min_value=0, value=int(st.session_state.game["seed"]), step=1)
        tvd_min, tvd_max = st.slider("Rango TVD (Landing)", 300.0, 4000.0, (900.0, 2200.0), step=50.0)
        disp_abs = st.slider("Máx |Desp| (Landing)", 0.0, 3000.0, 900.0, step=50.0)
        tol = st.slider("Tolerancia (m)", 10.0, 300.0, 80.0, step=5.0)
        dls_max_game = st.slider("DLS máximo permitido (°/30m)", 1.0, 15.0, 6.0, step=0.5)

        if st.button("🎲 Nueva misión", use_container_width=True):
            landing, toe = random_targets(int(seed), (tvd_min, tvd_max), (-disp_abs, disp_abs))
            st.session_state.game = {"seed": int(seed) + 1, "landing": landing, "toe": toe}

        if st.session_state.game["landing"] is None:
            landing, toe = random_targets(int(seed), (tvd_min, tvd_max), (-disp_abs, disp_abs))
            st.session_state.game = {"seed": int(seed) + 1, "landing": landing, "toe": toe}

        landing = st.session_state.game["landing"]
        toe = st.session_state.game["toe"]

        st.info(
            f"**Landing**: TVD={landing['TVD']:.1f}, NS={landing['DespNS']:.1f}, EW={landing['DespEW']:.1f}\n\n"
            f"**Toe**: TVD={toe['TVD']:.1f}, NS={toe['DespNS']:.1f}, EW={toe['DespEW']:.1f}"
        )

        st.subheader("🧩 Trayectoria a evaluar")
        source = st.radio("¿Cuál usar?", ["Generada (tab 2)", "Manual (tab 1)"], index=0)

        if source == "Generada (tab 2)":
            survey_eval = st.session_state.generated_survey.copy()
        else:
            ok, msg = validate_coords(st.session_state.manual_points)
            if not ok:
                st.warning(f"Tu tabla manual tiene un problema: {msg}")
                st.stop()
            survey_eval = compute_from_coords(st.session_state.manual_points)

        result = score_run(survey_eval, landing, toe, tol=tol, dls_max=dls_max_game)

        st.subheader("🏁 Resultado")
        st.metric("Distancia mínima a Landing", f"{result['d_land']:.1f} {UNITS}")
        st.metric("Distancia mínima a Toe", f"{result['d_toe']:.1f} {UNITS}")
        st.metric("Puntaje", f"{result['score']:.0f}")

        if result["penalty_dls"] > 0:
            st.error("Excediste el DLS máximo permitido. Ajusta tu diseño.")
        if result["win"]:
            st.success("¡Misión completada! 🎉")
        else:
            st.warning("Aún no. Ajusta (tipo, KOP, build/drop, azimut, lateral, etc.) y prueba otra vez.")

        st.caption("Tip: para misiones tipo horizontal, usa 'Horizontal' y ajusta azimut + longitud lateral.")

    with gcol2:
        st.subheader("🧭 Pozo 3D con objetivos")
        st.plotly_chart(make_3d_figure(survey_eval, targets=[landing, toe]), use_container_width=True)

        st.subheader("📋 Últimas estaciones")
        cols = ["MD", "Inc_deg", "Az_deg", "TVD", "DespNS", "DespEW", "DLS_deg_per_30m"]
        if "Segmento" in survey_eval.columns:
            cols = ["Segmento"] + cols
        st.dataframe(survey_eval.tail(12)[cols], use_container_width=True, hide_index=True)
