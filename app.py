"""
무드 믹서 - 10문항으로 감정·스트레스를 수치화하고, 지금 마음에 맞는 스포티파이 노래를 추천하는 Streamlit 앱
실행: streamlit run app.py
"""
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------
# 0. 페이지 설정 / 상수
# ------------------------------------------------------------------
st.set_page_config(page_title="무드 믹서", page_icon="🎚️", layout="wide")

try:
    APP_PASSWORD = st.secrets.get("APP_PASSWORD", "pnumis")
except Exception:
    APP_PASSWORD = "pnumis"

# 색상 토큰
INK, PANEL, LINE = "#080C0A", "#111A15", "#243229"
TEXT, MUTED = "#E8F1EA", "#8DA396"
GREEN, AMBER, CORAL, TEAL, VIOLET = "#1ED760", "#F2C14E", "#FF6F61", "#4FD1C5", "#A78BFA"
MOOD_COLOR = {"Energetic": GREEN, "Calm": TEAL, "Melancholic": VIOLET, "Intense": CORAL}
MOOD_KR = {"Energetic": "신나는", "Calm": "차분한", "Melancholic": "우울·감성적인", "Intense": "강렬한"}

# 설문
QUESTIONS = [
    "최근 일주일간 해야 할 일이 감당하기 어려울 만큼 쌓여 있다고 느꼈다.",
    "사소한 일에도 마음이 조마조마하고 초조했다.",
    "충분히 자도 몸이 개운하지 않고 피곤했다.",
    "집중이 잘 안 되고 쉽게 멍해졌다.",
    "평소 좋아하던 일에도 흥미가 생기지 않았다.",
    "이유 없이 기분이 가라앉거나 외롭다고 느꼈다.",
    "별일 아닌 일에도 짜증이 났다.",
    "다른 사람의 말이나 행동에 평소보다 예민하게 반응했다.",
    "최근 기분 좋은 일이나 기대되는 일이 있었다.",
    "내 생활을 스스로 잘 통제하고 있다고 느꼈다.",
]
LIKERT = {1: "전혀 그렇지 않다", 2: "그렇지 않다", 3: "보통이다", 4: "그렇다", 5: "매우 그렇다"}
AXES = {"긴장·불안": [0, 1], "피로": [2, 3], "우울·무기력": [4, 5], "분노·짜증": [6, 7], "활력·긍정": [8, 9]}
NEGATIVE_AXES = ["긴장·불안", "피로", "우울·무기력", "분노·짜증"]
LOW_CUT, HIGH_CUT = 40, 60

MOOD_PLAN = {
    "긴장·불안": ("Calm", "Energetic"),
    "피로": ("Calm", "Energetic"),
    "우울·무기력": ("Melancholic", "Energetic"),
    "분노·짜증": ("Intense", "Calm"),
}
PLAN_REASON = {
    "긴장·불안": "마음을 가라앉히는 차분한 곡으로 시작해요.",
    "피로": "에너지를 쓰지 않고 쉴 수 있는 차분한 곡으로 시작해요.",
    "우울·무기력": "지금 기분에 공감해 주는 감성적인 곡으로 시작해요.",
    "분노·짜증": "쌓인 감정을 풀어낼 수 있는 강렬한 곡으로 시작해요.",
}
AXIS_DESC = {
    "긴장·불안": "해야 할 일이 많거나 마음이 조급한 상태를 뜻해요.",
    "피로": "잠을 자도 회복이 덜 되고 집중이 흐트러지는 상태를 뜻해요.",
    "우울·무기력": "흥미가 줄고 기분이 가라앉아 있는 상태를 뜻해요.",
    "분노·짜증": "작은 자극에도 예민해지는 상태를 뜻해요.",
}
RELIEF_TIPS = {
    "긴장·불안": ["4초 들이마시고 6초 내쉬는 호흡을 5분간 반복하기", "해야 할 일을 종이에 모두 적고 우선순위 3개만 남기기", "카페인 섭취를 줄이고 따뜻한 차 마시기"],
    "피로": ["20분 이내의 짧은 낮잠 또는 눈 감고 쉬기", "가볍게 스트레칭하고 물 한 컵 마시기", "오늘은 잠드는 시간을 30분만 앞당기기"],
    "우울·무기력": ["10분 산책하며 햇볕 쬐기", "가까운 사람에게 안부 메시지 보내기", "아주 작은 일(방 정리 5분)부터 하나 끝내기"],
    "분노·짜증": ["잠시 자리를 벗어나 찬 물로 손 씻기", "감정을 메모장에 그대로 적고 지우기", "격한 운동(줄넘기, 빠르게 걷기)으로 에너지 발산하기"],
}

# 추천 / 분류
REC_FEATURES = ["valence", "energy", "acousticness", "danceability", "tempo"]
CLF_FEATURES = ["danceability", "acousticness", "instrumentalness", "speechiness", "liveness", "loudness", "tempo"]
MOOD_TARGETS = {
    "Energetic": dict(valence=0.80, energy=0.85, acousticness=0.10, danceability=0.75, tempo=125),
    "Calm": dict(valence=0.60, energy=0.30, acousticness=0.70, danceability=0.50, tempo=95),
    "Melancholic": dict(valence=0.20, energy=0.30, acousticness=0.70, danceability=0.40, tempo=100),
    "Intense": dict(valence=0.25, energy=0.90, acousticness=0.05, danceability=0.50, tempo=135),
}
GENRE_KEYWORDS = {
    "Pop": ["pop"],
    "Hip Hop / Rap": ["hip hop", "rap", "trap"],
    "R&B / Soul": ["r&b", "soul"],
    "Rock": ["rock"],
    "Indie": ["indie"],
    "Electronic / Dance": ["electro", "dance", "edm", "house", "techno", "rave", "tronica"],
    "Latin": ["latin", "reggaeton", "salsa", "bachata", "cumbia"],
    "K-pop": ["k-pop", "korean"],
    "Jazz": ["jazz"],
    "Folk / Acoustic": ["folk", "acoustic", "singer-songwriter"],
}
GENRE_OPTIONS = ["상관없음"] + list(GENRE_KEYWORDS)


# ------------------------------------------------------------------
# 1. 스타일
# ------------------------------------------------------------------
def H(s: str) -> str:
    """여러 줄 HTML/CSS를 한 줄로 합쳐 마크다운 코드블록 오인을 막는다."""
    return " ".join(line.strip() for line in s.strip().splitlines())


CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Black+Han+Sans&family=IBM+Plex+Sans+KR:wght@400;500;600;700&display=swap');
.stApp, .stApp p, .stApp label, .stApp li, .stApp input, .stApp button, .stApp textarea {{ font-family: 'IBM Plex Sans KR', sans-serif; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2.2rem; max-width: 1240px; }}
section[data-testid="stSidebar"] {{ background: #0D140F; border-right: 1px solid {LINE}; }}
.disp {{ font-family: 'Black Han Sans', 'IBM Plex Sans KR', sans-serif; font-weight: 400; letter-spacing: -0.01em; }}
.muted {{ color: {MUTED}; }}
.card {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 20px; padding: 22px 24px; }}
.card-sm {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 12px; padding: 14px 16px; }}
.hero-title {{ font-size: 3.6rem; line-height: 1.05; margin: 0 0 10px 0; color: {TEXT}; }}
.hero-sub {{ font-size: 1.05rem; color: {MUTED}; max-width: 620px; line-height: 1.65; margin: 0 0 22px 0; }}
.steps {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 26px; }}
.step b {{ color: {GREEN}; font-size: 1.5rem; display: block; margin-bottom: 4px; }}
.step span {{ font-size: .92rem; color: {MUTED}; line-height: 1.5; }}
.q {{ margin: 20px 0 4px 0; font-size: 1.02rem; line-height: 1.5; }}
.qn {{ display: inline-block; min-width: 2.1rem; color: {GREEN}; font-weight: 700; }}
.sec-title {{ font-size: 1.15rem; font-weight: 700; margin: 0 0 12px 0; }}
.mixer {{ display: flex; justify-content: space-between; gap: 6px; padding: 8px 2px 0 2px; }}
.fader {{ flex: 1; display: flex; flex-direction: column; align-items: center; gap: 8px; }}
.rail {{ position: relative; width: 10px; height: 220px; background: #1B2820; border-radius: 6px; }}
.fill {{ position: absolute; left: 0; right: 0; bottom: 0; border-radius: 6px; }}
.knob {{ position: absolute; left: 50%; width: 34px; height: 14px; margin-left: -17px; border-radius: 4px; background: {TEXT}; border: 2px solid {INK}; }}
.tick {{ position: absolute; left: -9px; right: -9px; height: 0; border-top: 1px dashed #3A4D41; }}
.fval {{ font-size: 1.35rem; color: {TEXT}; }}
.fname {{ font-size: .8rem; color: {MUTED}; text-align: center; line-height: 1.3; min-height: 2.1em; }}
.headline {{ font-size: 2.7rem; line-height: 1.15; margin: 4px 0 14px 0; color: {TEXT}; }}
.idxrow {{ display: flex; align-items: baseline; gap: 10px; margin-bottom: 18px; }}
.idx {{ font-size: 3.1rem; line-height: 1; }}
.pill {{ display: inline-block; padding: 3px 12px; border-radius: 999px; font-size: .85rem; font-weight: 700; }}
.kpis {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }}
.kpi-l {{ font-size: .8rem; color: {MUTED}; margin-bottom: 6px; }}
.kpi-v {{ font-size: 1.35rem; font-weight: 700; line-height: 1.2; }}
.kpi-s {{ font-size: .78rem; color: {MUTED}; margin-top: 4px; }}
.brief li {{ margin-bottom: 10px; line-height: 1.65; color: #CFDCD2; }}
.brief ul {{ padding-left: 1.1rem; margin: 0; }}
.meter {{ display: flex; align-items: center; gap: 8px; font-size: .78rem; color: {MUTED}; margin-top: 6px; }}
.meter span {{ width: 2.6rem; }}
.mtrack {{ flex: 1; height: 5px; background: #1B2820; border-radius: 3px; }}
.mfill {{ height: 5px; border-radius: 3px; background: {GREEN}; }}
.meter b {{ width: 2.2rem; text-align: right; color: {TEXT}; font-weight: 500; }}
.yr {{ font-size: .78rem; color: {MUTED}; margin-top: 6px; }}
.tip {{ border-left: 3px solid {GREEN}; padding: 6px 0 6px 14px; margin-bottom: 10px; color: #CFDCD2; line-height: 1.55; }}
.notice {{ font-size: .82rem; color: {MUTED}; margin-top: 22px; line-height: 1.6; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ color: {MUTED}; font-weight: 600; padding: 10px 16px; }}
.stTabs [aria-selected="true"] {{ color: {GREEN}; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {GREEN}; }}
button[kind="primary"], button[kind="primaryFormSubmit"] {{ background: {GREEN}; color: #04120A; border: none; border-radius: 999px; font-weight: 700; padding: .5rem 1.6rem; }}
button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover {{ background: #3BE477; color: #04120A; }}
button[kind="secondary"], button[kind="secondaryFormSubmit"] {{ border-radius: 999px; border: 1px solid {LINE}; background: transparent; }}
div[data-testid="stRadio"] label p {{ font-size: .92rem; }}
@media (max-width: 900px) {{
  .hero-title {{ font-size: 2.6rem; }} .headline {{ font-size: 2rem; }}
  .steps, .kpis {{ grid-template-columns: 1fr 1fr; }}
}}
"""


def inject_css():
    st.markdown(f"<style>{H(CSS)}</style>", unsafe_allow_html=True)


# ------------------------------------------------------------------
# 2. 데이터 / 모델
# ------------------------------------------------------------------
@st.cache_data(show_spinner="곡 데이터를 불러오는 중...")
def load_songs() -> pd.DataFrame:
    df = pd.read_csv("songs_for_app.csv")
    df["genre_list"] = df["genre_tags"].fillna("").str.split("|")
    return df


songs = load_songs()


def genre_mask(frame: pd.DataFrame, genre: str) -> pd.Series:
    kws = GENRE_KEYWORDS[genre]
    return frame["genre_list"].apply(lambda tags: any(k in t for t in tags for k in kws))


@st.cache_resource
def get_rec_scaler():
    return StandardScaler().fit(songs[REC_FEATURES])


@st.cache_resource
def get_projection():
    """분위기 지도용 PCA(2차원). 분류에 쓰는 7개 특성을 표준화한 뒤 투영."""
    scaler = StandardScaler().fit(songs[CLF_FEATURES])
    pca = PCA(n_components=2, random_state=42).fit(scaler.transform(songs[CLF_FEATURES]))
    return scaler, pca


@st.cache_data(show_spinner="KNN과 PCA 성능을 비교하는 중...")
def run_model_test() -> pd.DataFrame:
    X, y = songs[CLF_FEATURES], songs["mood"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def make(n_comp):
        steps = [("scaler", StandardScaler())]
        if n_comp:
            steps.append(("pca", PCA(n_components=n_comp, random_state=42)))
        steps.append(("knn", KNeighborsClassifier(n_neighbors=31)))
        return Pipeline(steps)

    rows = []
    for n in [None] + list(range(1, len(CLF_FEATURES) + 1)):
        pred = make(n).fit(X_tr, y_tr).predict(X_te)
        rows.append({
            "모델": "KNN (PCA 없음)" if n is None else f"KNN + PCA {n}개",
            "차원 수": len(CLF_FEATURES) if n is None else n,
            "정확도": accuracy_score(y_te, pred),
            "Macro-F1": f1_score(y_te, pred, average="macro"),
        })
    return pd.DataFrame(rows)


def recommend(mood, hour=None, genre=None, n=5, seed=0):
    """분위기 목표 프로필과 가까운 곡을 KNN으로 찾아 n개 반환. (표, 실제 적용된 장르)"""
    t = dict(MOOD_TARGETS[mood])
    if hour is not None and 0 <= hour < 6:  # 새벽이면 더 감성적으로
        t["energy"] = max(0.0, t["energy"] - 0.10)
        t["acousticness"] = min(1.0, t["acousticness"] + 0.15)
        t["valence"] = max(0.0, t["valence"] - 0.05)

    pool = songs[songs["mood"] == mood]
    applied = None
    if genre and genre != "상관없음":
        gp = pool[genre_mask(pool, genre)]
        if len(gp) >= 30:
            pool, applied = gp, genre

    scaler = get_rec_scaler()
    nn = NearestNeighbors(n_neighbors=min(30, len(pool))).fit(scaler.transform(pool[REC_FEATURES]))
    idx = nn.kneighbors(scaler.transform(pd.DataFrame([t], columns=REC_FEATURES)), return_distance=False)[0]
    near = pool.iloc[idx]
    rng = np.random.default_rng(seed)
    pick = near.iloc[rng.choice(len(near), size=min(n, len(near)), replace=False)]
    return pick.reset_index(drop=True), applied


# ------------------------------------------------------------------
# 3. 설문 점수화
# ------------------------------------------------------------------
def parse_hour(text):
    """'14', '02:30', '오후 2시' 같은 입력을 0~23 정수로 변환. 실패하면 None."""
    s = str(text or "").strip().lower()
    m = re.search(r"(\d{1,2})", s)
    if not m:
        return None
    h = int(m.group(1))
    if ("오후" in s or "pm" in s) and h < 12:
        h += 12
    if ("오전" in s or "am" in s) and h == 12:
        h = 0
    return h if 0 <= h <= 23 else None


def hour_label(h):
    period = "새벽" if h < 6 else "오전" if h < 12 else "오후" if h < 18 else "저녁"
    return f"{period} {h % 12 or 12}시"


def analyze(answers):
    scores = {k: (sum(answers[i] for i in idx) - 2) / 8 * 100 for k, idx in AXES.items()}
    index = float(np.mean([scores[a] for a in NEGATIVE_AXES] + [100 - scores["활력·긍정"]]))
    level = "낮음" if index < LOW_CUT else "보통" if index < HIGH_CUT else "높음"
    dominant = max(NEGATIVE_AXES, key=lambda a: scores[a])
    primary, secondary = MOOD_PLAN[dominant]
    return dict(scores=scores, index=index, level=level, dominant=dominant, primary=primary, secondary=secondary)


def level_color(score):
    return GREEN if score < LOW_CUT else AMBER if score < HIGH_CUT else CORAL


# ------------------------------------------------------------------
# 4. 그래프
# ------------------------------------------------------------------
def style_fig(fig, height=360):
    fig.update_layout(
        height=height, margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans KR, sans-serif", color=TEXT, size=13),
    )
    return fig


def radar_fig(scores):
    labels, vals = list(scores), list(scores.values())
    fig = go.Figure(go.Scatterpolar(
        r=vals + vals[:1], theta=labels + labels[:1], fill="toself",
        line=dict(color=GREEN, width=2.5), fillcolor="rgba(30,215,96,0.22)",
        hovertemplate="%{theta}: %{r:.0f}점<extra></extra>",
    ))
    fig.update_layout(polar=dict(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(range=[0, 100], gridcolor=LINE, linecolor=LINE, tickfont=dict(size=10, color=MUTED), tickvals=[25, 50, 75, 100]),
        angularaxis=dict(gridcolor=LINE, linecolor=LINE, tickfont=dict(size=14)),
    ), showlegend=False)
    return style_fig(fig, 380)


def gauge_fig(index):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(index, 1),
        number=dict(suffix="점", font=dict(size=46, color=TEXT)),
        gauge=dict(
            axis=dict(range=[0, 100], tickvals=[0, 40, 60, 100], tickcolor=MUTED),
            bar=dict(color=level_color(index), thickness=0.28), bgcolor="rgba(0,0,0,0)", borderwidth=0,
            steps=[dict(range=[0, LOW_CUT], color="#10301C"), dict(range=[LOW_CUT, HIGH_CUT], color="#35301A"),
                   dict(range=[HIGH_CUT, 100], color="#3A201D")],
        ),
    ))
    return style_fig(fig, 240)


def map_fig(highlight: pd.DataFrame):
    scaler, pca = get_projection()
    sample = songs.sample(min(2500, len(songs)), random_state=1)
    xy = pca.transform(scaler.transform(sample[CLF_FEATURES]))
    fig = go.Figure()
    for mood, color in MOOD_COLOR.items():
        m = (sample["mood"] == mood).to_numpy()
        fig.add_trace(go.Scattergl(
            x=xy[m, 0], y=xy[m, 1], mode="markers", name=MOOD_KR[mood],
            marker=dict(size=5, color=color, opacity=0.45),
            text=sample.loc[m, "track_name"] + " - " + sample.loc[m, "artists"],
            hovertemplate="%{text}<extra></extra>",
        ))
    if highlight is not None and len(highlight):
        hxy = pca.transform(scaler.transform(highlight[CLF_FEATURES]))
        fig.add_trace(go.Scatter(
            x=hxy[:, 0], y=hxy[:, 1], mode="markers", name="내 추천곡",
            marker=dict(size=15, symbol="star", color="#FFFFFF", line=dict(color=INK, width=1.5)),
            text=highlight["track_name"] + " - " + highlight["artists"], hovertemplate="%{text}<extra></extra>",
        ))
    var = pca.explained_variance_ratio_
    fig.update_layout(
        xaxis=dict(title=f"PC1 ({var[0]:.0%})", gridcolor=LINE, zeroline=False),
        yaxis=dict(title=f"PC2 ({var[1]:.0%})", gridcolor=LINE, zeroline=False),
        legend=dict(orientation="h", y=1.1),
    )
    return style_fig(fig, 420)


# ------------------------------------------------------------------
# 5. 화면 조각
# ------------------------------------------------------------------
def render_gate():
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.markdown(H("""<div style="height:12vh"></div>
            <div class="card"><div class="disp" style="font-size:2.6rem;margin-bottom:6px">무드 믹서</div>
            <div class="muted" style="margin-bottom:8px">입장 비밀번호를 입력해 주세요.</div></div>"""), unsafe_allow_html=True)
        pw = st.text_input("비밀번호", type="password", label_visibility="collapsed", placeholder="비밀번호")
        if st.button("입장하기", type="primary"):
            if pw == APP_PASSWORD:
                st.session_state["auth"] = True
                st.rerun()
            else:
                st.error("비밀번호가 맞지 않아요. 다시 확인해 주세요.")


def render_sidebar():
    with st.sidebar:
        st.markdown(f'<div class="disp" style="font-size:2rem;margin-bottom:2px">무드 믹서</div>'
                    f'<div class="muted" style="font-size:.88rem;margin-bottom:18px">지금 마음에 맞는 노래 찾기</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="sec-title">사전 조사</div>', unsafe_allow_html=True)
        st.text_input("지금 몇 시인가요?", key="time_text", placeholder="예: 14:30, 오후 2시, 02")
        st.selectbox("선호하는 장르", GENRE_OPTIONS, key="genre")
        h = parse_hour(st.session_state.get("time_text"))
        if st.session_state.get("time_text") and h is None:
            st.warning("시간을 알아보지 못했어요. 14, 14:30, 오후 2시처럼 입력해 주세요.")
        elif h is not None:
            st.caption(f"{hour_label(h)}로 인식했어요." + (" 새벽이라 더 감성적인 곡을 찾아요." if h < 6 else ""))
        if st.session_state.get("stage") == "result":
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            if st.button("설문 다시 하기"):
                for i in range(len(QUESTIONS)):
                    st.session_state.pop(f"q{i}", None)
                st.session_state["stage"] = "survey"
                st.rerun()
        st.markdown(f'<div class="notice">장르나 시간을 바꾸면 결과 화면의 추천곡이 바로 바뀌어요.</div>', unsafe_allow_html=True)


def render_survey():
    st.markdown(H("""<div class="hero-title disp">지금 마음의 볼륨은<br>어디쯤일까요?</div>
        <p class="hero-sub">질문 10개에 답하면 감정 5가지를 믹서처럼 수치로 보여 드리고,
        그에 맞는 노래를 골라 드려요. 스트레스가 높게 나오면 해소법도 함께 알려 드려요.</p>"""), unsafe_allow_html=True)
    st.markdown(H("""<div class="steps">
        <div class="card-sm step"><b>1</b><span>왼쪽에서 지금 시간과 좋아하는 장르를 골라요.</span></div>
        <div class="card-sm step"><b>2</b><span>최근 일주일의 느낌에 가까운 정도를 10문항에 답해요.</span></div>
        <div class="card-sm step"><b>3</b><span>감정 믹서, 스트레스 지수, 추천곡을 확인해요.</span></div></div>"""), unsafe_allow_html=True)

    with st.form("survey"):
        for i, q in enumerate(QUESTIONS):
            st.markdown(f'<div class="q"><span class="qn">{i + 1}</span>{q}</div>', unsafe_allow_html=True)
            st.radio(q, options=list(LIKERT), format_func=LIKERT.get, index=None, horizontal=True,
                     key=f"q{i}", label_visibility="collapsed")
        submitted = st.form_submit_button("결과 보기", type="primary")

    if submitted:
        answers = [st.session_state.get(f"q{i}") for i in range(len(QUESTIONS))]
        missing = [str(i + 1) for i, a in enumerate(answers) if a is None]
        hour = parse_hour(st.session_state.get("time_text"))
        if hour is None:
            st.error("왼쪽 사전 조사에서 현재 시간을 먼저 입력해 주세요. (예: 14:30, 오후 2시)")
        elif missing:
            st.error(f"아직 답하지 않은 문항이 있어요: {', '.join(missing)}번")
        else:
            st.session_state["answers"] = answers
            st.session_state["seed"] = 7
            st.session_state["stage"] = "result"
            st.rerun()


def render_mixer(scores):
    cols = []
    for name, v in scores.items():
        color = TEAL if name == "활력·긍정" else level_color(v)
        cols.append(f"""<div class="fader"><div class="fval disp">{v:.0f}</div>
            <div class="rail"><div class="tick" style="bottom:{LOW_CUT}%"></div><div class="tick" style="bottom:{HIGH_CUT}%"></div>
            <div class="fill" style="height:{v}%;background:{color}"></div>
            <div class="knob" style="bottom:calc({v}% - 7px)"></div></div>
            <div class="fname">{name}</div></div>""")
    st.markdown(H(f"""<div class="card"><div class="sec-title">감정 믹서</div><div class="mixer">{''.join(cols)}</div>
        <div class="notice" style="margin-top:14px">점선은 보통(40점)과 높음(60점) 경계예요.
        활력·긍정은 높을수록 좋은 상태예요.</div></div>"""), unsafe_allow_html=True)


def meter(label, value):
    return (f'<div class="meter"><span>{label}</span><div class="mtrack"><div class="mfill" style="width:{value * 100:.0f}%"></div></div>'
            f'<b>{value:.2f}</b></div>')


def spotify_embed(track_id):
    """스포티파이 임베드 플레이어(어두운 테마). 최신 Streamlit은 st.iframe, 구버전은 components.html 사용."""
    url = f"https://open.spotify.com/embed/track/{track_id}?utm_source=generator&theme=0"
    if hasattr(st, "iframe"):
        st.iframe(url, height=152)
    else:
        components.html(
            f'<iframe style="border-radius:12px" src="{url}" width="100%" height="152" frameborder="0" '
            f'allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe>',
            height=158)


def render_song_grid(df: pd.DataFrame, per_row=3):
    for start in range(0, len(df), per_row):
        cols = st.columns(per_row)
        for col, (_, r) in zip(cols, df.iloc[start:start + per_row].iterrows()):
            with col:
                spotify_embed(r["track_id"])
                st.markdown(H(f"""{meter('밝기', r['valence'])}{meter('에너지', r['energy'])}
                    <div class="yr">{int(r['release_year'])}년 발매</div>"""), unsafe_allow_html=True)


def build_brief(res, hour, genre, applied):
    idx, level, dom = res["index"], res["level"], res["dominant"]
    sc = res["scores"]
    items = []
    if level == "낮음":
        items.append(f"종합 스트레스 지수는 <b>{idx:.0f}점</b>으로 낮은 편이에요. 지금의 균형을 잘 유지하고 있어요.")
    elif level == "보통":
        items.append(f"종합 스트레스 지수는 <b>{idx:.0f}점</b>으로 보통 구간(40~59점)이에요. 약간의 긴장이 쌓여 있지만 크게 걱정할 수준은 아니에요.")
    else:
        items.append(f"종합 스트레스 지수는 <b>{idx:.0f}점</b>으로 높음 구간(60점 이상)이에요. 쉬어 줄 시간이 필요해 보여요.")
    items.append(f"부정 감정 중 가장 높은 요소는 <b>{dom}({sc[dom]:.0f}점)</b>이에요. {AXIS_DESC[dom]}")
    vit = sc["활력·긍정"]
    items.append(f"활력·긍정은 <b>{vit:.0f}점</b>으로 " + ("낮은 편이라 스트레스 지수를 끌어올리고 있어요." if vit < 40 else "보통이에요." if vit < 60 else "높은 편이라 스트레스를 버티는 힘이 되어 줘요."))
    if level != "낮음":
        s = f"{PLAN_REASON[dom]}"
        if level == "높음":
            s += f" 이어서 기분 전환용 {MOOD_KR[res['secondary']]} 곡도 골랐어요."
        items.append(s)
        extra = []
        if hour is not None and hour < 6:
            extra.append("새벽 시간대라 에너지는 낮추고 어쿠스틱한 곡을 더 반영했어요")
        if genre and genre != "상관없음":
            extra.append(f"선호 장르 {genre}에서 먼저 찾았어요" if applied else f"{genre} 곡이 부족해 장르와 상관없이 골랐어요")
        if extra:
            items.append(", ".join(extra) + ".")
    return "<ul>" + "".join(f"<li>{t}</li>" for t in items) + "</ul>"


def render_result():
    res = analyze(st.session_state["answers"])
    hour = parse_hour(st.session_state.get("time_text"))
    genre = st.session_state.get("genre", "상관없음")
    seed = st.session_state.get("seed", 7)
    idx, level, dom = res["index"], res["level"], res["dominant"]
    color = level_color(idx)

    # ---- 추천 계산 (낮음이면 노래를 추천하지 않음)
    recs1 = recs2 = None
    applied1 = None
    if level != "낮음":
        recs1, applied1 = recommend(res["primary"], hour, genre, 5, seed)
        if level == "높음":
            recs2, _ = recommend(res["secondary"], hour, genre, 3, seed)

    headline = {"낮음": "마음이 안정적이에요", "보통": "조금 무거운 하루였네요", "높음": "마음이 꽤 지쳐 있어요"}[level]

    # ---- 히어로: 믹서 + 헤드라인/KPI
    left, right = st.columns([5, 7], gap="large")
    with left:
        render_mixer(res["scores"])
    with right:
        rec_label = f"{MOOD_KR[res['primary']]} 곡" if level != "낮음" else "추천 없음"
        rec_sub = (f"{genre} 우선" if applied1 else "장르 무관") if level != "낮음" else "지수 유지 중"
        st.markdown(H(f"""<div class="headline disp">{headline}</div>
            <div class="idxrow"><span class="idx disp" style="color:{color}">{idx:.0f}</span><span class="muted">/ 100점</span>
            <span class="pill" style="background:{color}26;color:{color}">{level}</span></div>
            <div class="kpis">
            <div class="card-sm"><div class="kpi-l">종합 지수</div><div class="kpi-v">{idx:.1f}점</div><div class="kpi-s">낮음 40 미만, 높음 60 이상</div></div>
            <div class="card-sm"><div class="kpi-l">가장 높은 요소</div><div class="kpi-v">{dom}</div><div class="kpi-s">{res['scores'][dom]:.0f}점</div></div>
            <div class="card-sm"><div class="kpi-l">활력·긍정</div><div class="kpi-v">{res['scores']['활력·긍정']:.0f}점</div><div class="kpi-s">높을수록 좋아요</div></div>
            <div class="card-sm"><div class="kpi-l">추천 분위기</div><div class="kpi-v">{rec_label}</div><div class="kpi-s">{rec_sub}</div></div></div>"""),
                    unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["결과 요약", "추천곡과 해소법", "곡 분위기 지도"])

    # ---- 탭1: 차트 + 브리프
    with tab1:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown('<div class="sec-title">감정 레이더</div>', unsafe_allow_html=True)
            st.plotly_chart(radar_fig(res["scores"]))
        with c2:
            st.markdown('<div class="sec-title">종합 스트레스 지수</div>', unsafe_allow_html=True)
            st.plotly_chart(gauge_fig(idx))
            st.markdown(f'<div class="card brief"><div class="sec-title">분석 브리프</div>{build_brief(res, hour, genre, applied1)}</div>',
                        unsafe_allow_html=True)
        st.markdown('<div class="notice">이 결과는 진단이 아니라 참고용이에요. 힘든 상태가 길어지면 가까운 사람이나 전문 상담 기관과 이야기해 보세요.</div>',
                    unsafe_allow_html=True)

    # ---- 탭2: 추천곡 + 해소법
    with tab2:
        if level == "낮음":
            st.markdown(H(f"""<div class="card"><div class="headline disp" style="font-size:2rem">좋은 결과군요!</div>
                <p style="line-height:1.7;margin:0">스트레스 지수가 낮아서 이번에는 해소법이나 추천곡이 필요하지 않아요.
                이 지수를 유지하기 위해 산책이나 취미 활동을 틈틈이 해 보세요.</p></div>"""), unsafe_allow_html=True)
        else:
            top, btn = st.columns([5, 1])
            with top:
                st.markdown(f'<div class="sec-title">지금 기분에 맞는 {MOOD_KR[res["primary"]]} 노래</div>', unsafe_allow_html=True)
            with btn:
                if st.button("다른 곡 보기"):
                    st.session_state["seed"] = seed + 1
                    st.rerun()
            render_song_grid(recs1)
            if level == "높음":
                st.markdown(f'<div class="sec-title" style="margin-top:26px">기분 전환을 위한 {MOOD_KR[res["secondary"]]} 노래</div>',
                            unsafe_allow_html=True)
                render_song_grid(recs2)
                st.markdown(f'<div class="sec-title" style="margin-top:26px">스트레스 해소법: {dom}</div>', unsafe_allow_html=True)
                for t in RELIEF_TIPS[dom]:
                    st.markdown(f'<div class="tip">{t}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="sec-title" style="margin-top:26px">가벼운 팁</div><div class="tip">{RELIEF_TIPS[dom][0]}</div>',
                            unsafe_allow_html=True)

    # ---- 탭3: 모델/데이터
    with tab3:
        shown = pd.concat([d for d in (recs1, recs2) if d is not None]) if recs1 is not None else None
        st.markdown('<div class="sec-title">곡 분위기 지도 (PCA 2차원)</div>', unsafe_allow_html=True)
        st.caption("곡의 오디오 특성 7개를 2개 축으로 줄여 한 화면에 그렸어요. 흰 별이 추천된 곡이에요. 점 위에 마우스를 올리면 곡명이 보여요.")
        st.plotly_chart(map_fig(shown))

        a, b = st.columns(2, gap="large")
        with a:
            st.markdown('<div class="sec-title">분위기별 곡 수</div>', unsafe_allow_html=True)
            counts = songs["mood"].value_counts().reindex(list(MOOD_KR))
            fig = go.Figure(go.Bar(x=[MOOD_KR[m] for m in counts.index], y=counts.values,
                                   marker_color=[MOOD_COLOR[m] for m in counts.index],
                                   text=counts.values, textposition="outside"))
            fig.update_layout(yaxis=dict(gridcolor=LINE), xaxis=dict(gridcolor=LINE))
            st.plotly_chart(style_fig(fig, 320))
            st.caption(f"전체 {len(songs):,}곡, 발매 연도 {songs['release_year'].min()}~{songs['release_year'].max()}년")
        with b:
            st.markdown('<div class="sec-title">KNN 분류와 PCA 비교</div>', unsafe_allow_html=True)
            table = run_model_test()
            base = table.iloc[0]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=table["차원 수"].iloc[1:], y=table["정확도"].iloc[1:], mode="lines+markers",
                                     name="KNN + PCA", line=dict(color=GREEN, width=2.5)))
            fig.add_hline(y=base["정확도"], line=dict(color=CORAL, dash="dash"), annotation_text="PCA 없음", annotation_font_color=CORAL)
            fig.update_layout(xaxis=dict(title="주성분 개수", gridcolor=LINE, dtick=1), yaxis=dict(title="정확도", gridcolor=LINE),
                              showlegend=False)
            st.plotly_chart(style_fig(fig, 320))
        best = table.loc[table["정확도"].idxmax()]
        pca_best = table.iloc[1:].loc[table.iloc[1:]["정확도"].idxmax()]
        if best["모델"] == base["모델"]:
            verdict = (f"PCA 없이 7개 특성을 모두 쓴 KNN의 정확도가 <b>{base['정확도']:.1%}</b>로 가장 높았고, "
                       f"PCA를 적용한 모델 중에서는 {int(pca_best['차원 수'])}개 주성분일 때 {pca_best['정확도']:.1%}였어요. "
                       f"차원을 줄이면 계산은 가벼워지지만 이 데이터에서는 정보 손실이 더 컸어요.")
        else:
            verdict = (f"{best['모델']}의 정확도가 <b>{best['정확도']:.1%}</b>로 가장 높았어요. "
                       f"PCA 없는 KNN은 {base['정확도']:.1%}였어요.")
        st.markdown(f'<div class="card brief"><div class="sec-title">모델 테스트 결과</div><ul><li>{verdict}</li>'
                    f'<li>분위기 정답은 밝기(valence)와 에너지(energy)로 만들었기 때문에, 이 두 특성을 뺀 나머지 7개 특성으로 분위기를 맞히는 실험이에요.</li></ul></div>',
                    unsafe_allow_html=True)
        st.dataframe(table.style.format({"정확도": "{:.3f}", "Macro-F1": "{:.3f}"}), hide_index=True)


# ------------------------------------------------------------------
# 6. 실행
# ------------------------------------------------------------------
inject_css()

if not st.session_state.get("auth"):
    render_gate()
    st.stop()

st.session_state.setdefault("stage", "survey")
render_sidebar()
if st.session_state["stage"] == "result":
    render_result()
else:
    render_survey()
