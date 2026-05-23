# ============================================================
#  EcoSight 2.0 — 永續行銷決策智庫
#  app.py — 內建 PCA · 啞鈴圖 · 彈性欄位對應
# ============================================================

import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ── 頁面設定 ───────────────────────────────────────────────
st.set_page_config(
    page_title="EcoSight 2.0 永續行銷決策智庫",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS 注入 ─────────────────────────────────────────────────
_css_path = os.path.join(os.path.dirname(__file__), "style.css")
try:
    with open(_css_path, "r", encoding="utf-8") as _f:
        st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# ── 常數 ────────────────────────────────────────────────────
TPB_DIMS = ["KN", "TR", "PCE", "ATT", "SN", "PBC", "HAB", "INT", "BH"]
TPB_LABELS = {
    "KN": "知識 KN", "TR": "信任 TR", "PCE": "感知效能 PCE",
    "ATT": "態度 ATT", "SN": "主觀規範 SN", "PBC": "知覺控制 PBC",
    "HAB": "習慣 HAB", "INT": "意圖 INT", "BH": "行為 BH",
}
# 用於啞鈴圖的構面（排除 BH，以 BH 作為分群依據）
DUMBBELL_DIMS = ["KN", "TR", "PCE", "ATT", "SN", "PBC", "HAB", "INT"]

# ── Session State 初始化 ───────────────────────────────────
_defaults = {"df_raw": None, "col_map": {}, "use_mock": True, "file_name": ""}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================
#  資料生成 / 處理
# ============================================================

def generate_mock_data() -> pd.DataFrame:
    """生成包含 TPB 構面與人口變數的 Demo 資料"""
    rng = np.random.default_rng(42)
    n = 60
    return pd.DataFrame({
        "User_ID":       [f"DEMO_{1001+i}" for i in range(n)],
        "gender":        rng.choice(["男", "女", "其他"], n),
        "grade":         rng.choice(["大一", "大二", "大三", "大四", "碩士及以上"], n),
        "income":        rng.choice(["5,000以下", "5,001–8,000", "8,001–12,000", "12,001以上"], n),
        "living_status": rng.choice(["學校宿舍", "校外租屋", "住家裡"], n),
        "KN":  np.round(rng.uniform(2.5, 5.0, n), 2),
        "TR":  np.round(rng.uniform(2.0, 5.0, n), 2),
        "PCE": np.round(rng.uniform(2.5, 5.0, n), 2),
        "ATT": np.round(rng.uniform(3.0, 5.0, n), 2),
        "SN":  np.round(rng.uniform(2.0, 5.0, n), 2),
        "PBC": np.round(rng.uniform(2.0, 5.0, n), 2),
        "HAB": np.round(rng.uniform(1.0, 3.5, n), 2),
        "INT": np.round(rng.uniform(3.0, 5.0, n), 2),
        "BH":  np.round(rng.uniform(1.0, 4.0, n), 2),
    })


def apply_col_map(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """將使用者的欄位名稱對應到標準名稱"""
    rename = {v: k for k, v in col_map.items() if v and v in df.columns}
    return df.rename(columns=rename)


def compute_groups(df: pd.DataFrame) -> pd.DataFrame:
    """以 INT / BH 欄位的中位數分割產生分群標籤"""
    df = df.copy()
    int_med = df["INT"].median()
    bh_med  = df["BH"].median()
    df["intent_level"] = df["INT"].apply(lambda x: "高意圖" if x >= int_med else "低意圖")
    df["bh_level"]     = df["BH"].apply(lambda x: "高行為" if x >= bh_med else "低行為")
    return df


def compute_pca(df: pd.DataFrame) -> pd.DataFrame:
    """以 9 個 TPB 構面計算 PCA 座標"""
    df = df.copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[TPB_DIMS])
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(scaled)
    df["PC1"] = np.round(components[:, 0], 3)
    df["PC2"] = np.round(components[:, 1], 3)
    return df


def detect_demo_cols(df: pd.DataFrame) -> list[str]:
    """自動偵測非數值、可能是人口變數的欄位"""
    exclude = set(TPB_DIMS + ["User_ID", "PC1", "PC2", "intent_level", "bh_level"])
    return [c for c in df.columns if c not in exclude and df[c].dtype == object]


# ============================================================
#  圖表
# ============================================================

def make_scatter(df: pd.DataFrame, group_col: str) -> go.Figure:
    """PCA 散佈圖（依選定維度著色）"""
    label = group_col
    fig = px.scatter(
        df, x="PC1", y="PC2", color=group_col,
        color_discrete_sequence=px.colors.qualitative.Pastel,
        hover_data=["User_ID", "intent_level", "bh_level"] if "User_ID" in df.columns else None,
        labels={"PC1": "第一主成分（綜合綠色特徵）", "PC2": "第二主成分（輔助行為特徵）", group_col: label},
    )
    fig.update_traces(marker=dict(size=11, opacity=0.9, line=dict(width=1, color="#1E293B")))
    fig.add_hline(y=0, line_dash="dot", line_color="#475569", line_width=1)
    fig.add_vline(x=0, line_dash="dot", line_color="#475569", line_width=1)
    fig.update_layout(
        height=400, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(title="", orientation="h", yanchor="bottom", y=-0.28,
                    xanchor="left", x=0, font=dict(size=11, color="#F8FAFC")),
        xaxis=dict(gridcolor="#334155", zeroline=False,
                   title_font=dict(color="#94A3B8"), tickfont=dict(color="#94A3B8")),
        yaxis=dict(gridcolor="#334155", zeroline=False,
                   title_font=dict(color="#94A3B8"), tickfont=dict(color="#94A3B8")),
    )
    return fig


def make_sankey(df: pd.DataFrame, group_col: str) -> go.Figure:
    """轉換漏斗桑基圖"""
    groups = df[group_col].unique().tolist()
    all_nodes = groups + ["高意圖", "低意圖", "高行為", "低行為"]
    idx = {n: i for i, n in enumerate(all_nodes)}

    sources, targets, values, colors = [], [], [], []
    for g in groups:
        for intent in ["高意圖", "低意圖"]:
            cnt = len(df[(df[group_col] == g) & (df["intent_level"] == intent)])
            if cnt > 0:
                sources.append(idx[g]); targets.append(idx[intent])
                values.append(cnt)
                colors.append("rgba(59,130,246,0.35)" if intent == "高意圖" else "rgba(100,116,139,0.2)")

    for intent in ["高意圖", "低意圖"]:
        for bh in ["高行為", "低行為"]:
            cnt = len(df[(df["intent_level"] == intent) & (df["bh_level"] == bh)])
            if cnt > 0:
                sources.append(idx[intent]); targets.append(idx[bh])
                values.append(cnt)
                colors.append("rgba(16,185,129,0.35)" if bh == "高行為" else "rgba(239,68,68,0.25)")

    node_colors = (["#3B82F6"] * len(groups)
                   + ["#3B82F6", "#64748B", "#10B981", "#EF4444"])
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(pad=14, thickness=16, line=dict(color="#1E293B", width=0.5),
                  label=all_nodes, color=node_colors),
        link=dict(source=sources, target=targets, value=values, color=colors),
    ))
    fig.update_layout(height=400, margin=dict(l=5, r=5, t=5, b=5),
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#F8FAFC", size=12))
    return fig


def make_dumbbell(df: pd.DataFrame) -> go.Figure:
    """啞鈴圖：高行為 vs 低行為群組在各 TPB 構面的均值比較"""
    high = df[df["bh_level"] == "高行為"]
    low  = df[df["bh_level"] == "低行為"]

    dim_labels = [TPB_LABELS[d] for d in DUMBBELL_DIMS]
    high_means = [high[d].mean() if not high.empty else 0 for d in DUMBBELL_DIMS]
    low_means  = [low[d].mean()  if not low.empty  else 0 for d in DUMBBELL_DIMS]
    gaps       = [abs(h - l) for h, l in zip(high_means, low_means)]
    max_gap    = max(gaps) if max(gaps) > 0 else 1

    fig = go.Figure()

    # 連接線（顏色依落差大小：綠→橘→紅）
    for i, (lbl, h_val, l_val, gap) in enumerate(zip(dim_labels, high_means, low_means, gaps)):
        t = gap / max_gap
        r = int(16  + (239 - 16)  * t)
        g = int(185 + (68  - 185) * t)
        b = int(129 + (68  - 129) * t)
        fig.add_trace(go.Scatter(
            x=[l_val, h_val], y=[lbl, lbl], mode="lines",
            line=dict(color=f"rgba({r},{g},{b},0.65)", width=4),
            showlegend=False, hoverinfo="none",
        ))

    # 低行為點（橘紅）
    fig.add_trace(go.Scatter(
        x=low_means, y=dim_labels, mode="markers",
        name="低行為群組",
        marker=dict(color="#EF4444", size=16, symbol="circle",
                    line=dict(color="white", width=1.5)),
        customdata=[[f"{v:.2f}"] for v in low_means],
        hovertemplate="低行為：%{customdata[0]}<extra></extra>",
    ))

    # 高行為點（翡翠綠）
    fig.add_trace(go.Scatter(
        x=high_means, y=dim_labels, mode="markers",
        name="高行為群組",
        marker=dict(color="#10B981", size=16, symbol="circle",
                    line=dict(color="white", width=1.5)),
        customdata=[[f"{v:.2f}"] for v in high_means],
        hovertemplate="高行為：%{customdata[0]}<extra></extra>",
    ))

    fig.update_layout(
        height=380,
        margin=dict(l=10, r=30, t=10, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0.8, 5.5], title="構面平均分（1–5 量表）",
                   gridcolor="#334155", zeroline=False,
                   title_font=dict(color="#94A3B8"), tickfont=dict(color="#94A3B8")),
        yaxis=dict(autorange="reversed", gridcolor="#334155", zeroline=False,
                   tickfont=dict(color="#CBD5E1", size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18,
                    xanchor="center", x=0.5, font=dict(color="#F8FAFC", size=13)),
    )
    return fig


# ============================================================
#  側邊欄
# ============================================================

with st.sidebar:
    st.markdown("""
    <div class="logo-block">
        <span class="logo-icon">🌿</span>
        <div>
            <div class="logo-name">EcoSight 2.0</div>
            <div class="logo-tagline">永續行銷決策智庫</div>
        </div>
    </div><hr/>
    """, unsafe_allow_html=True)

    # ── 資料上傳 ──────────────────────────────────────────
    st.markdown('<div class="filter-label">📤 匯入數據（Excel / CSV）</div>', unsafe_allow_html=True)

    # 提供範本下載
    _template_path = os.path.join(os.path.dirname(__file__), "data", "template.xlsx")
    if os.path.exists(_template_path):
        with open(_template_path, "rb") as _tf:
            st.download_button(
                "⬇️ 下載資料範本", data=_tf.read(),
                file_name="EcoSight_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    uploaded = st.file_uploader("上傳問卷數據", type=["xlsx", "csv"], label_visibility="collapsed")

    if uploaded and uploaded.name != st.session_state.file_name:
        # 讀取新上傳的檔案
        try:
            if uploaded.name.endswith(".csv"):
                raw = pd.read_csv(uploaded)
            else:
                raw = pd.read_excel(uploaded)
            st.session_state.df_raw   = raw
            st.session_state.file_name = uploaded.name
            st.session_state.use_mock  = False
            st.session_state.col_map   = {}  # 清除舊對應
            st.success("✅ 資料匯入成功！")
        except Exception as e:
            st.error(f"❌ 讀取失敗：{e}")

    if st.session_state.use_mock or st.session_state.df_raw is None:
        st.info("💡 目前顯示 Demo 模擬數據")

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── 欄位對應（只在有上傳且缺少標準欄位時顯示）──────
    if not st.session_state.use_mock and st.session_state.df_raw is not None:
        raw_cols = st.session_state.df_raw.columns.tolist()
        missing  = [d for d in TPB_DIMS if d not in raw_cols]

        if missing:
            st.markdown('<div class="filter-label" style="color:#FBBF24">⚙️ 欄位對應（必填）</div>', unsafe_allow_html=True)
            st.caption(f"以下 {len(missing)} 個標準欄位不存在，請手動對應：")
            col_map = dict(st.session_state.col_map)
            for dim in missing:
                col_map[dim] = st.selectbox(
                    f"{TPB_LABELS[dim]}", options=["（略過）"] + raw_cols,
                    key=f"map_{dim}",
                    index=raw_cols.index(col_map.get(dim, "")) + 1
                          if col_map.get(dim) in raw_cols else 0,
                )
            st.session_state.col_map = {k: v for k, v in col_map.items() if v != "（略過）"}
            st.markdown("<hr/>", unsafe_allow_html=True)

    # ── 圖表分群維度選擇 ─────────────────────────────────
    st.markdown('<div class="filter-label" style="color:#10B981">📊 圖表分群依據</div>', unsafe_allow_html=True)
    # 先算出目前資料的人口變數欄位，待下方 df 建立後再回填
    # 預設清單
    _default_demo = ["gender", "grade", "income", "living_status"]
    _demo_label_map = {
        "gender": "性別", "grade": "年級",
        "income": "可支配生活費", "living_status": "居住狀況",
    }

    group_dim = st.selectbox(
        "分群維度", options=_default_demo,
        format_func=lambda x: _demo_label_map.get(x, x),
        label_visibility="collapsed",
    )

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── 多維度篩選 ────────────────────────────────────────
    st.markdown('<div class="filter-label">🎯 交叉篩選目標客群</div>', unsafe_allow_html=True)

    # 先用 mock 資料的預設值，待 df 建立後更新
    if st.session_state.use_mock or st.session_state.df_raw is None:
        _tmp = generate_mock_data()
    else:
        _tmp = apply_col_map(st.session_state.df_raw, st.session_state.col_map)

    sel_gender  = st.multiselect("性別",   _tmp.get("gender", pd.Series([])).unique() if "gender" in _tmp.columns else [],
                                  default=_tmp["gender"].unique().tolist() if "gender" in _tmp.columns else [])
    sel_grade   = st.multiselect("年級",   _tmp.get("grade", pd.Series([])).unique() if "grade" in _tmp.columns else [],
                                  default=_tmp["grade"].unique().tolist() if "grade" in _tmp.columns else [])
    sel_income  = st.multiselect("生活費", _tmp.get("income", pd.Series([])).unique() if "income" in _tmp.columns else [],
                                  default=_tmp["income"].unique().tolist() if "income" in _tmp.columns else [])
    sel_living  = st.multiselect("居住狀況", _tmp.get("living_status", pd.Series([])).unique() if "living_status" in _tmp.columns else [],
                                  default=_tmp["living_status"].unique().tolist() if "living_status" in _tmp.columns else [])


# ============================================================
#  資料處理主流程
# ============================================================

if st.session_state.use_mock or st.session_state.df_raw is None:
    df_src = generate_mock_data()
else:
    df_src = apply_col_map(st.session_state.df_raw.copy(), st.session_state.col_map)

# 過濾缺失 TPB 欄位
valid_dims = [d for d in TPB_DIMS if d in df_src.columns]
has_all_dims = len(valid_dims) == len(TPB_DIMS)

# 交叉篩選
filter_mask = pd.Series([True] * len(df_src), index=df_src.index)
for col, sel in [("gender", sel_gender), ("grade", sel_grade),
                 ("income", sel_income), ("living_status", sel_living)]:
    if col in df_src.columns and sel:
        filter_mask &= df_src[col].isin(sel)

df = df_src[filter_mask].copy()

if has_all_dims and not df.empty:
    df = compute_pca(df)
    df = compute_groups(df)


# ============================================================
#  主頁面渲染
# ============================================================

st.markdown("""
<div style="margin-bottom:1.5rem;">
    <div class="ecosight-title">🌿 EcoSight <span class="ecosight-accent">永續行銷決策智庫</span></div>
    <div class="ecosight-subtitle">TPB 理論框架 · 內建 PCA 降維 · 知行落差視覺化分析</div>
</div>
""", unsafe_allow_html=True)

# ── KPI 卡片 ─────────────────────────────────────────────
total      = len(df)
high_int   = int((df["intent_level"] == "高意圖").sum()) if "intent_level" in df.columns else 0
high_bh    = int((df["bh_level"]     == "高行為").sum()) if "bh_level"     in df.columns else 0
gap_group  = df[(df.get("intent_level", pd.Series()) == "高意圖") &
                (df.get("bh_level", pd.Series())     == "低行為")] if "intent_level" in df.columns else pd.DataFrame()
gap_rate   = len(gap_group) / high_int * 100 if high_int > 0 else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("📊 觀測樣本", f"{total} 人")
k2.metric("🎯 潛在綠色客群", f"{high_int} 人", "高意圖")
k3.metric("✅ 實際轉換客群", f"{high_bh} 人", "高行為")
k4.metric("⚠️ 知行落差流失率", f"{gap_rate:.1f}%",
          f"{len(gap_group)} 位流失風險", delta_color="inverse")

st.markdown("<br/>", unsafe_allow_html=True)

if df.empty:
    st.warning("⚠️ 目前篩選條件下沒有資料，請放寬左側過濾條件。")
elif not has_all_dims:
    missing_list = [d for d in TPB_DIMS if d not in df.columns]
    st.error(f"❌ 缺少 TPB 構面欄位：{', '.join(missing_list)}。請至側邊欄完成欄位對應。")
else:
    # ── Row 1：PCA 散佈 + 桑基圖 ─────────────────────────
    col_l, col_r = st.columns([55, 45], gap="medium")

    with col_l:
        st.markdown(f"""
        <div class="chart-card">
            <div class="chart-title">
                🔬 綠色消費 DNA 空間分佈
                <span class="chart-subtitle">— 依 {_demo_label_map.get(group_dim, group_dim)} 分群</span>
            </div>
        """, unsafe_allow_html=True)
        if group_dim in df.columns:
            st.plotly_chart(make_scatter(df, group_dim), use_container_width=True)
        else:
            st.info(f"欄位「{group_dim}」不存在於資料中。")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown(f"""
        <div class="chart-card">
            <div class="chart-title">
                🌊 轉換漏斗與流失斷點
                <span class="chart-subtitle">— 依 {_demo_label_map.get(group_dim, group_dim)} 流向</span>
            </div>
        """, unsafe_allow_html=True)
        if group_dim in df.columns:
            st.plotly_chart(make_sankey(df, group_dim), use_container_width=True)
        else:
            st.info(f"欄位「{group_dim}」不存在於資料中。")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Row 2：啞鈴圖（知行落差診斷）────────────────────
    st.markdown("""
    <div class="chart-card">
        <div class="chart-title">
            🎯 知行落差診斷（啞鈴圖）
            <span class="chart-subtitle">— 高行為 vs 低行為群組在各 TPB 構面的均值比較</span>
        </div>
        <div class="chart-hint">
            連接線顏色越深（橘紅）代表兩組落差越大，是最需要干預的構面。
        </div>
    """, unsafe_allow_html=True)

    db_col1, db_col2 = st.columns([7, 3])
    with db_col1:
        st.plotly_chart(make_dumbbell(df), use_container_width=True)
    with db_col2:
        st.markdown("<br/>", unsafe_allow_html=True)
        # 落差排名表
        high_grp = df[df["bh_level"] == "高行為"]
        low_grp  = df[df["bh_level"] == "低行為"]
        gap_df = pd.DataFrame({
            "構面": [TPB_LABELS[d] for d in DUMBBELL_DIMS],
            "高行為均值": [high_grp[d].mean() for d in DUMBBELL_DIMS],
            "低行為均值": [low_grp[d].mean()  for d in DUMBBELL_DIMS],
        })
        gap_df["落差"] = (gap_df["高行為均值"] - gap_df["低行為均值"]).round(2)
        gap_df = gap_df.sort_values("落差", ascending=False).reset_index(drop=True)
        gap_df["高行為均值"] = gap_df["高行為均值"].round(2)
        gap_df["低行為均值"] = gap_df["低行為均值"].round(2)
        st.dataframe(
            gap_df[["構面", "落差", "高行為均值", "低行為均值"]],
            use_container_width=True, hide_index=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Row 3：精準行銷名單匯出 ───────────────────────────
    st.markdown("""
    <div class="decision-section">
        <div class="decision-title">📋 高流失風險客群名單（精準行銷投放）</div>
        <div class="decision-subtitle">
            系統已自動篩選出「<b>高意圖卻低行為</b>」的知行落差族群，
            適合作為再行銷、教育型內容的優先觸達對象。
        </div>
    """, unsafe_allow_html=True)

    if not gap_group.empty:
        export_cols = ["User_ID"] if "User_ID" in gap_group.columns else []
        for c in ["gender", "grade", "income", "living_status", "INT", "BH", "HAB"]:
            if c in gap_group.columns:
                export_cols.append(c)
        st.dataframe(gap_group[export_cols], use_container_width=True, height=220, hide_index=True)
        csv_bytes = gap_group.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ 一鍵下載精準行銷名單（CSV）",
            data=csv_bytes, file_name="EcoSight_Target_List.csv", mime="text/csv",
        )
    else:
        st.info("ℹ️ 在目前的篩選條件下，無高流失風險客群。")

    st.markdown("</div>", unsafe_allow_html=True)
