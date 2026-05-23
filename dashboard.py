# ============================================================
#  EcoSight Sustainability Marketing Decision Hub
#  dashboard.py — 多維度人口變數版 (支援動態分群切換)
# ============================================================

import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ── 頁面基本設定 ─────
st.set_page_config(
    page_title="EcoSight 永續行銷決策智庫",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 注入 CSS ────────────────────────────────────────
css_path = os.path.join(os.path.dirname(__file__), "style.css")
try:
    with open(css_path, "r", encoding="utf-8") as f:
        css = f.read()
except FileNotFoundError:
    css = ""
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ============================================================
#  1. 資料讀取與防呆邏輯 (加入新的 4 大人口變數)
# ============================================================

def generate_mock_data():
    """ 生成包含四項人口變數的 Demo 模擬資料 """
    rng = np.random.default_rng(42)
    n_samples = 50 # 稍微增加一點樣本數讓圖比較好看
    user_ids = [f"DEMO_{1001 + i}" for i in range(n_samples)]
    
    # 建立人口變數池
    gender_pool = ["男", "女", "其他"]
    grade_pool = ["大一", "大二", "大三", "大四", "碩士及以上"]
    income_pool = ["5,000 元以下", "5,001 - 8,000 元", "8,001 - 12,000 元", "12,001 元以上"]
    living_pool = ["學校宿舍", "校外租屋", "住家裡"]
    
    data = pd.DataFrame({
        "User_ID": user_ids,
        "gender": rng.choice(gender_pool, n_samples),
        "grade": rng.choice(grade_pool, n_samples),
        "income": rng.choice(income_pool, n_samples),
        "living_status": rng.choice(living_pool, n_samples),
        
        "PC1_Score": np.round(rng.uniform(-3, 3, n_samples), 3), 
        "PC2_Score": np.round(rng.uniform(-2, 2, n_samples), 3), 
        
        # 刻意製造落差比例
        "Intent_Level": rng.choice(["高意圖", "低意圖"], n_samples, p=[0.7, 0.3]),
        "Behavior_Level": rng.choice(["高行為", "低行為"], n_samples, p=[0.4, 0.6]),
        
        "KN": np.round(rng.uniform(3, 5, n_samples), 2),
        "TR": np.round(rng.uniform(2, 5, n_samples), 2),
        "PCE": np.round(rng.uniform(3, 5, n_samples), 2),
        "ATT": np.round(rng.uniform(3, 5, n_samples), 2),
        "SN": np.round(rng.uniform(2, 5, n_samples), 2),
        "PBC": np.round(rng.uniform(2, 5, n_samples), 2),
        "HAB": np.round(rng.uniform(1, 3.5, n_samples), 2),
        "INT": np.round(rng.uniform(3, 5, n_samples), 2),
        "BH": np.round(rng.uniform(1, 4, n_samples), 2),
    })
    return data

# ============================================================
#  2. 側邊欄：檔案上傳與多維度過濾
# ============================================================
with st.sidebar:
    st.markdown("""
    <div class="logo-block">
        <span class="logo-icon">🌿</span>
        <div><div class="logo-name">EcoSight 決策智庫</div><div class="logo-tagline">Sustainability Hub</div></div>
    </div>
    <hr/>
    """, unsafe_allow_html=True)
    
    # 檔案上傳區
    st.markdown('<div class="filter-label">📤 匯入正式數據 (Excel/CSV)</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("上傳您的問卷數據檔", type=["xlsx", "csv"], label_visibility="collapsed")
    
    # 資料處理邏輯 (更新必要欄位)
    required_cols = ["gender", "grade", "income", "living_status", "PC1_Score", "PC2_Score", "Intent_Level", "Behavior_Level", "KN", "HAB"]
    use_mock = True
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_all = pd.read_csv(uploaded_file)
            else:
                df_all = pd.read_excel(uploaded_file)
            
            missing_cols = [col for col in required_cols if col not in df_all.columns]
            if missing_cols:
                st.error(f"❌ 檔案缺少必要欄位: {', '.join(missing_cols)}")
            else:
                st.success("✅ 資料匯入成功！圖表已更新。")
                use_mock = False
        except Exception as e:
            st.error(f"❌ 讀取失敗: {e}")

    if use_mock:
        st.info("💡 目前系統顯示 Demo 模擬數據")
        df_all = generate_mock_data()

    st.markdown("<hr/>", unsafe_allow_html=True)

    # 🌟 新功能：動態切換圖表分群維度
    st.markdown('<div class="filter-label" style="color:#10B981 !important;">📊 選擇圖表分群維度</div>', unsafe_allow_html=True)
    
    dim_map = {
        "gender": "性別", 
        "grade": "年級", 
        "income": "可支配生活費", 
        "living_status": "居住狀況"
    }
    
    group_dim = st.selectbox(
        "決定地圖與桑基圖的分類依據", 
        options=["gender", "grade", "income", "living_status"],
        format_func=lambda x: dim_map[x],
        label_visibility="collapsed"
    )

    st.markdown("<hr/>", unsafe_allow_html=True)

    # 🌟 多維度交叉篩選器
    st.markdown('<div class="filter-label">🎯 鎖定目標客群 (交叉篩選)</div>', unsafe_allow_html=True)
    
    # 建立過濾選項
    sel_gender = st.multiselect("性別", options=df_all["gender"].unique(), default=df_all["gender"].unique())
    sel_grade = st.multiselect("年級", options=df_all["grade"].unique(), default=df_all["grade"].unique())
    sel_income = st.multiselect("生活費", options=df_all["income"].unique(), default=df_all["income"].unique())
    sel_living = st.multiselect("居住狀況", options=df_all["living_status"].unique(), default=df_all["living_status"].unique())

# 進行多維度交叉過濾
df = df_all[
    df_all["gender"].isin(sel_gender) &
    df_all["grade"].isin(sel_grade) &
    df_all["income"].isin(sel_income) &
    df_all["living_status"].isin(sel_living)
].copy()

# ============================================================
#  3. 網頁渲染邏輯
# ============================================================
st.markdown("""
<div style='margin-bottom:1.5rem;'>
    <div class='ecosight-title'>🌿 EcoSight <span class='ecosight-accent'>永續行銷決策智庫</span></div>
    <div class='ecosight-subtitle'>多維度人口特徵交叉分析 · 動態數據匯入模式</div>
</div>
""", unsafe_allow_html=True)

# ── KPI 戰情列 ──
total = len(df)
high_intent = int((df["Intent_Level"] == "高意圖").sum()) if total > 0 else 0
high_behav = int((df["Behavior_Level"] == "高行為").sum()) if total > 0 else 0
churn_candidates = df[(df["Intent_Level"] == "高意圖") & (df["Behavior_Level"] == "低行為")]
churn_rate = (len(churn_candidates) / high_intent * 100) if high_intent > 0 else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("📊 觀測樣本總數", f"{total} 人")
k2.metric("🎯 潛在綠色客群", f"{high_intent} 人", delta=f"意圖顯著", delta_color="normal")
k3.metric("✅ 實際轉換客群", f"{high_behav} 人", delta=f"行為落實", delta_color="normal")
k4.metric("⚠️ 知行落差流失率", f"{churn_rate:.1f}%", delta=f"{len(churn_candidates)} 位流失風險", delta_color="inverse")
st.markdown("<br/>", unsafe_allow_html=True)

if df.empty:
    st.warning("⚠️ 目前的篩選條件下沒有資料，請放寬左側的過濾條件！")
else:
    # ── 圖表區 ──
    col_left, col_right = st.columns([5.5, 4.5], gap="medium")

    # PCA 圖 (動態色彩分群)
    with col_left:
        st.markdown(f"<div class='chart-card'><div class='chart-title'>🔬 綠色 DNA 空間分佈 (依 <b>{dim_map[group_dim]}</b> 分群)</div>", unsafe_allow_html=True)
        fig_scatter = px.scatter(
            df, x="PC1_Score", y="PC2_Score", color=group_dim,
            color_discrete_sequence=px.colors.qualitative.Pastel, # 使用自動適應色系
            hover_data={"User_ID": True, "Intent_Level": True, "Behavior_Level": True},
            labels={"PC1_Score": "第一主成分 (綜合綠色特徵)", "PC2_Score": "第二主成分 (輔助行為特徵)", group_dim: dim_map[group_dim]},
            size_max=14
        )
        fig_scatter.update_traces(marker=dict(size=12, opacity=0.9, line=dict(width=1, color="#1E293B")))
        fig_scatter.add_hline(y=0, line_dash="dot", line_color="#475569", line_width=1.5)
        fig_scatter.add_vline(x=0, line_dash="dot", line_color="#475569", line_width=1.5)
        fig_scatter.update_layout(
            height=430, margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(title="", orientation="h", yanchor="bottom", y=-0.22, xanchor="left", x=0, font=dict(size=12, color="#F8FAFC")),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#334155", zeroline=False, title_font=dict(color="#94A3B8"), tickfont=dict(color="#94A3B8")),
            yaxis=dict(gridcolor="#334155", zeroline=False, title_font=dict(color="#94A3B8"), tickfont=dict(color="#94A3B8"))
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Sankey & Radar
    with col_right:
        st.markdown(f"<div class='chart-card'><div class='chart-title'>🌊 轉換漏斗與流失斷點 (依 <b>{dim_map[group_dim]}</b> 流向)</div>", unsafe_allow_html=True)
        
        # 動態抓取目前選擇維度的唯一值
        dim_values_present = df[group_dim].unique().tolist()
        all_nodes = dim_values_present + ["高意圖", "低意圖", "高行為", "低行為"]
        node_idx = {name: i for i, name in enumerate(all_nodes)}
        
        sources, targets, values, link_colors = [], [], [], []
        
        # 第一層流向：選定的維度 -> 意圖
        for val in dim_values_present:
            for intent in ["高意圖", "低意圖"]:
                cnt = len(df[(df[group_dim] == val) & (df["Intent_Level"] == intent)])
                if cnt > 0:
                    sources.append(node_idx[val])
                    targets.append(node_idx[intent])
                    values.append(cnt)
                    link_colors.append("rgba(59,130,246,0.3)" if intent == "高意圖" else "rgba(100,116,139,0.2)")

        # 第二層流向：意圖 -> 行為
        for intent in ["高意圖", "低意圖"]:
            for behav in ["高行為", "低行為"]:
                cnt = len(df[(df["Intent_Level"] == intent) & (df["Behavior_Level"] == behav)])
                if cnt > 0:
                    sources.append(node_idx[intent])
                    targets.append(node_idx[behav])
                    values.append(cnt)
                    link_colors.append("rgba(16,185,129,0.3)" if behav == "高行為" else "rgba(239,68,68,0.25)")

        fig_sankey = go.Figure(go.Sankey(
            arrangement="snap",
            node=dict(pad=16, thickness=18, line=dict(color="#1E293B", width=1), label=all_nodes, color=["#3B82F6"]*len(all_nodes)),
            link=dict(source=sources, target=targets, value=values, color=link_colors)
        ))
        fig_sankey.update_layout(height=230, margin=dict(l=5, r=5, t=5, b=5), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#F8FAFC"))
        st.plotly_chart(fig_sankey, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # 雷達圖
        st.markdown("<div class='chart-card'><div class='chart-title'>🕸️ 弱點診斷 (TPB 構面剖析)</div>", unsafe_allow_html=True)
        TPB_DIMS = ["KN", "TR", "PCE", "ATT", "SN", "PBC", "HAB", "INT", "BH"]
        avg_scores = df[TPB_DIMS].mean().round(2).tolist()
        fig_radar = go.Figure(go.Scatterpolar(
            r=avg_scores + [avg_scores[0]], theta=["知識", "信任", "感知", "態度", "規範", "控制", "習慣", "意圖", "行為", "知識"],
            fill="toself", fillcolor="rgba(16,185,129,0.25)", line=dict(color="#10B981", width=2.5), marker=dict(size=6, color="#10B981")
        ))
        fig_radar.update_layout(
            height=230, margin=dict(l=30, r=30, t=20, b=10),
            polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=True, range=[1, 5], tickfont=dict(size=10, color="#64748B"), gridcolor="#334155", dtick=1), angularaxis=dict(tickfont=dict(size=11, color="#94A3B8"), gridcolor="#334155")),
            showlegend=False, paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── 決策落地區塊 ──
    st.markdown("<div class='decision-section'><div class='decision-title'>🎯 匯出高流失風險客群名單 (精準行銷投放)</div><div class='decision-subtitle'>系統已為您自動篩選出 <b>「高意圖卻低行為」</b> 的客群。</div>", unsafe_allow_html=True)

    if not churn_candidates.empty:
        # 表格現在會顯示這四個新的人口變數
        st.dataframe(
            churn_candidates[["User_ID", "gender", "grade", "income", "living_status", "Intent_Level", "Behavior_Level", "HAB"]],
            use_container_width=True, height=200
        )
        csv_bytes = churn_candidates.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ 一鍵下載精準行銷名單 (CSV)", data=csv_bytes, file_name="EcoSight_Target_List.csv", mime="text/csv")
    else:
        st.info("ℹ️ 在目前的篩選條件下，無高風險客群。")
    st.markdown("</div>", unsafe_allow_html=True)