"""
DemandAI — Previsor de Demanda Semanal
Aplicativo web para previsão de demanda com múltiplos métodos,
comparação automática e recomendação gerencial.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# ─────────────────────────────────────────────
# CONFIGURAÇÃO GERAL
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DemandAI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS — ANIMAÇÕES E VISUAL
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;700&display=swap');

/* Reset e base */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Animações */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes shimmer {
    0%   { background-position: -400px 0; }
    100% { background-position: 400px 0; }
}
@keyframes pulse-border {
    0%, 100% { box-shadow: 0 0 0 0 rgba(99,102,241,0); }
    50%       { box-shadow: 0 0 0 4px rgba(99,102,241,0.15); }
}

/* Cabeçalho principal */
.main-header {
    animation: fadeUp 0.6s ease both;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 16px;
    padding: 2.2rem 2.5rem 2rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #6366f1, #8b5cf6, transparent);
}
.main-header::after {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(99,102,241,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.main-header .badge {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.3);
    color: #a5b4fc;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 3px 12px;
    border-radius: 999px;
    margin-bottom: 0.8rem;
}
.main-header h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.1rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 0 0 0.3rem;
    letter-spacing: -0.5px;
}
.main-header p {
    color: #94a3b8;
    font-size: 0.92rem;
    margin: 0;
    font-weight: 400;
}

/* Cards de métricas */
.metric-card {
    animation: fadeUp 0.6s ease both;
    background: #1e293b;
    border: 1px solid rgba(148,163,184,0.1);
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    text-align: center;
    transition: border-color 0.2s, transform 0.2s;
    position: relative;
    overflow: hidden;
}
.metric-card:hover {
    border-color: rgba(99,102,241,0.35);
    transform: translateY(-2px);
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(148,163,184,0.15), transparent);
}
.metric-card .label {
    font-size: 0.7rem;
    color: #64748b;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
}
.metric-card .value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1;
}
.metric-card .sub {
    font-size: 0.75rem;
    color: #475569;
    margin-top: 0.3rem;
    font-weight: 400;
}
.metric-card .accent {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}

/* Títulos de seção */
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.85rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    border-bottom: 1px solid rgba(148,163,184,0.1);
    padding-bottom: 8px;
    margin-bottom: 1rem;
}

/* Card de recomendação */
.rec-card {
    animation: fadeUp 0.5s ease both;
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
    margin-top: 0.8rem;
    border: 1px solid;
    position: relative;
    overflow: hidden;
}
.rec-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
    border-radius: 4px 0 0 4px;
}
.rec-crescente   { background: rgba(16,185,129,0.06);  border-color: rgba(16,185,129,0.2); }
.rec-crescente::before   { background: #10b981; }
.rec-decrescente { background: rgba(245,158,11,0.06);  border-color: rgba(245,158,11,0.2); }
.rec-decrescente::before { background: #f59e0b; }
.rec-estavel     { background: rgba(99,102,241,0.06);  border-color: rgba(99,102,241,0.2); }
.rec-estavel::before     { background: #6366f1; }
.rec-irregular   { background: rgba(239,68,68,0.06);   border-color: rgba(239,68,68,0.2); }
.rec-irregular::before   { background: #ef4444; }

.rec-card h3 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    margin: 0 0 0.8rem 0.8rem;
    color: #e2e8f0;
}
.rec-card p {
    color: #94a3b8;
    line-height: 1.75;
    font-size: 0.9rem;
    margin: 0 0 0 0.8rem;
}
.rec-card li { margin-bottom: 4px; }

/* Badge melhor método */
.best-badge {
    display: inline-block;
    background: rgba(16,185,129,0.15);
    border: 1px solid rgba(16,185,129,0.3);
    color: #34d399;
    font-size: 0.68rem;
    font-weight: 700;
    padding: 2px 10px;
    border-radius: 999px;
    margin-left: 8px;
    vertical-align: middle;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Aviso */
.aviso-card {
    background: rgba(245,158,11,0.06);
    border: 1px solid rgba(245,158,11,0.2);
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    font-size: 0.83rem;
    color: #94a3b8;
    margin-top: 1.2rem;
    line-height: 1.6;
}
.aviso-card strong { color: #fbbf24; }

/* Divisor */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(148,163,184,0.12), transparent);
    margin: 1.5rem 0;
}

/* Animação sequencial dos cards */
.metric-card:nth-child(1) { animation-delay: 0.05s; }
.metric-card:nth-child(2) { animation-delay: 0.1s; }
.metric-card:nth-child(3) { animation-delay: 0.15s; }
.metric-card:nth-child(4) { animation-delay: 0.2s; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FUNÇÕES DE PREVISÃO
# ─────────────────────────────────────────────

def metodo_ingenuo(demandas, n_futuras):
    return [demandas[-1]] * n_futuras

def media_movel_simples(demandas, janela, n_futuras):
    previsoes = []
    historico = list(demandas)
    for _ in range(n_futuras):
        media = np.mean(historico[-janela:])
        previsoes.append(round(media, 2))
        historico.append(media)
    return previsoes

def media_movel_ponderada(demandas, n_futuras):
    pesos = [0.2, 0.3, 0.5]
    historico = list(demandas)
    previsoes = []
    for _ in range(n_futuras):
        ultimas = historico[-3:]
        if len(ultimas) < 3:
            ultimas = [historico[0]] * (3 - len(ultimas)) + ultimas
        valor = sum(p * v for p, v in zip(pesos, ultimas))
        previsoes.append(round(valor, 2))
        historico.append(valor)
    return previsoes

def _calcular_ultima_previsao_exp(demandas, alfa):
    f = demandas[0]
    for d in demandas[1:]:
        f = alfa * d + (1 - alfa) * f
    return f

def suavizacao_exponencial(demandas, alfa, n_futuras):
    f_base = alfa * demandas[-1] + (1 - alfa) * _calcular_ultima_previsao_exp(demandas, alfa)
    return [round(f_base, 2)] * n_futuras

def regressao_linear(demandas, n_futuras):
    n = len(demandas)
    X = np.arange(1, n + 1).reshape(-1, 1)
    y = np.array(demandas)
    modelo = LinearRegression().fit(X, y)
    X_fut = np.arange(n + 1, n + 1 + n_futuras).reshape(-1, 1)
    previsoes = modelo.predict(X_fut)
    return [round(max(0, v), 2) for v in previsoes]

def calcular_mae(real, previsto):
    erros = [abs(r - p) for r, p in zip(real, previsto)]
    return round(np.mean(erros), 2)

def previsao_in_sample(demandas, metodo, janela, alfa):
    n = len(demandas)
    if n < 4:
        return []
    previsoes = []
    for i in range(2, n):
        hist = demandas[:i]
        if metodo == "Ingênuo":
            p = hist[-1]
        elif metodo == "Média Móvel Simples":
            p = np.mean(hist[-janela:])
        elif metodo == "Média Móvel Ponderada":
            pesos = [0.2, 0.3, 0.5]
            ultimas = hist[-3:]
            if len(ultimas) < 3:
                ultimas = [hist[0]] * (3 - len(ultimas)) + list(ultimas)
            p = sum(pw * v for pw, v in zip(pesos, ultimas))
        elif metodo == "Suavização Exponencial":
            f = hist[0]
            for d in hist[1:]:
                f = alfa * d + (1 - alfa) * f
            p = alfa * hist[-1] + (1 - alfa) * f
        elif metodo == "Regressão Linear":
            X = np.arange(1, len(hist) + 1).reshape(-1, 1)
            y = np.array(hist)
            modelo = LinearRegression().fit(X, y)
            p = modelo.predict([[len(hist) + 1]])[0]
        else:
            p = hist[-1]
        previsoes.append(round(max(0, p), 2))
    return previsoes

def classificar_tendencia(demandas):
    n = len(demandas)
    X = np.arange(1, n + 1).reshape(-1, 1)
    y = np.array(demandas)
    modelo = LinearRegression().fit(X, y)
    slope = modelo.coef_[0]
    cv = np.std(demandas) / np.mean(demandas) if np.mean(demandas) != 0 else 0
    if cv > 0.20:
        return "irregular"
    elif slope > np.mean(demandas) * 0.02:
        return "crescente"
    elif slope < -np.mean(demandas) * 0.02:
        return "decrescente"
    else:
        return "estavel"


# ─────────────────────────────────────────────
# CABEÇALHO
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="badge">Administração da Produção</div>
    <h1>DemandAI</h1>
    <p>Previsão de demanda semanal com análise estatística e recomendação gerencial</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Configurações")
    nome_produto = st.text_input("Nome do produto", value="Produto A",
                                 placeholder="Ex.: Camiseta P, Parafuso M6...")

    st.markdown("---")
    st.markdown("#### Série Histórica")
    st.caption("Informe a demanda semanal separada por vírgula.")

    entrada_raw = st.text_area(
        "Demanda por semana (unidades)",
        value="120, 125, 130, 128, 135, 140, 145, 150, 148, 155, 160, 165",
        height=120,
    )

    st.markdown("---")
    st.markdown("#### Horizonte de Previsão")
    n_semanas_futuras = st.slider("Semanas a prever", 1, 12, 4)

    st.markdown("---")
    st.markdown("#### Métodos de Previsão")
    metodos_selecionados = st.multiselect(
        "Selecione os métodos",
        options=["Ingênuo", "Média Móvel Simples", "Média Móvel Ponderada",
                 "Suavização Exponencial", "Regressão Linear"],
        default=["Média Móvel Simples", "Suavização Exponencial", "Regressão Linear"]
    )

    st.markdown("---")
    st.markdown("#### Parâmetros")
    janela_mm = st.slider("Janela — Média Móvel", 2, 6, 3)
    alfa_exp  = st.slider("Alfa — Suavização Exponencial", 0.1, 0.9, 0.3, 0.05)


# ─────────────────────────────────────────────
# PARSE E VALIDAÇÃO
# ─────────────────────────────────────────────
def parse_demandas(texto):
    texto = texto.replace("\n", ",").replace(";", ",")
    partes = [p.strip() for p in texto.split(",") if p.strip()]
    valores = []
    for p in partes:
        try:
            v = float(p)
            if v < 0:
                return None, f"Valor negativo encontrado: {v}. Demanda não pode ser negativa."
            valores.append(v)
        except ValueError:
            return None, f"Valor inválido: '{p}'. Use apenas números."
    return valores, None

demandas, erro_parse = parse_demandas(entrada_raw)

if erro_parse:
    st.error(erro_parse)
    st.stop()

if not demandas:
    st.warning("Informe ao menos 4 semanas de demanda histórica.")
    st.stop()

if len(demandas) < 4:
    st.warning(f"Apenas {len(demandas)} semana(s) informada(s). Recomendamos ao menos 4 para maior confiabilidade.")

if not metodos_selecionados:
    st.info("Selecione ao menos um método de previsão na barra lateral.")
    st.stop()

n = len(demandas)
tendencia = classificar_tendencia(demandas)


# ─────────────────────────────────────────────
# MÉTRICAS
# ─────────────────────────────────────────────
cor_tend = {"crescente": "#10b981", "decrescente": "#f59e0b",
            "estavel": "#6366f1", "irregular": "#ef4444"}
label_tend = {"crescente": "Crescente", "decrescente": "Decrescente",
              "estavel": "Estável", "irregular": "Irregular"}

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Semanas Históricas</div>
        <div class="value">{n}</div>
        <div class="sub">semanas analisadas</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Demanda Média</div>
        <div class="value">{np.mean(demandas):.0f}</div>
        <div class="sub">unidades / semana</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Tendência</div>
        <div class="value" style="font-size:1.2rem;padding-top:6px;">
            <span class="accent" style="background:{cor_tend[tendencia]};width:10px;height:10px;display:inline-block;border-radius:50%;"></span>
            {label_tend[tendencia]}
        </div>
        <div class="sub">baseado em regressão linear</div>
    </div>""", unsafe_allow_html=True)
with col4:
    cv = np.std(demandas) / np.mean(demandas) * 100 if np.mean(demandas) else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Variabilidade (CV)</div>
        <div class="value">{cv:.1f}%</div>
        <div class="sub">coeficiente de variação</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CÁLCULO
# ─────────────────────────────────────────────
resultados = {}
maes = {}

for metodo in metodos_selecionados:
    if metodo == "Ingênuo":
        prev = metodo_ingenuo(demandas, n_semanas_futuras)
    elif metodo == "Média Móvel Simples":
        prev = media_movel_simples(demandas, janela_mm, n_semanas_futuras)
    elif metodo == "Média Móvel Ponderada":
        prev = media_movel_ponderada(demandas, n_semanas_futuras)
    elif metodo == "Suavização Exponencial":
        prev = suavizacao_exponencial(demandas, alfa_exp, n_semanas_futuras)
    elif metodo == "Regressão Linear":
        prev = regressao_linear(demandas, n_semanas_futuras)
    else:
        prev = [demandas[-1]] * n_semanas_futuras
    resultados[metodo] = prev

    if n >= 4:
        in_sample = previsao_in_sample(demandas, metodo, janela_mm, alfa_exp)
        real_trecho = demandas[2:]
        if in_sample and len(in_sample) == len(real_trecho):
            maes[metodo] = calcular_mae(real_trecho, in_sample)

melhor_metodo = min(maes, key=maes.get) if maes else None


# ─────────────────────────────────────────────
# ABAS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["Tabelas", "Grafico", "Comparacao de Metodos", "Recomendacao Gerencial"]
)

# ── TAB 1 ────────────────────────────────────
with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-title">Historico de Demanda</div>', unsafe_allow_html=True)
        df_hist = pd.DataFrame({
            "Semana": [f"Semana {i+1}" for i in range(n)],
            f"{nome_produto} — Demanda Real": [int(d) for d in demandas]
        })
        st.dataframe(df_hist, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown('<div class="section-title">Previsoes Futuras</div>', unsafe_allow_html=True)
        rows = {"Semana": [f"Semana {n+i+1}" for i in range(n_semanas_futuras)]}
        for metodo, prev in resultados.items():
            label = f"{metodo} *" if metodo == melhor_metodo else metodo
            rows[label] = [int(v) for v in prev]
        df_prev = pd.DataFrame(rows)
        st.dataframe(df_prev, use_container_width=True, hide_index=True)
        if melhor_metodo:
            st.caption(f"* Menor erro historico (MAE): {melhor_metodo}")

# ── TAB 2 ────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">Historico e Previsao</div>', unsafe_allow_html=True)

    semanas_hist = list(range(1, n + 1))
    semanas_fut  = list(range(n, n + n_semanas_futuras + 1))

    fig = go.Figure()

    # Área sombreada do histórico
    fig.add_trace(go.Scatter(
        x=semanas_hist, y=demandas,
        fill='tozeroy',
        fillcolor='rgba(99,102,241,0.05)',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False,
        hoverinfo='skip',
    ))

    # Linha histórica
    fig.add_trace(go.Scatter(
        x=semanas_hist, y=demandas,
        mode="lines+markers",
        name="Demanda Real",
        line=dict(color="#6366f1", width=2.5),
        marker=dict(size=7, color="#6366f1", line=dict(color="#818cf8", width=1.5)),
    ))

    cores = ["#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4"]
    for i, (metodo, prev) in enumerate(resultados.items()):
        y_linha = [demandas[-1]] + prev
        label = f"{metodo} (menor MAE)" if metodo == melhor_metodo else metodo
        fig.add_trace(go.Scatter(
            x=semanas_fut, y=y_linha,
            mode="lines+markers",
            name=label,
            line=dict(color=cores[i % len(cores)], width=2, dash="dot"),
            marker=dict(size=6, color=cores[i % len(cores)]),
        ))

    fig.add_vline(
        x=n,
        line_dash="dash",
        line_color="rgba(148,163,184,0.3)",
        line_width=1.5,
        annotation_text="Inicio da previsao",
        annotation_font_color="#64748b",
        annotation_font_size=11,
    )

    fig.update_layout(
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font=dict(color="#94a3b8", family="Inter"),
        title=dict(
            text=f"{nome_produto} — Serie Historica e Previsao",
            font=dict(size=14, color="#e2e8f0", family="Space Grotesk"),
        ),
        xaxis=dict(
            title="Semana",
            tickmode="linear", dtick=1,
            gridcolor="rgba(148,163,184,0.07)",
            color="#64748b",
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            title="Demanda (unidades)",
            gridcolor="rgba(148,163,184,0.07)",
            color="#64748b",
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)", font=dict(size=11),
        ),
        height=430,
        margin=dict(l=10, r=10, t=60, b=10),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1e293b", bordercolor="#334155", font=dict(color="#f1f5f9")),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── TAB 3 ────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">Comparacao entre Metodos</div>', unsafe_allow_html=True)

    if not maes:
        st.info("Informe ao menos 4 semanas de historico para calcular o erro dos metodos.")
    else:
        dados_comp = []
        for metodo in metodos_selecionados:
            prev = resultados[metodo]
            mae = maes.get(metodo)
            dados_comp.append({
                "Metodo": metodo,
                "Previsao Proxima Semana": int(prev[0]),
                "Media das Previsoes": f"{np.mean(prev):.1f}",
                "MAE": f"{mae:.2f}" if mae is not None else "—",
                "Avaliacao": "Menor erro historico" if metodo == melhor_metodo else ""
            })
        st.dataframe(pd.DataFrame(dados_comp), use_container_width=True, hide_index=True)

        st.markdown(f"""
        <div style="background:rgba(16,185,129,0.06);border:1px solid rgba(16,185,129,0.2);
                    border-radius:12px;padding:1rem 1.4rem;margin-top:1rem;">
            <span style="color:#34d399;font-weight:700;font-size:0.9rem;">
                Melhor metodo historico: {melhor_metodo}
            </span><br>
            <span style="color:#64748b;font-size:0.83rem;">
                MAE = {maes[melhor_metodo]:.2f} unidades de erro medio absoluto nas previsoes em amostra.<br>
                O menor erro historico nao garante a melhor previsao futura. Avalie o contexto antes de decidir.
            </span>
        </div>
        """, unsafe_allow_html=True)

        fig_mae = go.Figure(go.Bar(
            x=list(maes.keys()),
            y=list(maes.values()),
            marker_color=["#10b981" if m == melhor_metodo else "#334155" for m in maes.keys()],
            marker_line_color=["#34d399" if m == melhor_metodo else "#475569" for m in maes.keys()],
            marker_line_width=1.5,
            text=[f"{v:.2f}" for v in maes.values()],
            textposition="outside",
            textfont=dict(color="#94a3b8", size=11),
        ))
        fig_mae.update_layout(
            plot_bgcolor="#0f172a",
            paper_bgcolor="#0f172a",
            font=dict(color="#94a3b8", family="Inter"),
            title=dict(text="Erro Medio Absoluto (MAE) por Metodo",
                       font=dict(size=13, color="#e2e8f0", family="Space Grotesk")),
            yaxis=dict(title="MAE (unidades)", gridcolor="rgba(148,163,184,0.07)", color="#64748b"),
            xaxis=dict(color="#64748b"),
            height=300,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_mae, use_container_width=True)

# ── TAB 4 ────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">Recomendacao Gerencial</div>', unsafe_allow_html=True)

    config_rec = {
        "crescente":   ("rec-crescente",   "Demanda em Crescimento"),
        "decrescente": ("rec-decrescente", "Demanda em Queda"),
        "estavel":     ("rec-estavel",     "Demanda Estavel"),
        "irregular":   ("rec-irregular",   "Demanda Irregular"),
    }
    classe_css, titulo_rec = config_rec[tendencia]

    textos = {
        "crescente": f"""
            A demanda por <strong>{nome_produto}</strong> apresenta tendencia de crescimento consistente.<br><br>
            Recomenda-se verificar se a capacidade produtiva atual sera suficiente para atender as proximas semanas.
            Avalie a necessidade de ajuste de turnos, ampliacao de insumos e antecipacao de compras de materia-prima.
            Monitore os pedidos de clientes para confirmar se o crescimento e sustentavel.
        """,
        "decrescente": f"""
            A demanda por <strong>{nome_produto}</strong> apresenta tendencia de queda.<br><br>
            Recomenda-se cautela na producao para evitar excesso de estoque e aumento dos custos de armazenagem.
            Verifique se a queda tem causa temporaria (sazonalidade, promocao encerrada) ou estrutural.
            Envolva a equipe comercial para entender possiveis perdas de clientes ou acirramento da concorrencia.
        """,
        "estavel": f"""
            A demanda por <strong>{nome_produto}</strong> apresenta comportamento relativamente estavel.<br><br>
            A empresa pode usar a previsao como referencia para manter o planejamento de producao.
            Aproveite a estabilidade para otimizar processos, reduzir custos operacionais e ajustar o estoque de seguranca
            com base na variacao observada. Revise periodicamente conforme novos dados forem disponibilizados.
        """,
        "irregular": f"""
            A demanda por <strong>{nome_produto}</strong> apresenta alta variacao entre os periodos analisados.<br><br>
            Recomenda-se investigar fatores externos como sazonalidade, promocoes, eventos ou comportamento dos clientes
            antes de tomar decisoes de producao. Mantenha estoque de seguranca maior para absorver oscilacoes imprevistas
            e evite basear decisoes em apenas um metodo de previsao.
        """,
    }

    st.markdown(f"""
    <div class="rec-card {classe_css}">
        <h3>{titulo_rec}</h3>
        <p>{textos[tendencia]}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Resumo das Previsoes</div>', unsafe_allow_html=True)

    col_r1, col_r2, col_r3 = st.columns(3)
    todas_prevs = [v for lst in resultados.values() for v in lst]
    primeira_prev = list(resultados.values())[0][0] if resultados else 0

    with col_r1:
        st.metric("Previsao — Proxima Semana", f"{int(primeira_prev):,} un.".replace(",", "."))
    with col_r2:
        st.metric("Maior Previsao no Periodo", f"{int(max(todas_prevs)):,} un.".replace(",", "."))
    with col_r3:
        st.metric("Menor Previsao no Periodo", f"{int(min(todas_prevs)):,} un.".replace(",", "."))

    st.markdown("""
    <div class="aviso-card">
        <strong>Atencao:</strong> Previsao nao e certeza. Ela e uma estimativa baseada em dados historicos
        e pode ser afetada por mudancas de mercado, sazonalidade ou eventos imprevistos.
        O gestor deve analisar o contexto antes de tomar decisoes de producao.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# RODAPÉ
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#334155;font-size:0.75rem;
            padding:1.2rem 0;border-top:1px solid rgba(148,163,184,0.08);">
    DemandAI &nbsp;·&nbsp; Administracao da Producao &nbsp;·&nbsp;
    Python + Streamlit + IA Generativa
</div>
""", unsafe_allow_html=True)
