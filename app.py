"""
Previsor de Demanda Semanal
Aplicativo web para previsão de demanda com múltiplos métodos,
comparação automática e recomendação gerencial.
Desenvolvido para a disciplina de Administração da Produção.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# ─────────────────────────────────────────────
# CONFIGURAÇÃO GERAL DA PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Previsor de Demanda Semanal",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS PERSONALIZADO
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Paleta principal */
    :root {
        --azul:   #1a56db;
        --verde:  #057a55;
        --amber:  #d97706;
        --vermelho: #e02424;
        --cinza:  #6b7280;
    }

    /* Cabeçalho principal */
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #1a56db 60%, #3b82f6 100%);
        padding: 2rem 2.5rem 1.8rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { margin: 0; font-size: 2rem; font-weight: 800; letter-spacing: -0.5px; }
    .main-header p  { margin: 0.4rem 0 0; opacity: 0.85; font-size: 1rem; }

    /* Cartões de métricas */
    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.1rem 1.4rem;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,.06);
    }
    .metric-card .label { font-size: 0.78rem; color: var(--cinza); font-weight: 600; text-transform: uppercase; letter-spacing: .05em; }
    .metric-card .value { font-size: 1.8rem; font-weight: 800; color: #111827; }
    .metric-card .sub   { font-size: 0.8rem; color: var(--cinza); margin-top: 2px; }

    /* Cartão de recomendação */
    .rec-card {
        border-radius: 12px;
        padding: 1.3rem 1.6rem;
        margin-top: 1rem;
        border-left: 5px solid;
    }
    .rec-crescente   { background: #ecfdf5; border-color: var(--verde);    color: #064e3b; }
    .rec-decrescente { background: #fff7ed; border-color: var(--amber);    color: #78350f; }
    .rec-estavel     { background: #eff6ff; border-color: var(--azul);     color: #1e3a8a; }
    .rec-irregular   { background: #fef2f2; border-color: var(--vermelho); color: #7f1d1d; }

    /* Badge do melhor método */
    .best-badge {
        display: inline-block;
        background: #d1fae5;
        color: #065f46;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 2px 10px;
        border-radius: 999px;
        margin-left: 8px;
        vertical-align: middle;
    }

    /* Cabeçalho de seção */
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #111827;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 6px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FUNÇÕES DE PREVISÃO
# ─────────────────────────────────────────────

def metodo_ingenuo(demandas: list, n_futuras: int) -> list:
    """Previsão ingênua: repete o último valor observado."""
    return [demandas[-1]] * n_futuras


def media_movel_simples(demandas: list, janela: int, n_futuras: int) -> list:
    """Média das últimas 'janela' semanas, projetada para o futuro."""
    previsoes = []
    historico = list(demandas)
    for _ in range(n_futuras):
        media = np.mean(historico[-janela:])
        previsoes.append(round(media, 2))
        historico.append(media)
    return previsoes


def media_movel_ponderada(demandas: list, n_futuras: int) -> list:
    """Média ponderada das últimas 3 semanas (pesos 0.2, 0.3, 0.5)."""
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


def suavizacao_exponencial(demandas: list, alfa: float, n_futuras: int) -> list:
    """
    F(t+1) = alfa × D(t) + (1 - alfa) × F(t)
    Quanto maior o alfa, mais reativo a mudanças recentes.
    """
    f = demandas[0]
    for d in demandas[1:]:
        f = alfa * d + (1 - alfa) * f
    previsoes = []
    for _ in range(n_futuras):
        f = alfa * f + (1 - alfa) * f  # sem novo dado real: repete a última previsão
        previsoes.append(round(f, 2))
    # Corrigindo: para semanas futuras sem dados reais, a previsão é constante
    # (o loop acima não altera f). Retornamos a previsão pós-histórico repetida.
    f_base = alfa * demandas[-1] + (1 - alfa) * _calcular_ultima_previsao_exp(demandas, alfa)
    return [round(f_base, 2)] * n_futuras


def _calcular_ultima_previsao_exp(demandas: list, alfa: float) -> float:
    """Calcula a previsão acumulada após todo o histórico."""
    f = demandas[0]
    for d in demandas[1:]:
        f = alfa * d + (1 - alfa) * f
    return f


def regressao_linear(demandas: list, n_futuras: int) -> list:
    """Ajusta reta de tendência e projeta semanas futuras."""
    n = len(demandas)
    X = np.arange(1, n + 1).reshape(-1, 1)
    y = np.array(demandas)
    modelo = LinearRegression().fit(X, y)
    X_fut = np.arange(n + 1, n + 1 + n_futuras).reshape(-1, 1)
    previsoes = modelo.predict(X_fut)
    return [round(max(0, v), 2) for v in previsoes]


def calcular_mae(real: list, previsto: list) -> float:
    """Erro Médio Absoluto entre valores reais e previstos."""
    erros = [abs(r - p) for r, p in zip(real, previsto)]
    return round(np.mean(erros), 2)


def previsao_in_sample(demandas: list, metodo: str, janela: int, alfa: float) -> list:
    """
    Gera previsões dentro da amostra (in-sample) para calcular MAE.
    Usa as últimas N semanas como 'teste', prevendo uma a uma.
    """
    n = len(demandas)
    if n < 4:
        return []
    previsoes = []
    for i in range(2, n):  # começa a prever a partir do 3º ponto
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


def classificar_tendencia(demandas: list) -> str:
    """Classifica a tendência da demanda com base na regressão linear."""
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
    <h1>📦 Previsor de Demanda Semanal</h1>
    <p>Aplicativo de apoio ao planejamento da produção · Administração da Produção</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# BARRA LATERAL — ENTRADAS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configurações")
    nome_produto = st.text_input("Nome do produto", value="Produto A",
                                 placeholder="Ex.: Camiseta P, Parafuso M6…")

    st.markdown("---")
    st.markdown("#### 📋 Demandas históricas semanais")
    st.caption("Informe a demanda de cada semana, separada por vírgula ou quebra de linha.")

    entrada_raw = st.text_area(
        "Demandas (unidades por semana)",
        value="120, 125, 130, 128, 135, 140, 145, 150, 148, 155, 160, 165",
        height=130,
        help="Mínimo 4 semanas. Exemplo: 100, 110, 120, 130"
    )

    st.markdown("---")
    st.markdown("#### 🔮 Previsão futura")
    n_semanas_futuras = st.slider("Semanas a prever", min_value=1, max_value=12, value=4)

    st.markdown("---")
    st.markdown("#### 🔬 Métodos de previsão")
    metodos_selecionados = st.multiselect(
        "Selecione um ou mais métodos",
        options=["Ingênuo", "Média Móvel Simples", "Média Móvel Ponderada",
                 "Suavização Exponencial", "Regressão Linear"],
        default=["Média Móvel Simples", "Suavização Exponencial", "Regressão Linear"]
    )

    janela_mm = st.slider("Janela da Média Móvel (semanas)", 2, 6, 3,
                          help="Quantidade de semanas usadas no cálculo da média móvel.")
    alfa_exp = st.slider("Alfa — Suavização Exponencial", 0.1, 0.9, 0.3, 0.05,
                         help="0.1 = mais suave (reage pouco); 0.9 = muito reativo a mudanças.")

    st.markdown("---")
    calcular = st.button("▶ Calcular Previsão", type="primary", use_container_width=True)

# ─────────────────────────────────────────────
# PROCESSAMENTO DOS DADOS
# ─────────────────────────────────────────────

# Parseia a entrada
def parse_demandas(texto: str):
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
            return None, f"Valor inválido encontrado: '{p}'. Use apenas números."
    return valores, None


demandas, erro_parse = parse_demandas(entrada_raw)

# ─────────────────────────────────────────────
# VALIDAÇÕES
# ─────────────────────────────────────────────
if erro_parse:
    st.error(f"⚠️ {erro_parse}")
    st.stop()

if not demandas:
    st.warning("Informe ao menos 4 semanas de demanda histórica.")
    st.stop()

if len(demandas) < 4:
    st.warning(f"⚠️ Você informou apenas {len(demandas)} semana(s). Recomendamos ao menos 4 semanas para previsões confiáveis.")

if not metodos_selecionados:
    st.info("Selecione ao menos um método de previsão na barra lateral.")
    st.stop()

n = len(demandas)

# ─────────────────────────────────────────────
# MÉTRICAS RESUMO DO HISTÓRICO
# ─────────────────────────────────────────────
tendencia = classificar_tendencia(demandas)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Semanas históricas</div>
        <div class="value">{n}</div>
        <div class="sub">semanas</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Demanda média</div>
        <div class="value">{np.mean(demandas):.0f}</div>
        <div class="sub">unidades/semana</div>
    </div>""", unsafe_allow_html=True)
with col3:
    icone_tend = {"crescente": "📈", "decrescente": "📉", "estavel": "➡️", "irregular": "〰️"}
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Tendência</div>
        <div class="value">{icone_tend[tendencia]}</div>
        <div class="sub">{tendencia.capitalize()}</div>
    </div>""", unsafe_allow_html=True)
with col4:
    coef_var = np.std(demandas) / np.mean(demandas) * 100 if np.mean(demandas) else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Variabilidade (CV)</div>
        <div class="value">{coef_var:.1f}%</div>
        <div class="sub">coef. de variação</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CÁLCULO DAS PREVISÕES
# ─────────────────────────────────────────────
resultados = {}   # metodo -> lista de previsões futuras
maes = {}         # metodo -> MAE in-sample

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

    # MAE in-sample (precisa de ≥4 dados)
    if n >= 4:
        in_sample = previsao_in_sample(demandas, metodo, janela_mm, alfa_exp)
        real_trecho = demandas[2:]  # alinhado com in_sample (começa em i=2)
        if in_sample and len(in_sample) == len(real_trecho):
            maes[metodo] = calcular_mae(real_trecho, in_sample)

melhor_metodo = min(maes, key=maes.get) if maes else None

# ─────────────────────────────────────────────
# TAB 1: TABELAS  |  TAB 2: GRÁFICO  |  TAB 3: COMPARAÇÃO  |  TAB 4: RECOMENDAÇÃO
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📋 Tabelas", "📊 Gráfico", "⚖️ Comparação de Métodos", "🏭 Recomendação Gerencial"]
)

# ── TAB 1: TABELAS ───────────────────────────
with tab1:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">📅 Histórico de Demanda</div>', unsafe_allow_html=True)
        df_hist = pd.DataFrame({
            "Semana": [f"Semana {i+1}" for i in range(n)],
            f"Demanda Real — {nome_produto}": [f"{int(d):,}".replace(",", ".") for d in demandas]
        })
        st.dataframe(df_hist, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown('<div class="section-title">🔮 Previsões Futuras</div>', unsafe_allow_html=True)
        rows = {"Semana": [f"Semana {n+i+1}" for i in range(n_semanas_futuras)]}
        for metodo, prev in resultados.items():
            label = metodo
            if metodo == melhor_metodo:
                label = f"{metodo} ★"
            rows[label] = [f"{int(v):,}".replace(",", ".") for v in prev]
        df_prev = pd.DataFrame(rows)
        st.dataframe(df_prev, use_container_width=True, hide_index=True)
        if melhor_metodo:
            st.caption(f"★ Melhor método pelo menor erro médio histórico (MAE).")

# ── TAB 2: GRÁFICO ───────────────────────────
with tab2:
    st.markdown('<div class="section-title">📊 Histórico e Previsão</div>', unsafe_allow_html=True)

    semanas_hist = list(range(1, n + 1))
    semanas_fut  = list(range(n, n + n_semanas_futuras + 1))  # começa em n para conectar

    fig = go.Figure()

    # Linha do histórico
    fig.add_trace(go.Scatter(
        x=semanas_hist, y=demandas,
        mode="lines+markers",
        name="Demanda Real",
        line=dict(color="#1a56db", width=2.5),
        marker=dict(size=7, color="#1a56db"),
    ))

    # Linhas de previsão por método
    cores = ["#e74c3c", "#27ae60", "#f39c12", "#8e44ad", "#16a085"]
    for i, (metodo, prev) in enumerate(resultados.items()):
        y_linha = [demandas[-1]] + prev  # conecta ao último ponto real
        label = metodo if metodo != melhor_metodo else f"{metodo} ★ (menor MAE)"
        fig.add_trace(go.Scatter(
            x=semanas_fut, y=y_linha,
            mode="lines+markers",
            name=label,
            line=dict(color=cores[i % len(cores)], width=2, dash="dot"),
            marker=dict(size=6),
        ))

    # Linha vertical separando histórico de previsão
    fig.add_vline(x=n, line_dash="dash", line_color="#9ca3af",
                  annotation_text="Início da previsão", annotation_position="top right")

    fig.update_layout(
        title=dict(text=f"Previsão de Demanda — {nome_produto}", font=dict(size=16, color="#111827")),
        xaxis=dict(title="Semana", tickmode="linear", dtick=1, gridcolor="#f3f4f6"),
        yaxis=dict(title="Demanda (unidades)", gridcolor="#f3f4f6"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=450,
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("A linha vertical separa dados históricos (à esquerda) das previsões futuras (à direita).")

# ── TAB 3: COMPARAÇÃO ────────────────────────
with tab3:
    st.markdown('<div class="section-title">⚖️ Comparação entre Métodos</div>', unsafe_allow_html=True)

    if not maes:
        st.info("Informe ao menos 4 semanas de histórico para calcular o erro dos métodos.")
    else:
        dados_comp = []
        for metodo in metodos_selecionados:
            prev = resultados[metodo]
            media_prev = np.mean(prev)
            mae = maes.get(metodo, None)
            dados_comp.append({
                "Método": metodo,
                "Previsão Próx. Semana": f"{int(prev[0]):,}".replace(",", "."),
                "Média das Previsões": f"{media_prev:.1f}",
                "MAE (Erro Médio Absoluto)": f"{mae:.2f}" if mae is not None else "—",
                "Melhor?": "✅ Menor erro" if metodo == melhor_metodo else ""
            })
        df_comp = pd.DataFrame(dados_comp)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

        st.markdown(f"""
        <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;padding:1rem 1.3rem;margin-top:1rem;">
            <b>📌 Melhor método histórico: {melhor_metodo}</b><br>
            MAE = {maes[melhor_metodo]:.2f} unidades de erro médio nas previsões dentro da amostra.<br>
            <small>⚠️ Atenção: o menor erro histórico <b>não garante</b> a melhor previsão futura. O gestor deve analisar o contexto antes de decidir.</small>
        </div>
        """, unsafe_allow_html=True)

        # Gráfico de barras do MAE
        fig_mae = go.Figure(go.Bar(
            x=list(maes.keys()),
            y=list(maes.values()),
            marker_color=["#22c55e" if m == melhor_metodo else "#93c5fd" for m in maes.keys()],
            text=[f"{v:.2f}" for v in maes.values()],
            textposition="outside"
        ))
        fig_mae.update_layout(
            title="Erro Médio Absoluto (MAE) por Método",
            yaxis_title="MAE (unidades)",
            xaxis_title="Método",
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=320,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_mae, use_container_width=True)
        st.caption("Barras verdes indicam o método com menor erro histórico.")

# ── TAB 4: RECOMENDAÇÃO GERENCIAL ────────────
with tab4:
    st.markdown('<div class="section-title">🏭 Recomendação Gerencial</div>', unsafe_allow_html=True)

    # Determina classe CSS e ícone
    config_rec = {
        "crescente":   ("rec-crescente",   "📈", "Demanda em Crescimento"),
        "decrescente": ("rec-decrescente", "📉", "Demanda em Queda"),
        "estavel":     ("rec-estavel",     "➡️", "Demanda Estável"),
        "irregular":   ("rec-irregular",   "〰️", "Demanda Irregular"),
    }
    classe_css, icone, titulo_rec = config_rec[tendencia]

    # Textos gerenciais
    textos = {
        "crescente": f"""
            A demanda por <b>{nome_produto}</b> apresenta tendência de <b>crescimento</b>.<br><br>
            ✔ Recomenda-se verificar se a <b>capacidade produtiva atual</b> será suficiente para atender as
            semanas previstas.<br>
            ✔ Avaliar necessidade de horas extras, novos turnos ou ampliação de insumos.<br>
            ✔ Planejar com antecedência a compra de matéria-prima para evitar rupturas de estoque.<br>
            ✔ Monitorar os pedidos de clientes para confirmar se o crescimento é consistente.
        """,
        "decrescente": f"""
            A demanda por <b>{nome_produto}</b> apresenta tendência de <b>queda</b>.<br><br>
            ✔ Recomenda-se <b>cautela na produção</b> para evitar excesso de estoque e custo de armazenagem.<br>
            ✔ Verificar se a queda tem causa temporária (promoção encerrada, sazonalidade) ou estrutural.<br>
            ✔ Avaliar redução de lotes de produção para ajustar à demanda observada.<br>
            ✔ Conversar com equipe comercial para entender se há perda de clientes ou concorrência.
        """,
        "estavel": f"""
            A demanda por <b>{nome_produto}</b> apresenta comportamento <b>relativamente estável</b>.<br><br>
            ✔ A empresa pode usar a previsão como referência para manter o planejamento de produção.<br>
            ✔ Manter estoque de segurança proporcional à variação observada.<br>
            ✔ Aproveitar a estabilidade para <b>otimizar processos</b> e reduzir custos operacionais.<br>
            ✔ Revisar periodicamente à medida que novos dados forem disponíveis.
        """,
        "irregular": f"""
            A demanda por <b>{nome_produto}</b> apresenta <b>alta variação</b>.<br><br>
            ✔ Recomenda-se analisar fatores externos: promoções, sazonalidade, eventos ou
            comportamento dos clientes.<br>
            ✔ Evitar decisões de produção baseadas apenas em um método de previsão.<br>
            ✔ Manter estoque de segurança maior para absorver oscilações imprevistas.<br>
            ✔ Investigar causas da irregularidade antes de expandir ou reduzir capacidade.
        """
    }

    st.markdown(f"""
    <div class="rec-card {classe_css}">
        <h3 style="margin:0 0 .6rem;">{icone} {titulo_rec}</h3>
        <p style="margin:0;line-height:1.7;">{textos[tendencia]}</p>
    </div>
    """, unsafe_allow_html=True)

    # Resumo numérico
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📌 Resumo das Previsões")
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        primeira_prev = list(resultados.values())[0][0] if resultados else 0
        st.metric("Previsão — próxima semana",
                  f"{int(primeira_prev):,} un.".replace(",", "."),
                  help="Primeiro método selecionado")
    with col_r2:
        todas_prevs = [v for lst in resultados.values() for v in lst]
        st.metric("Maior previsão no período",
                  f"{int(max(todas_prevs)):,} un.".replace(",", "."))
    with col_r3:
        st.metric("Menor previsão no período",
                  f"{int(min(todas_prevs)):,} un.".replace(",", "."))

    st.markdown("""
    <br>
    <div style="background:#fefce8;border:1px solid #fde68a;border-radius:8px;padding:.9rem 1.2rem;font-size:.85rem;color:#78350f;">
        ⚠️ <b>Importante:</b> Previsão <b>não é certeza</b>. Ela é uma estimativa baseada em dados históricos e 
        pode ser afetada por mudanças de mercado, sazonalidade, eventos imprevistos ou comportamento 
        dos clientes. O gestor deve analisar o contexto antes de tomar decisões de produção.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# RODAPÉ
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#9ca3af;font-size:.78rem;padding:1rem 0;border-top:1px solid #e5e7eb;">
    Previsor de Demanda Semanal · Administração da Produção · Desenvolvido com Python + Streamlit + IA Generativa
</div>
""", unsafe_allow_html=True)
