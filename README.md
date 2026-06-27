# DemandAI

Aplicativo web de previsão de demanda semanal, desenvolvido em Python com Streamlit, para apoio ao planejamento da produção em pequenas e médias empresas.

Projeto acadêmico da disciplina de Administração da Produção, desenvolvido com apoio de IA generativa (metodologia *vibe coding*).

## Acesse o app

[Abrir DemandAI]([ttps://share.streamlit.io](https://demandai-gacgqh65gqqfrxkepajeqx.streamlit.app/#demand-ai))

## O que o app faz

A partir do histórico de demanda semanal de um produto, o DemandAI calcula previsões para as próximas semanas usando múltiplos métodos estatísticos, compara o desempenho de cada método e gera uma recomendação gerencial em linguagem simples.

## Funcionalidades

- Entrada manual de demandas históricas semanais
- Previsão configurável de 1 a 12 semanas futuras
- Cinco métodos de previsão: Ingênuo, Média Móvel Simples, Média Móvel Ponderada, Suavização Exponencial e Regressão Linear
- Gráfico interativo com histórico e previsão
- Comparação automática de métodos pelo erro médio absoluto (MAE)
- Recomendação gerencial conforme a tendência da demanda (crescente, decrescente, estável ou irregular)
- Validação de dados de entrada (valores negativos, dados insuficientes, etc.)

## Tecnologias utilizadas

- Python
- Streamlit
- Pandas e NumPy
- Plotly
- Scikit-learn

## Como executar localmente

```bash
git clone https://github.com/RaissaKelly00/DemandAI.git
cd DemandAI
pip install -r requirements.txt
python -m streamlit run app.py
```

O app abre automaticamente em `http://localhost:8501`.

## Aviso

As previsões geradas são estimativas baseadas em dados históricos e não substituem a análise de contexto do gestor de produção. Fatores como sazonalidade, promoções e eventos de mercado podem alterar significativamente a demanda real.

## Disciplina

Administração da Produção — Análise e Desenvolvimento de Sistemas (ADS)
