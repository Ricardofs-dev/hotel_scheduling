"""
app.py — Housekeeping Scheduler
================================
Protótipo de apoio à decisão para escalonamento de equipas de Housekeeping.

O utilizador introduz inputs operacionais da semana (taxa de ocupação,
check-ins, check-outs, evento especial) e a app:
  1. Constrói internamente as features necessárias para o modelo de previsão
  2. Prevê N_min e N_ideal por (dia × turno × função)
  3. Corre o modelo de otimização híbrido (internos + externos)
  4. Apresenta a escala semanal gerada e os KPIs operacionais/económicos

Estrutura de pastas esperada:
    hotel_scheduling/
    ├── app/
    │   └── app.py           ← este ficheiro
    ├── src/
    │   ├── optimizer.py
    │   └── input_builder.py
    ├── data/
    │   └── synthetic/       ← CSVs do dataset
    └── outputs/             ← ficheiros .pkl (gb_model, gb_config, ...)

Executar com:
    streamlit run app/app.py
"""

import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import date, timedelta

import streamlit as st

# ── Paths relativos à raiz do projecto ───────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT_DIR / "data" / "synthetic"   # CSVs do dataset
OUTPUTS_DIR = ROOT_DIR / "outputs"              # ficheiros .pkl exportados pelo Bloco 3

sys.path.insert(0, str(ROOT_DIR / "src"))
from optimizer     import (
    preparar_parametros, construir_modelo, resolver,
    extrair_escala, calcular_kpis, mostrar_escala,
)
from input_builder import (
    construir_semana_operacional, gerar_features_previsao,
)


# ════════════════════════════════════════════════════════════════════════════
# Cache de recursos (carregado uma vez por sessão)
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="A carregar modelos de previsão...")
def carregar_modelos():
    """Carrega os modelos .pkl do OUTPUTS_DIR (exportados pelo Bloco 3)."""
    gb_min   = joblib.load(OUTPUTS_DIR / "gb_model.pkl")
    gb_ideal = joblib.load(OUTPUTS_DIR / "gb_model_v2.pkl")
    feats    = joblib.load(OUTPUTS_DIR / "gb_features.pkl")
    return gb_min, gb_ideal, feats


# ════════════════════════════════════════════════════════════════════════════
# Layout
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title = "Housekeeping Scheduler",
    page_icon  = "🏨",
    layout     = "wide",
)

st.title("🏨 Housekeeping Scheduler")
st.caption(
    "Protótipo de apoio à decisão para escalonamento de equipas — "
    "Pós-Graduação Data Science · ISAG · 2026"
)


# ════════════════════════════════════════════════════════════════════════════
# Sidebar: inputs operacionais
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("📋 Inputs operacionais")

    # ── Data ─────────────────────────────────────────────────────────────────
    st.subheader("Semana a planear")

    # Gerar todas as segundas-feiras válidas de 2026 (52 semanas)
    _todas_segundas = [
        date(2026, 1, 5) + timedelta(weeks=i) for i in range(52)
    ]
    # Sugerir a semana mais próxima da data actual
    hoje       = date.today()
    _idx_default = min(
        range(len(_todas_segundas)),
        key=lambda i: abs((_todas_segundas[i] - hoje).days)
    )
    semana_inicio = st.selectbox(
        "Segunda-feira da semana a planear",
        options  = _todas_segundas,
        index    = _idx_default,
        format_func = lambda d: f"{d.strftime('%d/%m/%Y')}  —  semana {d.strftime('%W')}",
        help     = "Seleccionar a semana de planeamento. Apenas semanas completas de 2026.",
    )
    semana_fim = semana_inicio + timedelta(days=6)
    st.success(f"📅 **{semana_inicio.strftime('%d/%m/%Y')}** (Seg) "
               f"a **{semana_fim.strftime('%d/%m/%Y')}** (Dom)")

    st.divider()

    # ── Ocupação ──────────────────────────────────────────────────────────────
    st.subheader("Ocupação prevista")

    taxa_ocu = st.slider(
        "Taxa de ocupação — dias de semana (%)",
        min_value = 0,
        max_value = 100,
        value     = 65,
        step      = 5,
        help      = (
            "Taxa base de ocupação aplicada de Segunda a Quinta. "
            "A app aplica ajustamentos progressivos no final da semana: "
            "Sexta +5%, Sábado +15%, Domingo +10%."
        ),
    )

    col1, col2 = st.columns(2)
    n_checkins  = col1.number_input(
        "Check-ins (total semana)",
        min_value=0, max_value=500, value=80, step=5,
        help="Total de check-ins esperados para os 7 dias. Distribuídos internamente com mais peso Qui–Sáb.",
    )
    n_checkouts = col2.number_input(
        "Check-outs (total semana)",
        min_value=0, max_value=500, value=75, step=5,
        help="Total de check-outs esperados para os 7 dias. Distribuídos internamente com mais peso Seg–Ter.",
    )

    evento = st.checkbox(
        "Evento especial na semana",
        value=False,
        help="Eventos (congressos, feiras, etc.) podem aumentar a necessidade de cobertura.",
    )

    st.divider()

    # ── Parâmetros avançados ──────────────────────────────────────────────────
    with st.expander("⚙️ Parâmetros avançados (pesos da função objetivo)"):
        st.caption(
            "Os pesos controlam o trade-off entre objectivos de qualidade da escala. "
            "Os valores recomendados resultam da análise de sensibilidade do projecto."
        )
        w1 = st.slider("w1 — Preferências de turno",  1, 15,  5,
                       help="Penaliza alocações em turnos não preferidos pelos colaboradores.")
        w2 = st.slider("w2 — Cobertura ideal",         1, 20, 10,
                       help="Penaliza slots abaixo do número ideal de colaboradores (S4).")
        w3 = st.slider("w3 — Equidade de carga",       1, 10,  2,
                       help="Penaliza desequilíbrios na distribuição de turnos (S3).")

    st.divider()
    correr = st.button("🚀 Gerar escala", type="primary", use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# Estado inicial — antes de clicar em "Gerar escala"
# ════════════════════════════════════════════════════════════════════════════

if not correr:
    st.info(
        "👈 Preenche os inputs na barra lateral e clica em **Gerar escala** "
        "para gerar a escala semanal optimizada."
    )
    with st.expander("ℹ️ Como funciona"):
        st.markdown(
            """
            **Pipeline:**
            1. A app constrói as features operacionais para os 7 dias da semana
            2. O modelo Gradient Boosting prevê o número mínimo e ideal de colaboradores
               por dia × turno × função
            3. O modelo de otimização híbrido gera a escala, usando internos primeiro
               e activando externos apenas quando necessário
            4. São apresentados a escala e os KPIs operacionais e económicos

            **Taxa de ocupação:** taxa base aplicada de Segunda a Quinta,
            com ajustamentos progressivos: Sexta +5%, Sábado +15%, Domingo +10%.

            **Semana de planeamento:** apenas semanas completas de 2026 estão disponíveis.
            Feriados nacionais são detectados automaticamente na construção das features.
            """
        )
    st.stop()


# ════════════════════════════════════════════════════════════════════════════
# Pipeline — executar ao clicar em "Gerar escala"
# ════════════════════════════════════════════════════════════════════════════

try:
    gb_min, gb_ideal, gb_feats = carregar_modelos()
except FileNotFoundError as err:
    st.error(
        f"❌ Ficheiros de modelo não encontrados em `{OUTPUTS_DIR}`.\n\n"
        f"Certifica-te de que o Bloco 3 foi executado e os artefactos exportados.\n\n"
        f"Ficheiro em falta: `{err.filename}`"
    )
    st.stop()

datas = pd.date_range(str(semana_inicio), periods=7)
D     = [d.strftime("%Y-%m-%d") for d in datas]

# ── Passo 1: Construir inputs e prever ───────────────────────────────────────
with st.spinner("A construir inputs e prever cobertura..."):
    df_inputs = construir_semana_operacional(
        semana_inicio, taxa_ocu, n_checkins, n_checkouts, evento
    )
    X_alvo  = gerar_features_previsao(df_inputs, gb_feats)
    y_min   = np.maximum(1, np.round(gb_min.predict(X_alvo))).astype(int)
    y_ideal = np.maximum(1, np.ceil(gb_ideal.predict(X_alvo))).astype(int)

    df_inputs = df_inputs.copy()
    df_inputs["N_min"]   = y_min
    df_inputs["N_ideal"] = y_ideal

    N_min   = {
        (r["data"].strftime("%Y-%m-%d"), r["turno_id"], r["funcao"]): int(r["N_min"])
        for _, r in df_inputs.iterrows()
    }
    N_ideal_d = {
        (r["data"].strftime("%Y-%m-%d"), r["turno_id"], r["funcao"]): int(r["N_ideal"])
        for _, r in df_inputs.iterrows()
    }

# ── Passo 2: Preparar parâmetros e optimizar ─────────────────────────────────
with st.spinner("A optimizar escala..."):
    params = preparar_parametros(DATA_DIR, datas)
    prob, x, y_ext, delta, e = construir_modelo(
        params, N_min, N_ideal_d, D, w1, w2, w3
    )
    status, tempo = resolver(prob)
    escala, escala_int, escala_ext = extrair_escala(x, y_ext)
    kpis = calcular_kpis(
        escala, escala_ext, params, N_min, N_ideal_d,
        D, params["T"], params["F"], status,
    )


# ════════════════════════════════════════════════════════════════════════════
# Outputs
# ════════════════════════════════════════════════════════════════════════════

# ── Cabeçalho de resultado ───────────────────────────────────────────────────
status_icon = "✅" if status == "Optimal" else "⚠️"
st.subheader(f"{status_icon} Resultado — {status}  ·  {tempo}s")
st.caption(
    f"Semana de {semana_inicio.strftime('%d/%m/%Y')} a {semana_fim.strftime('%d/%m/%Y')}  ·  "
    f"Taxa base: {taxa_ocu}%  ·  Check-ins: {n_checkins}  ·  Check-outs: {n_checkouts}"
)

if status == "Infeasible":
    st.error(
        "O modelo não encontrou solução viável com os recursos disponíveis "
        "para os níveis de cobertura previstos. "
        "Tenta reduzir a taxa de ocupação ou verificar a disponibilidade dos colaboradores."
    )

# ── KPI cards ────────────────────────────────────────────────────────────────
def fmt_kpi(val, fmt_str=None):
    """Formata KPI para apresentação. None → 'N/D'. 0 é um valor válido."""
    if val is None:
        return "N/D"
    return f"{val:{fmt_str}}" if fmt_str else str(val)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Cobertura mínima",   fmt_kpi(kpis["cobertura"]))
c2.metric("Custo fixo",
          f"{fmt_kpi(kpis['custo_fixo'], '.0f')} €" if kpis["custo_fixo"] is not None else "N/D",
          help="Custo semanal comprometido da equipa interna (independente das alocações)")
c3.metric("Custo incremental",
          f"{fmt_kpi(kpis['custo_inc'], '.0f')} €" if kpis["custo_inc"] is not None else "N/D",
          help="Custo real adicional dos recursos externos activados")
c4.metric("Externos activados", fmt_kpi(kpis["n_ext"]))
c5.metric("Défice ideal (S4)",  fmt_kpi(kpis["deficit"]),
          help="Slots abaixo do número óptimo de colaboradores")

# ── Escala semanal ────────────────────────────────────────────────────────────
st.subheader("📋 Escala semanal gerada")
if status == "Optimal" and not escala.empty:
    pivot = mostrar_escala(escala)
    st.dataframe(pivot, use_container_width=True)
    st.caption("Emp = Empregada de andares  ·  Sup = Supervisora  ·  Aux = Auxiliar de limpeza  ·  — = Folga")
else:
    st.warning(
        "Não existe solução admissível para esta semana com os recursos internos disponíveis. "
        "A escala não pode ser apresentada. Considera ajustar a taxa de ocupação "
        "ou verificar a disponibilidade da equipa."
    )

# ── Cobertura prevista ────────────────────────────────────────────────────────
with st.expander("📊 Cobertura prevista (N_min e N_ideal por slot)"):
    df_cob = (
        df_inputs[["data","turno_id","funcao","N_min","N_ideal"]]
        .copy()
        .sort_values(["data","turno_id","funcao"])
        .reset_index(drop=True)
    )
    df_cob["data"] = df_cob["data"].dt.strftime("%Y-%m-%d")
    st.dataframe(df_cob, use_container_width=True)
    st.caption(
        f"N_min total: **{sum(N_min.values())}** turnos  ·  "
        f"N_ideal total: **{sum(N_ideal_d.values())}** turnos"
    )

# ── Recursos externos activados ──────────────────────────────────────────────
if kpis["n_ext"] is not None and kpis["n_ext"] > 0:
    with st.expander(f"🔧 Recursos externos activados ({kpis['n_ext']})"):
        st.dataframe(
            escala_ext.sort_values(["data","turno_id","funcao"]).reset_index(drop=True),
            use_container_width=True,
        )
        st.caption(
            f"Custo incremental: **{kpis['custo_inc']:.2f}€**  ·  "
            f"Custo fixo da equipa interna: **{kpis['custo_fixo']:.2f}€** (já comprometido)"
        )

# ── Inputs expandidos (para transparência) ───────────────────────────────────
with st.expander("🔍 Inputs operacionais construídos internamente"):
    df_show = (
        df_inputs[["data","turno_id","funcao","taxa_ocupacao","quartos_ocupados",
                   "n_checkins","n_checkouts","tipo_dia","epoca"]]
        .drop_duplicates(["data","tipo_dia"])
        [["data","taxa_ocupacao","quartos_ocupados","n_checkins","n_checkouts","tipo_dia","epoca"]]
        .sort_values("data")
        .reset_index(drop=True)
    )
    df_show["data"] = df_show["data"].dt.strftime("%Y-%m-%d")
    st.dataframe(df_show, use_container_width=True)
    st.caption(
        "Taxa base com ajustamentos progressivos: Sexta +5%, Sábado +15%, Domingo +10%. "
        "Check-ins e check-outs distribuídos com pesos típicos de hotelaria — totais semanais exactos garantidos."
    )
