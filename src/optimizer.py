"""
optimizer.py — Módulo de otimização de escalas para Housekeeping
=================================================================
Encapsula todo o pipeline de otimização do Bloco 4:
  - preparar_parametros   : carrega CSVs e constrói parâmetros do modelo
  - prever_cobertura      : carrega modelos GB e prevê N_min / N_ideal
  - construir_modelo      : cria o problema ILP (cenário híbrido — modelo principal)
  - resolver              : chama o solver CBC
  - extrair_escala        : extrai alocações das variáveis de decisão
  - calcular_kpis         : calcula KPIs com tratamento de Infeasible
  - mostrar_escala        : devolve pivot semanal legível

Decisões metodológicas incorporadas:
  - Custo interno fixo/comprometido — fora da função objetivo
  - Pesos w1/w2/w3 calibrados por análise de sensibilidade
  - Elegibilidade funcional hierárquica (qualificação + hierarquia operacional)
"""

import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from pulp import (
    LpProblem, LpMinimize, LpVariable, lpSum,
    LpStatus, value, PULP_CBC_CMD
)


# ── Constantes ──────────────────────────────────────────────────────────────

TURNOS_PADRAO = ["T1", "T2"]

MAP_FUNCAO_QUAL = {
    "Empregada de andares": "quartos_saida",
    "Supervisora":          "supervisao",
    "Auxiliar de limpeza":  "areas_publicas",
}

# Hierarquia funcional de Housekeeping
# Chave: função a executar | Valor: perfis contratuais elegíveis
ELEGIBILIDADE_HIERARQUICA = {
    "Supervisora": [
        "Supervisora", "Governanta",
    ],
    "Empregada de andares": [
        "Empregada de andares", "Supervisora", "Governanta",
    ],
    "Auxiliar de limpeza": [
        "Auxiliar de limpeza", "Empregada de andares",
        "Supervisora",          "Governanta",
    ],
}

FEATURES_PREVISAO = [
    "turno_id", "funcao", "taxa_ocupacao", "quartos_ocupados",
    "n_checkins", "n_checkouts", "evento_especial", "tipo_dia",
    "epoca", "mes", "dia_semana", "quartos_a_limpar_estimados",
    "carga_total_min", "quartos_checkout_total", "quartos_stayover_total",
]

CAT_COLS = ["turno_id", "funcao", "tipo_dia", "epoca", "dia_semana"]


# ════════════════════════════════════════════════════════════════════════════
# 1. PREPARAR PARÂMETROS
# ════════════════════════════════════════════════════════════════════════════

def preparar_parametros(DATA_DIR: Path, datas, T: list = None) -> dict:
    """
    Carrega todos os CSVs e constrói os parâmetros necessários para o modelo.

    Parâmetros
    ----------
    DATA_DIR : Path
        Pasta com os ficheiros CSV do dataset.
    datas : DatetimeIndex
        Datas da semana-alvo (pd.date_range).
    T : list, opcional
        Turnos a considerar. Default: ["T1", "T2"].

    Devolve
    -------
    dict com:
        C, T, F, R          : conjuntos
        D                   : lista de datas em formato string
        H_t, H_c, W_c, M_c : parâmetros de contrato
        Cost_c              : custo/hora por colaborador
        Custo_fixo_total    : custo fixo semanal da equipa (comprometido)
        Disp                : disponibilidade por (c, d, t)
        Pref                : score de preferência de turno por (c, t)
        FolgaMatch          : flag folga preferida por (c, d)
        Qual                : qualificação técnica por (c, f)
        Elegivel            : qualificação + hierarquia por (c, f)
        Cf                  : colaboradores elegíveis por função
        Cost_r              : custo/hora por recurso externo
        Disp_r              : disponibilidade por (r, d, t)
        Qual_r              : qualificação por (r, f)
        Rf                  : recursos externos elegíveis por função
        mu                  : média de turnos/colaborador na semana
    """
    if T is None:
        T = TURNOS_PADRAO

    DATA_DIR = Path(DATA_DIR)
    D = [d.strftime("%Y-%m-%d") for d in datas]

    # ── Carregar CSVs ────────────────────────────────────────────────────────
    colab   = pd.read_csv(DATA_DIR / "colaboradores.csv")
    turnos  = pd.read_csv(DATA_DIR / "turnos.csv")
    cob     = pd.read_csv(DATA_DIR / "cobertura_necessidades.csv",
                          parse_dates=["data"])
    disp_df = pd.read_csv(DATA_DIR / "disponibilidade.csv",
                          parse_dates=["data"])
    qual_df = pd.read_csv(DATA_DIR / "qualificacoes.csv")
    pref_df = pd.read_csv(DATA_DIR / "preferencias.csv")
    rec_ext = pd.read_csv(DATA_DIR / "recursos_externos.csv")

    # ── Conjuntos base ───────────────────────────────────────────────────────
    C = sorted(colab[colab["ativo"] == 1]["colaborador_id"].unique())
    F = sorted(cob[cob["turno_id"].isin(T)]["funcao"].unique())
    R = sorted(rec_ext[rec_ext["ativo"] == 1]["recurso_id"].unique())

    # ── Parâmetros de contrato ───────────────────────────────────────────────
    H_t    = dict(zip(turnos["turno_id"], turnos["duracao_horas"]))
    H_c    = dict(zip(colab["colaborador_id"], colab["horas_max_dia"]))
    W_c    = dict(zip(colab["colaborador_id"], colab["dias_max_trabalho_semana"]))
    M_c    = dict(zip(colab["colaborador_id"], colab["max_dias_consecutivos"]))
    Cost_c = dict(zip(colab["colaborador_id"], colab["custo_hora"]))

    # Custo fixo semanal: comprometido independentemente das alocações
    # Custo fixo: horas_max_dia × dias_max_trabalho_semana × custo_hora
    # Usa a capacidade contratual individual — mais rigoroso que usar
    # a média da duração dos turnos
    H_c_contrato = dict(zip(colab["colaborador_id"], colab["horas_max_dia"]))
    Custo_fixo_total = sum(
        Cost_c[c] * H_c_contrato[c] * W_c[c] for c in C
    )

    # ── Disponibilidade ──────────────────────────────────────────────────────
    # O dataset cobre 2025. Para semanas de 2026, os registos por data exacta
    # não existem — o filtro devolveria um DataFrame vazio, tornando o modelo
    # Infeasible por razão calendárica, não operacional.
    #
    # Dois modos automáticos:
    #   Exacto  — se existem registos para as datas pedidas (ex: testes com 2025)
    #   Padrão  — generaliza por dia da semana (dayofweek) com threshold histórico
    #
    # Threshold=0.85: preserva ~99% dos slots, reflecte realidade operacional.
    # Infeasible continua possível quando N_min excede a capacidade disponível.
    THRESH_DISP = 0.85

    disp_w_exact = disp_df[
        (disp_df["data"].isin(datas)) &
        (disp_df["turno_id"].isin(T))
    ]

    if len(disp_w_exact) > 0:
        # Modo exacto — dados históricos disponíveis para estas datas
        Disp = {
            (r["colaborador_id"], r["data"].strftime("%Y-%m-%d"), r["turno_id"]): r["disponivel"]
            for _, r in disp_w_exact.iterrows()
        }
    else:
        # Modo padrão — generalizar por dia da semana
        taxa_dow = (
            disp_df[disp_df["turno_id"].isin(T)]
            .assign(dia_semana=disp_df["data"].dt.dayofweek)
            .groupby(["colaborador_id", "turno_id", "dia_semana"])["disponivel"]
            .mean()
            .reset_index()
        )
        _pattern = {
            (r["colaborador_id"], r["turno_id"], int(r["dia_semana"])): r["disponivel"]
            for _, r in taxa_dow.iterrows()
        }
        Disp = {
            (c, d, t): 1 if _pattern.get((c, t, pd.Timestamp(d).dayofweek), 0.0) >= THRESH_DISP else 0
            for c in C for d in D for t in T
        }
        _n_disp = sum(Disp.values())
        print(f"  Disponibilidade: modo padrão — {_n_disp}/{len(Disp)} slots "
              f"({_n_disp/len(Disp)*100:.0f}%) threshold={THRESH_DISP}")

    # ── Preferências ─────────────────────────────────────────────────────────
    pref_dict = {
        (r["colaborador_id"], r["turno_id"]): r["score_preferencia_turno"]
        for _, r in pref_df.iterrows()
    }
    Pref = {(c, t): pref_dict.get((c, t), 1) for c in C for t in T}

    folga_pref = pref_df.groupby("colaborador_id")["folga_preferida_dow"].first().to_dict()
    FolgaMatch = {
        (c, d): 1 if pd.Timestamp(d).dayofweek == folga_pref.get(c, -1) else 0
        for c in C for d in D
    }

    # ── Qualificação técnica ─────────────────────────────────────────────────
    qual_set = set(
        (r["colaborador_id"], r["qualificacao"])
        for _, r in qual_df[qual_df["pode_executar"] == 1].iterrows()
    )
    Qual = {
        (c, f): 1 if (c, MAP_FUNCAO_QUAL[f]) in qual_set else 0
        for c in C for f in F
    }

    # ── Elegibilidade hierárquica ──────────────────────────────────────────────
    Perfil_c = dict(zip(colab["colaborador_id"], colab["funcao_principal"]))
    Elegivel = {
        (c, f): 1
        if Qual.get((c, f), 0) == 1
        and Perfil_c.get(c, "") in ELEGIBILIDADE_HIERARQUICA.get(f, [])
        else 0
        for c in C for f in F
    }
    Cf = {f: [c for c in C if Elegivel.get((c, f), 0) == 1] for f in F}

    # ── Recursos externos ────────────────────────────────────────────────────
    Cost_r = dict(zip(rec_ext["recurso_id"], rec_ext["custo_hora"]))

    Qual_r = {}
    for _, r in rec_ext.iterrows():
        quals = r["qualificacoes"].split(",")
        for f in F:
            Qual_r[(r["recurso_id"], f)] = (
                1 if MAP_FUNCAO_QUAL[f] in quals else 0
            )

    Rf = {f: [r for r in R if Qual_r.get((r, f), 0) == 1] for f in F}

    Disp_r = {}
    for _, r in rec_ext.iterrows():
        rid = r["recurso_id"]
        for d in D:
            is_fds = pd.Timestamp(d).dayofweek >= 5
            for t in T:
                Disp_r[(rid, d, t)] = (
                    0 if (is_fds and r["fds_disponivel"] == 0) else 1
                )

    # ── μ: média de turnos/colaborador na semana ─────────────────────────────
    mu = len(D) * sum(W_c[c] for c in C) / len(C) / 7

    print(f"Parâmetros preparados:")
    print(f"  {len(C)} colaboradores | {len(D)} dias | Turnos: {T}")
    print(f"  {len(F)} funções | {len(R)} recursos externos")
    print(f"  Custo fixo semanal: {Custo_fixo_total:.2f}€  [comprometido]")

    return dict(
        C=C, T=T, F=F, R=R, D=D,
        H_t=H_t, H_c=H_c, W_c=W_c, M_c=M_c,
        Cost_c=Cost_c, Custo_fixo_total=Custo_fixo_total,
        Disp=Disp, Pref=Pref, FolgaMatch=FolgaMatch,
        Qual=Qual, Elegivel=Elegivel, Cf=Cf,
        Cost_r=Cost_r, Disp_r=Disp_r, Qual_r=Qual_r, Rf=Rf,
        mu=mu,
    )


# ════════════════════════════════════════════════════════════════════════════
# 2. PREVER COBERTURA
# ════════════════════════════════════════════════════════════════════════════

def prever_cobertura(DATA_DIR: Path, datas, params: dict, OUTPUTS_DIR: Path = None) -> tuple:
    """
    Carrega os modelos Gradient Boosting guardados pelo Bloco 3
    e prevê N_min e N_ideal para a semana-alvo.

    Devolve
    -------
    N_min : dict  {(d, t, f): int}   → hard constraint H1
    N_ideal : dict  {(d, t, f): int} → referência soft constraint S4
    df_prev : DataFrame               → tabela previsão vs real
    """
    DATA_DIR = Path(DATA_DIR)
    # OUTPUTS_DIR: pasta dos ficheiros .pkl (default = DATA_DIR para retrocompatibilidade)
    PKL_DIR  = Path(OUTPUTS_DIR) if OUTPUTS_DIR is not None else DATA_DIR
    T, F     = params["T"], params["F"]
    D        = params["D"]

    # Carregar modelos (de PKL_DIR — separado dos CSVs se necessário)
    gb_min   = joblib.load(PKL_DIR / "gb_model.pkl")
    gb_ideal = joblib.load(PKL_DIR / "gb_model_v2.pkl")
    features = joblib.load(PKL_DIR / "gb_features.pkl")

    # Construir dataset operacional da semana
    ativ    = pd.read_csv(DATA_DIR / "atividade_hotel.csv",    parse_dates=["data"])
    atv_tip = pd.read_csv(DATA_DIR / "atividade_tipologia.csv", parse_dates=["data"])
    cob     = pd.read_csv(DATA_DIR / "cobertura_necessidades.csv", parse_dates=["data"])
    cob     = cob[cob["turno_id"].isin(T)].copy()

    atv_tip_agg = atv_tip.groupby("data", as_index=False).agg(
        carga_total_min        = ("carga_limpeza_min",  "sum"),
        quartos_checkout_total = ("quartos_checkout",   "sum"),
        quartos_stayover_total = ("quartos_stayover",   "sum"),
    )
    ativ_m     = ativ.merge(atv_tip_agg, on="data", how="left")
    ativ_semana = ativ_m[ativ_m["data"].isin(datas)]
    cob_semana  = cob[cob["data"].isin(datas)].copy()
    df_alvo     = cob_semana.merge(ativ_semana, on="data", how="left")

    if "dia_semana_x" in df_alvo.columns:
        df_alvo = df_alvo.rename(
            columns={"dia_semana_y": "dia_semana"}
        ).drop(columns=["dia_semana_x"])

    # Encoding e alinhamento de features
    X_alvo = pd.get_dummies(df_alvo[FEATURES_PREVISAO], columns=CAT_COLS, drop_first=False)
    X_alvo = X_alvo.reindex(columns=features, fill_value=0)

    # Previsão N_min (round) e N_ideal (ceil)
    y_min   = np.maximum(1, np.round(gb_min.predict(X_alvo))).astype(int)
    y_ideal = np.maximum(1, np.ceil(gb_ideal.predict(X_alvo))).astype(int)

    df_alvo = df_alvo.copy()
    df_alvo["N_min_previsto"]   = y_min
    df_alvo["N_ideal_previsto"] = y_ideal
    df_alvo["N_min_real"]       = df_alvo["colaboradores_minimos"].values
    df_alvo["diff"]             = df_alvo["N_min_previsto"] - df_alvo["N_min_real"]

    N_min = {
        (r["data"].strftime("%Y-%m-%d"), r["turno_id"], r["funcao"]): int(r["N_min_previsto"])
        for _, r in df_alvo.iterrows()
    }
    N_ideal = {
        (r["data"].strftime("%Y-%m-%d"), r["turno_id"], r["funcao"]): int(r["N_ideal_previsto"])
        for _, r in df_alvo.iterrows()
    }

    iguais = (df_alvo["diff"] == 0).sum()
    print(f"Previsão de cobertura:")
    print(f"  {len(N_min)} slots previstos | {iguais}/{len(N_min)} previsões exactas")
    print(f"  N_min total: {sum(N_min.values())} turnos")

    df_prev = df_alvo[[
        "data", "turno_id", "funcao",
        "N_min_real", "N_min_previsto", "diff"
    ]].sort_values(["data", "turno_id", "funcao"]).reset_index(drop=True)

    return N_min, N_ideal, df_prev


# ════════════════════════════════════════════════════════════════════════════
# 3. CONSTRUIR MODELO (cenário híbrido — modelo principal)
# ════════════════════════════════════════════════════════════════════════════

def construir_modelo(
    params: dict,
    N_min: dict,
    N_ideal: dict,
    D: list,
    w1: int = 5,
    w2: int = 10,
    w3: int = 2,
) -> tuple:
    """
    Constrói o modelo ILP principal — cenário híbrido com prioridade aos internos.

    Variáveis de decisão:
      x_{c,d,t,f} ∈ {0,1}  colaboradores internos  [filtrado por Elegivel — P4]
      y_{r,d,t,f} ∈ {0,1}  recursos externos        [só activados quando necessários]

    Hard constraints:
      H1  internos + externos ≥ N_min
      H2  máx. 1 turno/dia por colaborador interno
      H2e máx. 1 turno/dia por recurso externo
      H5  horas diárias ≤ limite contratual
      H6  dias/semana ≤ limite contratual
      H10 dias consecutivos ≤ limite contratual

    Soft constraints (penalizadas na função objetivo):
      S1  preferências de turno  (peso w1)
      S2  folga no dia preferido (peso w1)
      S3  equidade de carga      (peso w3)
      S4  défice face ao ideal   (peso w2)

    Função objetivo:
      min Z = custo_externo_incremental
            + w1·(S1 + S2) + w2·S4 + w3·S3

    O custo interno é fixo/comprometido — não entra na função objetivo.

    Devolve
    -------
    prob, x, y, delta, e
    """
    C  = params["C"];  T  = params["T"];  F  = params["F"]
    R  = params["R"]
    H_t = params["H_t"];  H_c = params["H_c"]
    W_c = params["W_c"];  M_c = params["M_c"]
    Cost_r   = params["Cost_r"]
    Disp     = params["Disp"];    Disp_r = params["Disp_r"]
    Elegivel = params["Elegivel"]; Qual_r = params["Qual_r"]
    Cf       = params["Cf"];      Rf     = params["Rf"]
    Pref     = params["Pref"];    FolgaMatch = params["FolgaMatch"]
    mu       = params["mu"]

    prob = LpProblem("Housekeeping_Hibrido", LpMinimize)

    # ── Variáveis de decisão ─────────────────────────────────────────────────
    x = {
        (c, d, t, f): LpVariable(f"x_{c}_{d}_{t}_{f}", cat="Binary")
        for c in C for d in D for t in T for f in F
        if Disp.get((c, d, t), 0) == 1 and Elegivel.get((c, f), 0) == 1
    }
    y = {
        (r, d, t, f): LpVariable(f"y_{r}_{d}_{t}_{f}", cat="Binary")
        for r in R for d in D for t in T for f in F
        if Disp_r.get((r, d, t), 0) == 1 and Qual_r.get((r, f), 0) == 1
    }

    # ── Variáveis auxiliares (soft constraints) ───────────────────────────────
    delta = {
        (d, t, f): LpVariable(f"delta_{d}_{t}_{f}", lowBound=0)
        for d in D for t in T for f in F if (d, t, f) in N_ideal
    }
    e = {c: LpVariable(f"e_{c}", lowBound=0) for c in C}

    # ── H1: cobertura mínima (internos + externos) ───────────────────────────
    for d in D:
        for t in T:
            for f in F:
                internos = [x[(c, d, t, f)] for c in Cf[f]  if (c, d, t, f) in x]
                externos = [y[(r, d, t, f)] for r in Rf[f]  if (r, d, t, f) in y]
                todos    = internos + externos
                # H1 criada sempre que existe necessidade mínima,
                # mesmo se todos == [] (slot impossível → contribui para Infeasible)
                if (d, t, f) in N_min:
                    prob += lpSum(todos) >= N_min[(d, t, f)], f"H1_{d}_{t}_{f}"

    # ── H2: máx. 1 turno/dia por interno ─────────────────────────────────────
    for c in C:
        for d in D:
            cv = [x[(c, d, t, f)] for t in T for f in F if (c, d, t, f) in x]
            if cv:
                prob += lpSum(cv) <= 1, f"H2_{c}_{d}"

    # ── H2e: máx. 1 turno/dia por recurso externo ────────────────────────────
    for r in R:
        for d in D:
            cv = [y[(r, d, t, f)] for t in T for f in F if (r, d, t, f) in y]
            if cv:
                prob += lpSum(cv) <= 1, f"H2e_{r}_{d}"

    # ── H5: horas diárias ≤ limite contratual ────────────────────────────────
    for c in C:
        for d in D:
            cv = [H_t[t] * x[(c, d, t, f)] for t in T for f in F if (c, d, t, f) in x]
            if cv:
                prob += lpSum(cv) <= H_c[c], f"H5_{c}_{d}"

    # ── H6: dias/semana ≤ limite contratual ──────────────────────────────────
    for c in C:
        cv = [x[(c, d, t, f)] for d in D for t in T for f in F if (c, d, t, f) in x]
        if cv:
            prob += lpSum(cv) <= W_c[c], f"H6_{c}"

    # ── H10: dias consecutivos ≤ limite contratual ───────────────────────────
    for c in C:
        mc = int(M_c[c])
        for start in range(len(D) - mc):
            window = D[start:start + mc + 1]
            cv = [x[(c, d, t, f)] for d in window for t in T for f in F if (c, d, t, f) in x]
            if cv:
                prob += lpSum(cv) <= mc, f"H10_{c}_{start}"

    # ── S3: equidade de carga ─────────────────────────────────────────────────
    for c in C:
        tc = [x[(c, d, t, f)] for d in D for t in T for f in F if (c, d, t, f) in x]
        if tc:
            prob += e[c] >= lpSum(tc) - mu, f"S3a_{c}"
            prob += e[c] >= mu - lpSum(tc), f"S3b_{c}"

    # ── S4: défice face ao ideal ──────────────────────────────────────────────
    # Implementação robusta: restrição criada mesmo se al vazio (lpSum([]) = 0)
    for d in D:
        for t in T:
            for f in F:
                if (d, t, f) in N_ideal and (d, t, f) in delta:
                    internos = [x[(c, d, t, f)] for c in Cf[f]  if (c, d, t, f) in x]
                    externos = [y[(r, d, t, f)] for r in Rf[f]  if (r, d, t, f) in y]
                    prob += (
                        delta[(d, t, f)] >= N_ideal[(d, t, f)] - lpSum(internos + externos),
                        f"S4_{d}_{t}_{f}",
                    )

    # ── Função objetivo ─────────────────────────────────────────────────────────
    custo_ext    = lpSum(Cost_r[r] * H_t[t] * y[(r, d, t, f)] for (r, d, t, f) in y)
    pen_pref     = lpSum((2 - Pref[(c, t)]) * x[(c, d, t, f)] for (c, d, t, f) in x)
    pen_folga    = lpSum(FolgaMatch[(c, d)] * x[(c, d, t, f)] for (c, d, t, f) in x)
    pen_deficit  = lpSum(delta[(d, t, f)] for (d, t, f) in delta)
    pen_equidade = lpSum(e[c] for c in C)

    prob += (
        custo_ext
        + w1 * (pen_pref + pen_folga)
        + w2 * pen_deficit
        + w3 * pen_equidade
    )

    print(f"Modelo construído:")
    print(f"  Variáveis internas (x): {len(x)}")
    print(f"  Variáveis externas (y): {len(y)}")
    print(f"  Restrições: {len(prob.constraints)}")
    print(f"  Pesos: w1={w1}  w2={w2}  w3={w3}")

    return prob, x, y, delta, e


# ════════════════════════════════════════════════════════════════════════════
# 4. RESOLVER
# ════════════════════════════════════════════════════════════════════════════

def resolver(prob) -> tuple:
    """
    Chama o solver CBC e devolve o resultado.

    Devolve
    -------
    status : str   ("Optimal", "Infeasible", ...)
    tempo  : float (segundos)
    """
    t0 = time.time()
    prob.solve(PULP_CBC_CMD(msg=0))
    tempo  = round(time.time() - t0, 2)
    status = LpStatus[prob.status]
    print(f"Solver: {status}  |  Tempo: {tempo}s")
    return status, tempo


# ════════════════════════════════════════════════════════════════════════════
# 5. EXTRAIR ESCALA
# ════════════════════════════════════════════════════════════════════════════

def extrair_escala(x: dict, y: dict = None) -> tuple:
    """
    Extrai alocações com valor > 0.5 das variáveis de decisão.

    Devolve
    -------
    escala_total    : DataFrame  (internos + externos)
    escala_internos : DataFrame
    escala_externos : DataFrame
    """
    rows_int = [
        {"colaborador_id": c, "data": d, "turno_id": t, "funcao": f, "tipo": "Interno"}
        for (c, d, t, f), var in x.items()
        if value(var) is not None and value(var) > 0.5
    ]
    rows_ext = []
    if y:
        rows_ext = [
            {"colaborador_id": r, "data": d, "turno_id": t, "funcao": f, "tipo": "Externo"}
            for (r, d, t, f), var in y.items()
            if value(var) is not None and value(var) > 0.5
        ]

    escala_int = pd.DataFrame(rows_int)
    escala_ext = pd.DataFrame(rows_ext)
    escala_tot = pd.DataFrame(rows_int + rows_ext)

    n_ext = escala_ext["colaborador_id"].nunique() if not escala_ext.empty else 0
    print(f"Escala extraída: {len(rows_int)} alocações internas | {len(rows_ext)} externas ({n_ext} recursos)")

    return escala_tot, escala_int, escala_ext


# ════════════════════════════════════════════════════════════════════════════
# 6. CALCULAR KPIs
# ════════════════════════════════════════════════════════════════════════════

def calcular_kpis(
    escala_total: pd.DataFrame,
    escala_ext:   pd.DataFrame,
    params:       dict,
    N_min:        dict,
    N_ideal:      dict,
    D:            list,
    T:            list,
    F:            list,
    status:       str,
) -> dict:
    """
    Calcula KPIs operacionais com lógica condicional:
    - status != "Optimal" → todas as métricas None (N/D)

    Devolve
    -------
    dict com:
        custo_fixo    : custo comprometido da equipa interna
        custo_inc     : custo incremental dos externos
        custo_total   : custo_fixo + custo_inc
        cobertura     : string "42/42 (100%)"
        slots_ok      : int
        violacoes     : int
        deficit       : int (défice face ao ideal)
        n_int         : colaboradores internos escalados
        n_ext         : recursos externos activados
        n_aloc        : total de alocações
    """
    _nd = dict(
        custo_fixo=None, custo_inc=None, custo_total=None,
        cobertura=None, slots_ok=None, violacoes=None,
        deficit=None, n_int=None, n_ext=None, n_aloc=None,
    )
    if status != "Optimal":
        return _nd

    Cost_c           = params["Cost_c"]
    Cost_r           = params["Cost_r"]
    H_t              = params["H_t"]
    Custo_fixo_total = params["Custo_fixo_total"]

    custo_fixo = Custo_fixo_total
    custo_inc  = (
        sum(Cost_r[r["colaborador_id"]] * H_t[r["turno_id"]]
            for _, r in escala_ext.iterrows())
        if not escala_ext.empty else 0
    )

    def _n_alocados(d, t, f):
        if escala_total.empty:
            return 0
        return len(escala_total[
            (escala_total["data"] == d) &
            (escala_total["turno_id"] == t) &
            (escala_total["funcao"] == f)
        ])

    slots_ok = sum(
        1 for (d, t, f), n in N_min.items()
        if _n_alocados(d, t, f) >= n
    )
    deficit = sum(
        max(0, N_ideal.get((d, t, f), 0) - _n_alocados(d, t, f))
        for d in D for t in T for f in F
        if (d, t, f) in N_ideal
    )

    n_slots  = len(N_min)
    n_int    = escala_total[escala_total["tipo"] == "Interno"]["colaborador_id"].nunique() if not escala_total.empty else 0
    n_ext    = escala_ext["colaborador_id"].nunique() if not escala_ext.empty else 0

    return dict(
        custo_fixo  = custo_fixo,
        custo_inc   = custo_inc,
        custo_total = custo_fixo + custo_inc,
        cobertura   = f"{slots_ok}/{n_slots} ({slots_ok / n_slots * 100:.0f}%)",
        slots_ok    = slots_ok,
        violacoes   = n_slots - slots_ok,
        deficit     = deficit,
        n_int       = n_int,
        n_ext       = n_ext,
        n_aloc      = len(escala_total),
    )


# ════════════════════════════════════════════════════════════════════════════
# 7. MOSTRAR ESCALA
# ════════════════════════════════════════════════════════════════════════════

def mostrar_escala(escala: pd.DataFrame) -> pd.DataFrame:
    """
    Devolve pivot semanal legível:
    linhas = colaboradores, colunas = (data, turno), valores = função abreviada.
    Células vazias = folga (—).
    """
    if escala is None or escala.empty:
        return pd.DataFrame()

    ABREV = {
        "Empregada de andares": "Emp",
        "Supervisora":          "Sup",
        "Auxiliar de limpeza":  "Aux",
    }
    esc = escala.copy()
    esc["abrev"] = esc["funcao"].map(ABREV).fillna("?")

    pivot = esc.pivot_table(
        index="colaborador_id",
        columns=["data", "turno_id"],
        values="abrev",
        aggfunc=lambda x: x.iloc[0],
    )
    return pivot.fillna("—")


# ════════════════════════════════════════════════════════════════════════════
# AUXILIAR: imprimir tabela comparativa de KPIs
# ════════════════════════════════════════════════════════════════════════════

def imprimir_kpis(kpis: dict, semana_inicio: str, semana_fim: str):
    """Imprime os KPIs de forma formatada."""
    def fmt(v, fs=None):
        if v is None:
            return "N/D"
        return f"{v:{fs}}" if fs else str(v)

    status = "Optimal" if kpis["custo_fixo"] is not None else "Infeasible"
    print("=" * 55)
    print("  KPIs — Escala semanal")
    print("=" * 55)
    print(f"  Semana          : {semana_inicio} a {semana_fim}")
    print(f"  Status          : {status}")
    print("-" * 55)
    print(f"  Custo fixo      : {fmt(kpis['custo_fixo'],  '.2f')} €  [comprometido]")
    print(f"  Custo externos  : {fmt(kpis['custo_inc'],   '.2f')} €  [incremental]")
    print(f"  Custo total     : {fmt(kpis['custo_total'], '.2f')} €")
    print("-" * 55)
    print(f"  Cobertura mín.  : {fmt(kpis['cobertura'])}")
    print(f"  Violações H1    : {fmt(kpis['violacoes'])}")
    print(f"  Défice ideal    : {fmt(kpis['deficit'])}")
    print("-" * 55)
    print(f"  Internos escal. : {fmt(kpis['n_int'])}")
    print(f"  Externos activ. : {fmt(kpis['n_ext'])}")
    print(f"  Total alocações : {fmt(kpis['n_aloc'])}")
    print("=" * 55)
