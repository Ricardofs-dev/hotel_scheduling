"""
input_builder.py — Camada de transformação de inputs operacionais
=================================================================
Converte os inputs do utilizador (taxa de ocupação, check-ins, etc.)
nas features que o modelo de previsão (Gradient Boosting) espera.

Funções públicas:
    get_segunda_feira(data)              : ajusta qualquer data para a Segunda-feira da semana
    get_tipo_dia(data)                   : classifica o dia (Semana / Sabado / Domingo / Feriado)
    get_epoca(mes)                       : classifica a época (Alta / Media / Baixa)
    construir_semana_operacional(...)    : constrói DataFrame com 42 linhas (7d × 2T × 3F)
    gerar_features_previsao(...)         : aplica encoding e alinha com gb_features.pkl
"""

from datetime import date, timedelta
import pandas as pd
import numpy as np

# ── Constantes operacionais ──────────────────────────────────────────────────

N_QUARTOS           = 120    # total de quartos do hotel
CARGA_MEDIA_MIN     = 45     # minutos de limpeza por quarto (estimativa média)
RATIO_QUARTOS_LIMPAR = 0.75  # proporção de quartos que precisam limpeza por dia

FEATURES_MODELO = [
    "turno_id", "funcao", "taxa_ocupacao", "quartos_ocupados",
    "n_checkins", "n_checkouts", "evento_especial", "tipo_dia",
    "epoca", "mes", "dia_semana", "quartos_a_limpar_estimados",
    "carga_total_min", "quartos_checkout_total", "quartos_stayover_total",
]
CAT_COLS = ["turno_id", "funcao", "tipo_dia", "epoca", "dia_semana"]

# ── Feriados 2026 — Portugal (datas fixas + datas móveis calculadas) ──────────
# Páscoa 2026: 5 de Abril
# Corpo de Deus 2026: 4 de Junho (60 dias após Páscoa)

FERIADOS_2026 = {
    date(2026,  1,  1),   # Ano Novo
    date(2026,  2, 17),   # Carnaval (Terça-feira de Carnaval — não feriado nacional mas comum)
    date(2026,  4,  3),   # Sexta-feira Santa
    date(2026,  4,  5),   # Páscoa
    date(2026,  4, 25),   # Dia da Liberdade
    date(2026,  5,  1),   # Dia do Trabalhador
    date(2026,  6,  4),   # Corpo de Deus
    date(2026,  6, 10),   # Dia de Portugal
    date(2026,  8, 15),   # Assunção de Nossa Senhora
    date(2026, 10,  5),   # Implantação da República
    date(2026, 11,  1),   # Dia de Todos os Santos
    date(2026, 12,  1),   # Restauração da Independência
    date(2026, 12,  8),   # Imaculada Conceição
    date(2026, 12, 25),   # Natal
}

# Pesos de distribuição diária (Seg=0 ... Dom=6)
# Refletem padrões operacionais típicos de hotelaria urbana
_PESO_CHECKIN  = [0.05, 0.10, 0.15, 0.20, 0.25, 0.15, 0.10]   # concentra Qui-Sex
_PESO_CHECKOUT = [0.20, 0.25, 0.20, 0.15, 0.10, 0.05, 0.05]   # concentra Seg-Ter
_BOOST_FDS     = [1.00, 1.00, 1.00, 1.00, 1.05, 1.15, 1.10]   # Sáb/Dom têm +ocupação


# ════════════════════════════════════════════════════════════════════════════
# FUNÇÕES AUXILIARES
# ════════════════════════════════════════════════════════════════════════════

def get_segunda_feira(data) -> date:
    """
    Devolve sempre a Segunda-feira da semana à qual 'data' pertence.
    Aceita date, datetime ou string 'YYYY-MM-DD'.

    Exemplos:
        get_segunda_feira(date(2026, 1, 7))  → date(2026, 1, 5)   # era Quarta
        get_segunda_feira(date(2026, 1, 5))  → date(2026, 1, 5)   # já era Segunda
    """
    if isinstance(data, str):
        data = date.fromisoformat(data)
    return data - timedelta(days=data.weekday())


def get_tipo_dia(d: date) -> str:
    """
    Classifica o dia para o campo 'tipo_dia' do modelo.
    Prioridade: Feriado > Domingo > Sabado > Semana
    """
    if d in FERIADOS_2026:
        return "Feriado"
    if d.weekday() == 6:
        return "Domingo"
    if d.weekday() == 5:
        return "Sabado"
    return "Semana"


def get_epoca(mes: int) -> str:
    """
    Classifica a época com base no mês — coerente com o dataset de treino.
    Alta: Jun–Set | Media: Mar–Mai + Out–Nov | Baixa: Jan–Fev + Dez
    """
    if mes in [6, 7, 8, 9]:
        return "Alta"
    if mes in [3, 4, 5, 10, 11]:
        return "Media"
    return "Baixa"


def _distribuir_total(total: int, pesos: list) -> list:
    """
    Distribui 'total' por 7 dias usando 'pesos', garantindo que
    a soma é exactamente igual a 'total' (correcção no dia de maior peso).

    Parâmetros
    ----------
    total  : int — valor total a distribuir
    pesos  : list[float] — 7 pesos (não precisam de somar 1)

    Devolve
    -------
    list[int] — 7 valores inteiros que somam exactamente 'total'
    """
    pesos_norm = [p / sum(pesos) for p in pesos]
    valores    = [int(total * p) for p in pesos_norm]
    residuo    = total - sum(valores)
    # Atribuir o resíduo ao dia com maior peso (minimiza distorção)
    idx_maior  = pesos_norm.index(max(pesos_norm))
    valores[idx_maior] += residuo
    assert sum(valores) == total, f"Erro na distribuição: {sum(valores)} ≠ {total}"
    return valores


# ════════════════════════════════════════════════════════════════════════════
# FUNÇÃO PRINCIPAL: construir_semana_operacional
# ════════════════════════════════════════════════════════════════════════════

def construir_semana_operacional(
    semana_inicio,
    taxa_ocupacao_base: float,
    n_checkins_semana: int,
    n_checkouts_semana: int,
    evento_especial: bool = False,
) -> pd.DataFrame:
    """
    Constrói o DataFrame de inputs operacionais para a semana-alvo.

    A taxa_ocupacao_base é aplicada de Segunda a Quinta.
    São aplicados ajustamentos progressivos no final da semana:
    Sexta +5% (_BOOST_FDS[4]=1.05), Sábado +15% (_BOOST_FDS[5]=1.15),
    Domingo +10% (_BOOST_FDS[6]=1.10).
    A taxa média efectiva da semana será ligeiramente superior ao valor base.

    Os totais de check-ins e check-outs são distribuídos pelos 7 dias com
    pesos típicos de hotelaria e correcção de resíduos para garantir que
    a soma diária é exactamente igual ao total semanal introduzido.

    Parâmetros
    ----------
    semana_inicio         : date | str 'YYYY-MM-DD' — qualquer dia da semana
                            (ajustado automaticamente para Segunda-feira)
    taxa_ocupacao_base    : float [0–100] — taxa base dos dias de semana
    n_checkins_semana     : int — total de check-ins na semana
    n_checkouts_semana    : int — total de check-outs na semana
    evento_especial       : bool — evento especial na semana

    Devolve
    -------
    DataFrame com 42 linhas (7 dias × 2 turnos × 3 funções)
    com todas as features necessárias para o modelo de previsão.
    """
    # Garantir Segunda-feira
    segunda = get_segunda_feira(semana_inicio)
    datas   = pd.date_range(segunda, periods=7)

    turnos  = ["T1", "T2"]
    funcoes = ["Empregada de andares", "Supervisora", "Auxiliar de limpeza"]

    # Distribuir check-ins e check-outs garantindo totais exactos
    checkins_dia  = _distribuir_total(n_checkins_semana,  _PESO_CHECKIN)
    checkouts_dia = _distribuir_total(n_checkouts_semana, _PESO_CHECKOUT)

    rows = []
    for i, data in enumerate(datas):
        # Taxa de ocupação com boost de fim-de-semana
        taxa_dia    = min(100.0, round(taxa_ocupacao_base * _BOOST_FDS[i], 1))
        quartos_ocu = round(taxa_dia / 100 * N_QUARTOS)
        checkins    = checkins_dia[i]
        checkouts   = checkouts_dia[i]

        quartos_lim  = round(quartos_ocu * RATIO_QUARTOS_LIMPAR)
        carga_total  = quartos_lim * CARGA_MEDIA_MIN

        # Stayover: quartos que ficam sem ser checkout — garantir >= 0
        stayover = max(0, quartos_ocu - checkouts)

        tipo_dia = get_tipo_dia(data.date() if hasattr(data, 'date') else data)
        epoca    = get_epoca(data.month)

        for turno in turnos:
            for funcao in funcoes:
                rows.append({
                    "data"                      : data,
                    "turno_id"                  : turno,
                    "funcao"                    : funcao,
                    "taxa_ocupacao"             : taxa_dia,
                    "quartos_ocupados"          : quartos_ocu,
                    "n_checkins"                : checkins,
                    "n_checkouts"               : checkouts,
                    "evento_especial"           : int(evento_especial),
                    "tipo_dia"                  : tipo_dia,
                    "epoca"                     : epoca,
                    "mes"                       : data.month,
                    "dia_semana"                : data.dayofweek,
                    "quartos_a_limpar_estimados": quartos_lim,
                    "carga_total_min"           : carga_total,
                    "quartos_checkout_total"    : checkouts,
                    "quartos_stayover_total"    : stayover,
                })

    df = pd.DataFrame(rows)

    # Verificações de consistência
    assert len(df) == 42, f"DataFrame deve ter 42 linhas, tem {len(df)}"
    assert df["quartos_stayover_total"].min() >= 0, "stayover negativo detectado"

    # Verificar totais de check-ins e check-outs (por função, valores repetidos — dividir por 6)
    total_ci = df[df["turno_id"]=="T1"].groupby("data")["n_checkins"].first().sum()
    total_co = df[df["turno_id"]=="T1"].groupby("data")["n_checkouts"].first().sum()
    assert total_ci == n_checkins_semana,  f"Check-ins: {total_ci} ≠ {n_checkins_semana}"
    assert total_co == n_checkouts_semana, f"Check-outs: {total_co} ≠ {n_checkouts_semana}"

    return df


# ════════════════════════════════════════════════════════════════════════════
# FUNÇÃO DE ENCODING: gerar_features_previsao
# ════════════════════════════════════════════════════════════════════════════

def gerar_features_previsao(
    df_inputs: pd.DataFrame,
    gb_features: list,
) -> pd.DataFrame:
    """
    Aplica one-hot encoding ao DataFrame de inputs e alinha as colunas
    com as features do modelo treinado (gb_features.pkl).

    Colunas ausentes após encoding ficam a zero — garante compatibilidade
    mesmo que o DataFrame de inputs não cubra todas as categorias do treino.

    Parâmetros
    ----------
    df_inputs   : DataFrame produzido por construir_semana_operacional()
    gb_features : list — colunas carregadas de gb_features.pkl

    Devolve
    -------
    DataFrame com exactamente as colunas de gb_features, pela ordem correcta.
    """
    X = pd.get_dummies(df_inputs[FEATURES_MODELO], columns=CAT_COLS, drop_first=False)
    X = X.reindex(columns=gb_features, fill_value=0)
    assert list(X.columns) == list(gb_features), "Colunas não alinhadas com gb_features.pkl"
    return X
