"""
generate_dataset.py
Gera o dataset semi-sintético para o departamento de Housekeeping.
Produz as 7 tabelas definidas no escopo do projeto.
"""

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta, time as dtime
import random
import os

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

OUTPUT_DIR = r"C:\Users\ricar\OneDrive\Ambiente de Trabalho\Pós-Graduações\ISAG\Disciplinas\2º Trimestre\Projeto Final\hotel_scheduling\data\synthetic"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mapeamento de dias da semana para português
DIAS_PT = {0: "Segunda", 1: "Terça", 2: "Quarta", 3: "Quinta",
           4: "Sexta", 5: "Sábado", 6: "Domingo"}

# ─────────────────────────────────────────────
# 1. TABELA DE COLABORADORES
# ─────────────────────────────────────────────
FIRST_NAMES = ["Ana","Mariana","Sofia","Beatriz","Inês","Carla","Joana","Liliana",
               "Patrícia","Cláudia","Rita","Sandra","Mónica","Teresa","Filipa",
               "João","Miguel","Pedro","Rui","Luís","Carlos","António","Jorge","Paulo","Nuno"]
LAST_NAMES  = ["Silva","Santos","Ferreira","Pereira","Costa","Rodrigues","Martins",
               "Sousa","Carvalho","Lopes","Oliveira","Almeida","Fernandes","Gomes","Ribeiro"]

N_COLABORADORES = 20

# Gerar atributos base com lógica coerente entre colunas
_contratos    = random.choices(["Full-time","Part-time"], weights=[0.75, 0.25], k=N_COLABORADORES)
_funcoes      = random.choices(
    ["Empregada de andares","Supervisora","Governanta","Auxiliar de limpeza"],
    weights=[0.55, 0.20, 0.10, 0.15], k=N_COLABORADORES
)
_experiencia  = random.choices(["Junior","Intermédio","Sénior"], weights=[0.30, 0.45, 0.25], k=N_COLABORADORES)

# Custo/hora: varia com função e experiência
_custo_map = {
    ("Empregada de andares","Junior"): 6.5,
    ("Empregada de andares","Intermédio"): 7.5,
    ("Empregada de andares","Sénior"): 8.5,
    ("Supervisora","Junior"): 9.0,
    ("Supervisora","Intermédio"): 10.5,
    ("Supervisora","Sénior"): 12.0,
    ("Governanta","Junior"): 11.0,
    ("Governanta","Intermédio"): 13.0,
    ("Governanta","Sénior"): 15.0,
    ("Auxiliar de limpeza","Junior"): 6.0,
    ("Auxiliar de limpeza","Intermédio"): 6.8,
    ("Auxiliar de limpeza","Sénior"): 7.5,
}

# Nível de função derivado da função principal
_nivel_funcao_map = {
    "Governanta":            "Chefia",
    "Supervisora":           "Supervisão",
    "Empregada de andares":  "Operacional",
    "Auxiliar de limpeza":   "Operacional",
}

colaboradores = pd.DataFrame({
    "colaborador_id":          [f"C{str(i+1).zfill(3)}" for i in range(N_COLABORADORES)],
    "nome":                    [f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}" for _ in range(N_COLABORADORES)],
    "unidade_hotel":           ["Unidade 1"] * N_COLABORADORES,
    "departamento_principal":  ["Housekeeping"] * N_COLABORADORES,
    "funcao_principal":        _funcoes,
    "nivel_funcao":            [_nivel_funcao_map[f] for f in _funcoes],
    "nivel_experiencia":       _experiencia,
    "tipo_contrato":           _contratos,
    "horas_semanais_contrato": [40 if c == "Full-time" else 20 for c in _contratos],
    "horas_max_dia":           [8  if c == "Full-time" else 6  for c in _contratos],
    "dias_max_trabalho_semana":[5  if c == "Full-time" else 3  for c in _contratos],
    "custo_hora":              [round(_custo_map.get((f, e), 7.5) + random.uniform(-0.3, 0.3), 2)
                                for f, e in zip(_funcoes, _experiencia)],
    # Chefia e Supervisão podem fazer noite; Operacionais Sénior também
    "pode_noite":              [1 if (f in ["Supervisora","Governanta"] or e == "Sénior")
                                else int(random.random() > 0.5)
                                for f, e in zip(_funcoes, _experiencia)],
    # Part-time têm menos disponibilidade ao fim de semana
    "pode_fim_semana":         [1 if c == "Full-time" else int(random.random() > 0.3)
                                for c in _contratos],
    # Feriados: Chefia sempre pode; Supervisão na maioria; Operacional menos
    "pode_feriados":           [1 if _nivel_funcao_map[f] == "Chefia"
                                else (int(random.random() > 0.2) if _nivel_funcao_map[f] == "Supervisão"
                                else int(random.random() > 0.4))
                                for f in _funcoes],
    # Rotativo: Full-time têm maior probabilidade de horário rotativo
    "rotativo":                [int(random.random() > 0.3) if c == "Full-time"
                                else int(random.random() > 0.6)
                                for c in _contratos],
    # Full-time: até 5 dias consecutivos; Part-time: até 3
    "max_dias_consecutivos":   [random.choice([4, 5]) if c == "Full-time" else random.choice([2, 3])
                                for c in _contratos],
    "data_admissao":           [date(2018,1,1) + timedelta(days=random.randint(0, 2000)) for _ in range(N_COLABORADORES)],
    "ativo":                   [1] * N_COLABORADORES,
})

# ─────────────────────────────────────────────
# 2. TABELA DE TURNOS
# ─────────────────────────────────────────────
turnos = pd.DataFrame({
    "turno_id":        ["T1", "T2", "T3"],
    "departamento":    ["Housekeeping", "Housekeeping", "Housekeeping"],
    "nome_turno":      ["Manhã", "Tarde", "Noite"],
    "hora_inicio":     ["07:00", "14:00", "22:00"],
    "hora_fim":        ["14:00", "22:00", "07:00"],
    "duracao_horas":   [7, 8, 9],
    "tipo_turno":      ["Principal", "Principal", "Reforço"],
    "turno_noite":     [0, 0, 1],
    "cruza_meia_noite":[0, 0, 1],
    "requer_pausa":    [1, 1, 1],
    "horas_pausa":     [0.5, 1.0, 1.0],
    "ordem_turno":     [1, 2, 3],
    "ativo":           [1, 1, 1],
})

# ─────────────────────────────────────────────
# 3. TABELA DE NECESSIDADE DE COBERTURA
# ─────────────────────────────────────────────
START = date(2025, 1, 1)
END   = date(2025, 12, 31)
dias  = [START + timedelta(d) for d in range((END - START).days + 1)]

def necessidade(dia, turno):
    """Regras de negócio reais de housekeeping."""
    dow = dia.weekday()  # 0=Seg … 6=Dom
    mes = dia.month
    # Sazonalidade: inverno (jan/fev) mais baixo, verão mais alto
    base = {"T1": 6, "T2": 4, "T3": 2}[turno]
    if mes in [7, 8]:
        base = int(base * 1.4)
    # Fins de semana: mais check-outs → mais necessidade de manhã
    if dow >= 5 and turno == "T1":
        base += 2
    return max(1, base + random.randint(-1, 1))

# Necessidade por função dentro do housekeeping
# Cada turno requer um mix de funções
_funcoes_cobertura = {
    "T1": [("Empregada de andares", 0.60), ("Supervisora", 0.25), ("Auxiliar de limpeza", 0.15)],
    "T2": [("Empregada de andares", 0.55), ("Supervisora", 0.25), ("Auxiliar de limpeza", 0.20)],
    "T3": [("Empregada de andares", 0.50), ("Supervisora", 0.30), ("Auxiliar de limpeza", 0.20)],
}

# Origem da necessidade: maioritariamente histórico, alguma previsão, pouca manual
_origens = ["manual", "previsão", "histórico"]
_origens_pesos = [0.10, 0.25, 0.65]

cobertura_rows = []
for dia in dias:
    for turno in ["T1", "T2", "T3"]:
        n_min  = necessidade(dia, turno)
        n_ideal = n_min + 1
        origem = random.choices(_origens, weights=_origens_pesos)[0]

        for funcao, proporcao in _funcoes_cobertura[turno]:
            # Distribuir necessidade mínima e ideal proporcionalmente por função
            min_f   = max(1, round(n_min * proporcao))
            ideal_f = max(1, round(n_ideal * proporcao))
            cobertura_rows.append({
                "data":                   dia.strftime("%Y-%m-%d"),
                "dia_semana":             DIAS_PT[dia.weekday()],
                "departamento":           "Housekeeping",
                "funcao":                 funcao,
                "turno_id":               turno,
                "colaboradores_minimos":  min_f,
                "colaboradores_ideais":   ideal_f,
                "origem_necessidade":     origem,
            })
cobertura = pd.DataFrame(cobertura_rows)

# ─────────────────────────────────────────────
# 4. TABELA DE DISPONIBILIDADE
# ─────────────────────────────────────────────

# Horas disponíveis por turno (quando disponível)
_horas_turno = {"T1": 7, "T2": 8, "T3": 9}

# Motivos de indisponibilidade e respetivos pesos
_motivos_indisp = [
    "folga", "férias", "baixa", "formação",
    "consulta_medica", "restrição_noite", "nao_autorizado_fds",
    "preferencia_pessoal", "outro"
]
_motivos_pesos = [0.25, 0.20, 0.15, 0.10, 0.10, 0.08, 0.05, 0.05, 0.02]

# Hora de início de cada turno (para calcular horas_antecedencia)
_hora_inicio_turno = {"T1": dtime(7, 0), "T2": dtime(14, 0), "T3": dtime(22, 0)}

# Motivos plausíveis para baixas inesperadas (< 12h antecedência)
_motivos_inesperados = ["baixa", "consulta_medica", "outro"]
_motivos_inesperados_pesos = [0.65, 0.25, 0.10]

disp_rows = []
d_counter = 1
for colab in colaboradores["colaborador_id"]:
    contrato  = colaboradores.loc[colaboradores["colaborador_id"] == colab, "tipo_contrato"].values[0]
    pode_noite = colaboradores.loc[colaboradores["colaborador_id"] == colab, "pode_noite"].values[0]
    pode_fds   = colaboradores.loc[colaboradores["colaborador_id"] == colab, "pode_fim_semana"].values[0]

    # Part-time só cobre T1 e T2; sem pode_noite exclui T3
    turnos_possiveis = ["T1", "T2"]
    if contrato == "Full-time" and pode_noite:
        turnos_possiveis.append("T3")

    for dia in dias:
        dow = dia.weekday()  # 0=Seg … 6=Dom
        eh_fds = dow >= 5

        for turno in turnos_possiveis:
            horas_turno = _horas_turno[turno]

            # Indisponibilidade por fim de semana
            if eh_fds and not pode_fds:
                disp_rows.append({
                    "disponibilidade_id":        f"D{str(d_counter).zfill(4)}",
                    "colaborador_id":            colab,
                    "data":                      dia.strftime("%Y-%m-%d"),
                    "turno_id":                  turno,
                    "disponivel":                0,
                    "tipo_disponibilidade":      "Indisponível",
                    "horas_disponiveis":         0,
                    "motivo_indisponibilidade":  "nao_autorizado_fds",
                    "tipo_indisponibilidade":    "planeada",
                    "comunicado_em":             (datetime.combine(dia, dtime(0,0)) - timedelta(days=random.randint(7,30))).strftime("%Y-%m-%d %H:%M"),
                    "horas_antecedencia":        round(random.uniform(168, 720), 1),
                    "ativo":                     1,
                })
                d_counter += 1
                continue

            # Indisponibilidade pontual
            r = random.random()
            if r < 0.05:
                motivo = random.choices(_motivos_indisp, weights=_motivos_pesos)[0]
                # Baixas e consultas médicas têm ~30% de chance de ser inesperadas (<12h)
                inesperada = motivo in ["baixa","consulta_medica"] and random.random() < 0.30
                inicio_turno_dt = datetime.combine(dia, _hora_inicio_turno[turno])
                if inesperada:
                    horas_ant  = round(random.uniform(0.5, 11.9), 1)
                    motivo     = random.choices(_motivos_inesperados, weights=_motivos_inesperados_pesos)[0]
                    tipo_indisp = "inesperada"
                else:
                    horas_ant   = round(random.uniform(12, 720), 1)
                    tipo_indisp = "planeada"
                comunicado = (inicio_turno_dt - timedelta(hours=horas_ant)).strftime("%Y-%m-%d %H:%M")
                disp_rows.append({
                    "disponibilidade_id":        f"D{str(d_counter).zfill(4)}",
                    "colaborador_id":            colab,
                    "data":                      dia.strftime("%Y-%m-%d"),
                    "turno_id":                  turno,
                    "disponivel":                0,
                    "tipo_disponibilidade":      "Indisponível",
                    "horas_disponiveis":         0,
                    "motivo_indisponibilidade":  motivo,
                    "tipo_indisponibilidade":    tipo_indisp,
                    "comunicado_em":             comunicado,
                    "horas_antecedencia":        horas_ant,
                    "ativo":                     1,
                })
            # Disponibilidade parcial (~8%)
            elif r < 0.13:
                horas_parc = random.choice([4, round(horas_turno * 0.5)])
                inicio_turno_dt = datetime.combine(dia, _hora_inicio_turno[turno])
                horas_ant = round(random.uniform(12, 336), 1)
                disp_rows.append({
                    "disponibilidade_id":        f"D{str(d_counter).zfill(4)}",
                    "colaborador_id":            colab,
                    "data":                      dia.strftime("%Y-%m-%d"),
                    "turno_id":                  turno,
                    "disponivel":                1,
                    "tipo_disponibilidade":      "Parcial",
                    "horas_disponiveis":         horas_parc,
                    "motivo_indisponibilidade":  random.choice(["consulta_medica","formação","preferencia_pessoal"]),
                    "tipo_indisponibilidade":    "planeada",
                    "comunicado_em":             (inicio_turno_dt - timedelta(hours=horas_ant)).strftime("%Y-%m-%d %H:%M"),
                    "horas_antecedencia":        horas_ant,
                    "ativo":                     1,
                })
            # Disponibilidade total (~87%)
            else:
                disp_rows.append({
                    "disponibilidade_id":        f"D{str(d_counter).zfill(4)}",
                    "colaborador_id":            colab,
                    "data":                      dia.strftime("%Y-%m-%d"),
                    "turno_id":                  turno,
                    "disponivel":                1,
                    "tipo_disponibilidade":      "Total",
                    "horas_disponiveis":         horas_turno,
                    "motivo_indisponibilidade":  None,
                    "tipo_indisponibilidade":    None,
                    "comunicado_em":             None,
                    "horas_antecedencia":        None,
                    "ativo":                     1,
                })
            d_counter += 1

disponibilidade = pd.DataFrame(disp_rows)

# ─────────────────────────────────────────────
# 5. TABELA DE QUALIFICAÇÕES
# ─────────────────────────────────────────────
# Qualificações específicas de Housekeeping (sem acentos para uso em código/modelo)
qualificacoes_lista = [
    "quartos_saida",       # limpeza de quartos com check-out
    "quartos_ocupados",    # limpeza de quartos com hóspede
    "areas_publicas",      # corredores, lobbies, etc.
    "limpeza_profunda",    # limpeza profunda / desinfeção
    "reposicao_amenities",       # gestão de roupa e produtos
    "supervisao",          # supervisão de equipa
    "primeiros_socorros",  # certificado de primeiros socorros
]

# Nível de qualificação por função (lógica coerente com nivel_funcao)
_nivel_qual_map = {
    "Governanta":            (4, 5),   # nível alto
    "Supervisora":           (3, 4),
    "Empregada de andares":  (2, 4),
    "Auxiliar de limpeza":   (1, 3),
}

# Qualificações obrigatórias por função
_quals_obrigatorias = {
    "Governanta":            ["supervisao","quartos_saida","quartos_ocupados","reposicao_amenities"],
    "Supervisora":           ["supervisao","quartos_saida","quartos_ocupados"],
    "Empregada de andares":  ["quartos_saida","quartos_ocupados"],
    "Auxiliar de limpeza":   ["areas_publicas"],
}

comp_rows = []
q_counter = 1
for colab in colaboradores["colaborador_id"]:
    funcao = colaboradores.loc[colaboradores["colaborador_id"] == colab, "funcao_principal"].values[0]
    nivel_min, nivel_max = _nivel_qual_map.get(funcao, (2, 4))
    obrigatorias = _quals_obrigatorias.get(funcao, ["quartos_saida"])

    # Qualificações obrigatórias + algumas adicionais aleatórias
    opcionais = [q for q in qualificacoes_lista if q not in obrigatorias]
    adicionais = random.sample(opcionais, k=random.randint(0, min(2, len(opcionais))))
    todas = obrigatorias + adicionais

    for qual in todas:
        comp_rows.append({
            "qualificacao_id":   f"Q{str(q_counter).zfill(3)}",
            "colaborador_id":    colab,
            "departamento":      "Housekeeping",
            "qualificacao":      qual,
            "nivel_qualificacao": random.randint(nivel_min, nivel_max),
            "pode_executar":     1,
            "certificado":       int(random.random() > 0.6),
            "ativo":             1,
        })
        q_counter += 1
competencias = pd.DataFrame(comp_rows)

# ─────────────────────────────────────────────
# 6. TABELA DE PREFERÊNCIAS
# ─────────────────────────────────────────────
# score_preferencia_turno: 0=evitar, 1=indiferente, 2=preferido
# A preferência é coerente com pode_noite: quem não pode noite tende a evitar T3
pref_rows = []
for colab in colaboradores["colaborador_id"]:
    pode_noite = colaboradores.loc[colaboradores["colaborador_id"] == colab, "pode_noite"].values[0]
    pode_fds   = colaboradores.loc[colaboradores["colaborador_id"] == colab, "pode_fim_semana"].values[0]
    # Folga preferida: quem não pode fds prefere folgar ao sáb/dom; os restantes são mais flexíveis
    folga_dow = random.choice([5, 6]) if not pode_fds else random.choice([0, 1, 2, 3, 4, 5, 6])
    for turno in ["T1", "T2", "T3"]:
        if turno == "T3" and not pode_noite:
            score = 0  # evitar noite
        else:
            score = random.choices([0, 1, 2], weights=[0.15, 0.40, 0.45])[0]
        pref_rows.append({
            "colaborador_id":          colab,
            "turno_id":                turno,
            "score_preferencia_turno": score,
            "folga_preferida_dow":     folga_dow,
            "ativo":                   1,
        })
preferencias = pd.DataFrame(pref_rows)

# ─────────────────────────────────────────────
# 7. TABELA DE REGISTO DE ATIVIDADE (procura)
# ─────────────────────────────────────────────
N_QUARTOS = 120

# Feriados nacionais portugueses de 2025
FERIADOS_2025 = {
    date(2025, 1, 1),   # Ano Novo
    date(2025, 4, 18),  # Sexta-feira Santa (Páscoa 20/Abr)
    date(2025, 4, 25),  # Dia da Liberdade
    date(2025, 5, 1),   # Dia do Trabalhador
    date(2025, 6, 10),  # Dia de Portugal
    date(2025, 6, 19),  # Corpo de Deus (60 dias após Páscoa)
    date(2025, 8, 15),  # Assunção de Nossa Senhora
    date(2025, 10, 5),  # Implantação da República
    date(2025, 11, 1),  # Todos os Santos
    date(2025, 12, 1),  # Restauração da Independência
    date(2025, 12, 8),  # Imaculada Conceição
    date(2025, 12, 25), # Natal
}

def get_tipo_dia(dia):
    if dia in FERIADOS_2025:
        return "Feriado"
    dow = dia.weekday()
    if dow == 5:
        return "Sabado"
    if dow == 6:
        return "Domingo"
    return "Semana"

def get_epoca(mes):
    if mes in [6, 7, 8, 9]:
        return "Alta"
    if mes in [3, 4, 5, 10, 11]:
        return "Media"
    return "Baixa"

EVENTOS = ["Congresso", "Casamento", "Conferencia", "Gala", "Evento_corporativo", "Sem_evento"]
EVENTOS_PESOS = [0.02, 0.02, 0.02, 0.01, 0.01, 0.92]

atividade_rows = []
for dia in dias:
    dow = dia.weekday()
    mes = dia.month
    tipo_dia = get_tipo_dia(dia)
    epoca    = get_epoca(mes)

    # Taxa de ocupação: sazonalidade + tipo de dia + aleatoriedade
    base_occ = {"Alta": 0.82, "Media": 0.62, "Baixa": 0.45}[epoca]
    if tipo_dia in ["Sabado", "Domingo", "Feriado"]:
        base_occ = min(0.98, base_occ + 0.12)
    elif dow == 4:  # Sexta
        base_occ = min(0.98, base_occ + 0.08)

    taxa_occ         = round(min(0.99, max(0.10, base_occ + np.random.normal(0, 0.04))), 2)
    quartos_ocupados = int(N_QUARTOS * taxa_occ)
    checkins         = int(quartos_ocupados * random.uniform(0.25, 0.45))
    checkouts        = int(quartos_ocupados * random.uniform(0.20, 0.40))

    desc_evento  = random.choices(EVENTOS, weights=EVENTOS_PESOS)[0]
    tem_evento   = int(desc_evento != "Sem_evento")

    # Quartos a limpar: check-outs (limpeza total) + 50% ocupados (limpeza diária)
    quartos_limpar = checkouts + int(quartos_ocupados * 0.5)

    atividade_rows.append({
        "data":                      dia.strftime("%Y-%m-%d"),
        "dia_semana":                DIAS_PT[dia.weekday()],
        "mes":                       mes,
        "tipo_dia":                  tipo_dia,
        "epoca":                     epoca,
        "departamento":              "Housekeeping",
        "taxa_ocupacao":             taxa_occ,
        "quartos_ocupados":          quartos_ocupados,
        "total_quartos":             N_QUARTOS,
        "n_checkins":                checkins,
        "n_checkouts":               checkouts,
        "evento_especial":           tem_evento,
        "descricao_evento":          desc_evento,
        "quartos_a_limpar_estimados": quartos_limpar,
    })
atividade = pd.DataFrame(atividade_rows)


# ─────────────────────────────────────────────
# 8. TABELA DE TIPOLOGIAS DE QUARTO
# ─────────────────────────────────────────────
tipologias_quarto = pd.DataFrame({
    "tipologia_id":              ["TQ01","TQ02","TQ03","TQ04","TQ05"],
    "tipologia":                 ["Single","Double","Twin","Junior Suite","Suite"],
    "m2":                        [18.0, 28.0, 30.0, 45.0, 70.0],
    "tempo_limpeza_saida_min":   [25, 35, 38, 55, 80],
    "tempo_limpeza_stayover_min":[15, 20, 22, 35, 50],
    "fator_complexidade":        [1.0, 1.3, 1.4, 1.8, 2.5],
    "n_quartos_existentes":      [30, 50, 25, 10, 5],
    "departamento":              ["Housekeeping"] * 5,
    "unidade_hotel":             ["Unidade 1"] * 5,
    "ativo":                     [1, 1, 1, 1, 1],
})

# ─────────────────────────────────────────────
# 9. TABELA DE ATIVIDADE POR TIPOLOGIA
# ─────────────────────────────────────────────
# Distribuição proporcional dos quartos por tipologia (soma = n_quartos_existentes = 120)
_proporcao_tipologia = dict(zip(
    tipologias_quarto["tipologia_id"],
    tipologias_quarto["n_quartos_existentes"] / tipologias_quarto["n_quartos_existentes"].sum()
))

# Lookup de tempos de limpeza por tipologia (para carga_limpeza_min)
_tempo_saida    = dict(zip(tipologias_quarto["tipologia_id"], tipologias_quarto["tempo_limpeza_saida_min"]))
_tempo_stayover = dict(zip(tipologias_quarto["tipologia_id"], tipologias_quarto["tempo_limpeza_stayover_min"]))

at_rows = []
at_counter = 1
for dia in dias:
    row_atv = atividade[atividade["data"] == dia.strftime("%Y-%m-%d")].iloc[0]
    q_ocupados_total = row_atv["quartos_ocupados"]
    q_checkout_total = row_atv["n_checkouts"]

    # Distribuição proporcional com ruído diário (Dirichlet)
    ruido = np.random.dirichlet(
        [p * 20 for p in _proporcao_tipologia.values()]
    )
    proporcoes_dia = dict(zip(_proporcao_tipologia.keys(), ruido))

    # Passo 1: distribuir quartos_ocupados garantindo soma exacta
    tq_ids   = list(tipologias_quarto["tipologia_id"])
    q_ocup_f = [q_ocupados_total * proporcoes_dia[t] for t in tq_ids]
    q_ocup_i = [int(v) for v in q_ocup_f]
    resto    = q_ocupados_total - sum(q_ocup_i)
    # Distribuir o resto pelos maiores resíduos fracionais
    residuos = sorted(range(len(tq_ids)), key=lambda i: -(q_ocup_f[i] - q_ocup_i[i]))
    for i in residuos[:resto]:
        q_ocup_i[i] += 1

    # Passo 2: distribuir quartos_checkout garantindo soma exacta e <= ocupados
    q_check_f = [q_checkout_total * proporcoes_dia[t] for t in tq_ids]
    q_check_i = [min(q_ocup_i[i], int(q_check_f[i])) for i in range(len(tq_ids))]
    resto_c   = min(q_checkout_total, sum(q_ocup_i)) - sum(q_check_i)
    residuos_c = sorted(range(len(tq_ids)),
                        key=lambda i: -(q_check_f[i] - q_check_i[i]))
    for i in residuos_c:
        if resto_c <= 0:
            break
        espaco = q_ocup_i[i] - q_check_i[i]
        delta  = min(1, espaco, resto_c)
        q_check_i[i] += delta
        resto_c -= delta

    # Aplicar cap por capacidade máxima de cada tipologia
    cap_map = dict(zip(tipologias_quarto["tipologia_id"], tipologias_quarto["n_quartos_existentes"]))
    for i in range(len(tq_ids)):
        cap = cap_map[tq_ids[i]]
        if q_ocup_i[i] > cap:
            excesso = q_ocup_i[i] - cap
            q_ocup_i[i]  = cap
            q_check_i[i] = min(q_check_i[i], cap)
            # redistribuir excesso proporcionalmente pelas tipologias com folga
            for j in range(len(tq_ids)):
                if j != i:
                    folga = cap_map[tq_ids[j]] - q_ocup_i[j]
                    delta = min(excesso, max(0, folga))
                    q_ocup_i[j] += delta
                    excesso -= delta
                    if excesso <= 0:
                        break

    for idx, tq_id in enumerate(tq_ids):
        q_ocup  = q_ocup_i[idx]
        q_check = min(q_check_i[idx], q_ocup)
        q_stay  = q_ocup - q_check          # garante checkout + stayover = ocupados

        carga = (q_check * _tempo_saida[tq_id] +
                 q_stay  * _tempo_stayover[tq_id])

        at_rows.append({
            "atividade_tipologia_id": f"AT{str(at_counter).zfill(4)}",
            "data":                   dia.strftime("%Y-%m-%d"),
            "tipologia_id":           tq_id,
            "unidade_hotel":          "Unidade 1",
            "departamento":           "Housekeeping",
            "quartos_ocupados":       q_ocup,
            "quartos_checkout":       q_check,
            "quartos_stayover":       q_stay,
            "carga_limpeza_min":      carga,
        })
        at_counter += 1

atividade_tipologia = pd.DataFrame(at_rows)

# ─────────────────────────────────────────────
# 10. TABELA DE RECURSOS EXTERNOS
# ─────────────────────────────────────────────
recursos_externos = pd.DataFrame({
    "recurso_id":                ["RE001","RE002","RE003","RE004","RE005"],
    "fornecedor":                ["Agência HK Pro","Agência HK Pro","Freelancer","Agência CleanTeam","Freelancer"],
    "tipo_recurso":              ["agencia","agencia","freelancer","agencia","freelancer"],
    "nome_contacto":             ["Maria Fonseca","João Esteves","Ana Ribeiro","Carlos Mota","Sofia Nunes"],
    "departamento":              ["Housekeeping"] * 5,
    "unidade_hotel":             ["Unidade 1"] * 5,
    "custo_hora":                [12.50, 12.50, 10.00, 11.00, 9.50],
    "qualificacoes":             [
        "quartos_saida,quartos_ocupados,areas_publicas",
        "quartos_saida,quartos_ocupados",
        "quartos_saida,areas_publicas",
        "quartos_saida,quartos_ocupados,areas_publicas,supervisao",
        "areas_publicas,reposicao_amenities",
    ],
    "disponibilidade_garantida": [1, 1, 0, 1, 0],
    "tempo_resposta_horas":      [2.0, 2.0, 4.0, 1.5, 6.0],
    "turno_noite_disponivel":    [0, 0, 1, 1, 0],
    "fds_disponivel":            [1, 1, 0, 1, 1],
    "ativo":                     [1, 1, 1, 1, 1],
})

# ─────────────────────────────────────────────
# GUARDAR FICHEIROS
# ─────────────────────────────────────────────
tabelas = {
    "colaboradores": colaboradores,
    "turnos": turnos,
    "cobertura_necessidades": cobertura,
    "disponibilidade": disponibilidade,
    "qualificacoes": competencias,
    "preferencias": preferencias,
    "atividade_hotel": atividade,
    "tipologias_quarto": tipologias_quarto,
    "atividade_tipologia": atividade_tipologia,
    "recursos_externos": recursos_externos,
}

for nome, df in tabelas.items():
    path = os.path.join(OUTPUT_DIR, f"{nome}.csv")
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"[OK] {nome}.csv  →  {len(df)} linhas")

print(f"\nDataset guardado em: {OUTPUT_DIR}")