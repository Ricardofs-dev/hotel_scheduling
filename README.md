# Housekeeping Scheduler

Sistema integrado de apoio à decisão para escalonamento de equipas de Housekeeping em contexto hoteleiro.

Projeto Final de Pós-Graduação em Data Science e Business Intelligence — ISAG · 2026

---

## Descrição

O sistema combina **previsão de necessidades de cobertura** com **otimização de escalas semanais**, produzindo uma escala operacional que equilibra custos, preferências dos colaboradores e qualidade de serviço.

O pipeline integra três componentes:

1. **Modelo preditivo (Gradient Boosting)** — prevê o número mínimo e ideal de colaboradores por slot (dia × turno × função), a partir de inputs operacionais da semana (taxa de ocupação, check-ins, check-outs)
2. **Modelo de otimização ILP híbrido (CBC)** — gera a escala semanal, priorizando colaboradores internos e activando recursos externos apenas quando necessário para satisfazer a cobertura mínima
3. **Aplicação Streamlit** — interface operacional que permite ao utilizador introduzir os inputs da semana e obter a escala e os KPIs em tempo real

---

## Estrutura do repositório

```
hotel_scheduling/
├── app/
│   └── app.py                  # Aplicação Streamlit
├── src/
│   ├── optimizer.py            # Pipeline de otimização (8 funções públicas)
│   └── input_builder.py        # Transformação de inputs operacionais em features
├── notebooks/
│   ├── 01_formulacao.ipynb     # Formulação matemática do modelo ILP
│   ├── 02_dados.ipynb          # Preparação e documentação dos dados
│   ├── 03_previsao.ipynb       # Treino, validação e exportação do modelo preditivo
│   └── 04_otimizacao.ipynb     # Demonstração do pipeline de otimização
├── notebooks_dev/
│   ├── Bloco_3_dev.ipynb       # Desenvolvimento da componente de previsão
│   └── Bloco_4_dev.ipynb       # Desenvolvimento do modelo de otimização
├── data/
│   └── synthetic/              # Dataset semi-sintético (10 CSVs)
├── outputs/                    # Artefactos .pkl incluídos no repositório
├── README.md                   # Documentação principal
└── requirements.txt            # Dependências do projeto
```

---

## Requisitos

```
python >= 3.10
pandas
numpy
scikit-learn
pulp
joblib
streamlit
```

Instalação:

```bash
pip install pandas numpy scikit-learn pulp joblib streamlit
```

---

## Como executar

### Execução rápida

Os artefactos `.pkl` já estão incluídos em `outputs/`, pelo que a aplicação pode ser testada directamente após clonar o repositório:

```bash
streamlit run app/app.py
```

A aplicação permite seleccionar qualquer semana de 2026, introduzir os inputs operacionais e gerar a escala optimizada com KPIs.

### Reprodução completa do pipeline

Para reproduzir o treino do modelo e gerar novos artefactos:

**1. Treinar o modelo preditivo**

Executar `notebooks/03_previsao.ipynb` do início ao fim. Isto exporta para `outputs/`:
- `gb_model.pkl` — modelo para `N_min` (cobertura mínima)
- `gb_model_v2.pkl` — modelo para `N_ideal` (cobertura ideal)
- `gb_features.pkl` — colunas após encoding
- `gb_config.pkl` — semana-alvo para o Bloco 4

Estes ficheiros já se encontram incluídos no repositório e são usados directamente pela app e pelo notebook 4.

**2. Gerar uma escala via notebook**

Executar `notebooks/04_otimizacao.ipynb` após o passo anterior. A semana otimizada é a que ficou configurada em `gb_config.pkl`.

---

## Exemplos de resultados

A tabela seguinte apresenta dois exemplos representativos de semanas testadas:

| Semana | Época | Status | Cobertura | Custo externo |
|---|---|---|---|---|
| 19–25 Janeiro | Baixa | Optimal | 100% | 0 € |
| 4–10 Agosto | Alta | Optimal | 100% | 277 € |

O modelo preditivo (Gradient Boosting) obteve **MAE = 0.20** e **R² = 0.84** na validação com split 80/20 temporal.

---

## Formulação

O modelo ILP minimiza:

$$\min Z = \sum_{r,d,t,f} Cost_r \cdot H_t \cdot y_{r,d,t,f} + w_1(S_1 + S_2) + w_2 \cdot S_4 + w_3 \cdot S_3$$

onde:
- o **custo interno** é fixo e comprometido — não entra na função objetivo
- $y_{r,d,t,f}$ são os recursos externos, activados apenas quando necessário
- $w_1 = 5$, $w_2 = 10$, $w_3 = 2$ — pesos calibrados por análise de sensibilidade

A formulação completa está documentada em `notebooks/01_formulacao.ipynb`.

---

## Dataset

Dataset semi-sintético com 10 tabelas CSV, construído para simular um hotel urbano de 120 quartos com uma equipa de Housekeeping de 20 colaboradores internos e um painel de recursos externos.

As tabelas incluem: colaboradores, turnos, cobertura histórica, disponibilidade, qualificações, preferências, actividade hoteleira, tipologias de quarto, actividade por tipologia e recursos externos.

---

## Notebooks de desenvolvimento

Os notebooks em `notebooks_dev/` documentam o processo de desenvolvimento:

- **Bloco_3_dev** — comparação de modelos preditivos (Regressão Linear, Random Forest, Gradient Boosting), análise de feature importance, validação com TimeSeriesSplit
- **Bloco_4_dev** — evolução do modelo ILP desde o cenário base (só internos) até ao cenário híbrido, análise de sensibilidade aos pesos da função objetivo, elegibilidade funcional hierárquica

Os notebooks em `notebooks/` correspondem à versão final e reprodutível do projeto; os notebooks em `notebooks_dev/` preservam a evolução técnica e metodológica que conduziu a essa versão.
