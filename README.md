# ONS Energy Deficit Risk

Classificação de risco de déficit energético para o **subsistema SE** do setor elétrico
brasileiro (ONS — Operador Nacional do Sistema Elétrico), combinando modelagem de dados
com dbt Core sobre Snowflake e um modelo de Machine Learning servido em Streamlit.

[![CI](https://github.com/israelmteixeira1/ons-energy-deficit-risk/actions/workflows/dbt-ci.yml/badge.svg)](https://github.com/israelmteixeira1/ons-energy-deficit-risk/actions/workflows/dbt-ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Problem

Balanço de energia do Sistema Elétrico Nacional é definido por uma rede de hidrelias,
termelegas, eólicas e solares cuyo déficit é função simultânea de:

- **demanda** (carga diária),
- **disponibilidade de geração** (fator de capacidade, geradores térmicos),
- **intercâmbio** entre subsistemas (o gargalo estrutural do SE),
- **armazenamento** (energia armazenável e risco hídrico),
- **clima** (precipitação, temperatura, umidade, vento).

A decisão operacional (acionar térmicas contratadas, revisar intercâmbio, comunicar
ONS/CCEE) precisa de uma **classificação de risco para D-1**, não de um modelo
físico de ponta. Este projeto entrega essa classificação em três faixas — *Baixo*,
*Medio*, *Alto* — com uma sugestão de playbook operacional associada.

## Data

| Fonte | Conteúdo | Destino |
|---|---|---|
| ONS | Balanço de energia, EAR/ENA diários, disponibilidade de usinas, geração térmica despachada, fator de capacidade, restrição fotovoltaica | Snowflake `LAB_ONS.RAW_ONS` → `LAB_ONS.CORE_ONS` |
| INMET | Precipitação, temperatura, umidade, vento (CSV horário) | S3 `csvclima01` (us-east-2) → `LAB_ONS.CORE_ONS` |

O pipeline dbt tem **21 modelos** (13 em `core/`, 8 em `staging/`), **33 testes de
dados** e **98 sources** declaradas.

```bash
dbt ls --resource-type model   # 21 modelos
dbt ls --resource-type test    # 33 testes
```

## Architecture

```mermaid
flowchart LR
    ONSS[ONS / dados<br/>do setor] --> AB[Airbyte]
    INMET[INMET<br/>CSV horário] --> AB
    AF[Airflow DAG<br/>airbyte_multi_sync] -.aciona.-> AB
    AB --> RAW[(Snowflake<br/>LAB_ONS · RAW_ONS)]
    RAW --> DBT[dbt Core 1.10<br/>staging → core]
    DBT --> CORE[(Snowflake<br/>LAB_ONS · CORE_ONS)]
    CORE --> NB[Notebook<br/>features + treino<br/>+ grid search temporal]
    NB --> ART[app/modelo_RandomForest.joblib<br/>app/features.json]
    ART --> APP[Streamlit<br/>app/app.py · porta 8501]
```

| Camada | Tecnologia | Onde |
|---|---|---|
| Ingestão | Airbyte (chamado por Airflow) | `dags/airbyte_multi_sync_dag.py` |
| Warehouse | Snowflake — `LAB_ONS` | — |
| Transformação | dbt Core 1.10.11 + dbt-snowflake 1.10.2 + dbt_utils 0.9.6 | `models/`, `macros/`, `tests/` |
| ML | scikit-learn 1.2.2 (RandomForest) | `notebook/notebook.ipynb` |
| Serving | Streamlit 1.35.0 | `app/app.py` |

## ML Model

**Ficha completa: [`MODEL_CARD.md`](MODEL_CARD.md)** — dados, features, métricas,
limitações e reprodutibilidade.

- **Algoritmo:** `RandomForestClassifier` — `n_estimators=600`, `max_depth=8`,
  `min_samples_leaf=5`, `max_features="sqrt"`, `random_state=42`
- **Saída:** 3 classes — `Baixo = 0`, `Medio = 1`, `Alto = 2`
- **Features:** 34 (lags de sistema, clima, calendário e flags), **em ordem posicional** —
  a ordem faz parte do contrato do artefato
- **Métricas (alegações históricas do notebook, não re-verificadas aqui):**
  holdout **QWK 0.747 / ACC 0.831**; time-series CV **QWK 0.593 ± 0.141**

### Top-5 importâncias (impureza, extraídas do artefato publicado)

| # | Feature | Importância |
|---|---|---|
| 1 | `INTERCAMBIO_MW_MEDIO_DIA_d-1` | 0.1482 |
| 2 | `IMPORT_RATIO_d-1` | 0.1333 |
| 3 | `INTERCAMBIO_MW_MEDIO_DIA_7d` | 0.0659 |
| 4 | `STRESS_HIDRICO` | 0.0471 |
| 5 | `INTERCAMBIO_MW_MEDIO_DIA_min_7d` | 0.0441 |

O modelo é dominado pelo intercâmbio e pelo balanço entre geração e carga — o que
coerentemente com a definição do risco no setor.

> ⚠️ O artefato é **travado em scikit-learn 1.2.2**: a partir do 1.3 a carga do
> `.joblib` falha por mudança no layout do nó da árvore. Use o pins de
> `app/requirements.txt` (Python 3.11).

## Quickstart

```bash
# build da imagem (a partir da raiz do repo)
docker build -t ons-energy-deficit-risk .

# sobe o app em http://localhost:8501
docker run --rm -p 8501:8501 ons-energy-deficit-risk
```

O contêiner sobe apenas o app Streamlit (o `dbt` roda no CI e contra o Snowflake).

## Local dev

### App (Streamlit)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r app/requirements.txt

cd app            # o app resolve features.json / modelo / CSV a partir do CWD
streamlit run app.py
```

### dbt

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install dbt-core==1.10.11 dbt-snowflake==1.10.2

mkdir -p ~/.dbt && nano ~/.dbt/profiles.yml   # ver bloco "ons_ml" no README original
dbt deps
dbt debug        # testa a conexão
dbt run
dbt test
```

Comandos úteis: `dbt run --select staging.*`, `dbt run --select nome_do_modelo`,
`dbt clean`.

### CI

`.github/workflows/dbt-ci.yml` roda a cada PR: `dbt deps` → `dbt parse` → `dbt ls`
(com verificação de que os 8 modelos de staging resolvem para os nomes snake_case
corretos). `dbt compile` e `dbt test` exigem conexão viva com o Snowflake e **não**
rodam no CI de pull request.

## Limitations

- **Um único subsistema.** Treinado e avaliado só no SUDESTE — não há evidência de
  generalização para os demais.
- **Base pequena.** 579 observações diárias para 34 features e 3 classes: variância alta,
  apesar do grid search temporal.
- **Alvo autorregulado.** O ground truth é o quantil do intercâmbio, não a classificação
  oficial do ONS.
- **Sem monitoramento de concept drift.** Nada detecta mudança de regime ou retreina o
  modelo.
- **Métricas são alegações do autor.** Os números de holdout e TS-CV vêm das saídas do
  notebook de 2025-10-17; não foram reexecutados neste repositório (o retreinamento exige
  credenciais de Snowflake e S3).
- **Artefato versionado.** Ver [ML Model](#ml-model).
- **Testes de dados exigem credenciais.** `dbt test` só roda com acesso ao Snowflake.

## License

[MIT](LICENSE) — © 2025 israelmteixeira1.
