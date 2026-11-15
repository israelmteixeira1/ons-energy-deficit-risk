# Model Card — Classificador de Risco de Déficit Energético (subsistema SE)

## Modelo

| Campo | Valor |
|---|---|
| Artefato | `app/modelo_RandomForest.joblib` (2,5 MB) |
| Algoritmo | `RandomForestClassifier` (scikit-learn) |
| Hyperparâmetros | `n_estimators=600`, `max_depth=8`, `min_samples_leaf=5`, `max_features="sqrt"`, `criterion="gini"`, `bootstrap=True`, `random_state=42`, `class_weight={0: 0.990, 1: 1.084, 2: 0.936}` |
| Classes | `Baixo = 0`, `Medio = 1`, `Alto = 2` (`classes_ = [0, 1, 2]`) |
| Features de entrada | 34 (`feature_names_in_`) |
| Formato | joblib/pickle do scikit-learn |

> **Status da verificação:** hyperparâmetros, mapa de classes e lista de features foram
> verificados por **carga estrita** do artefato publicado (`joblib.load` sob
> scikit-learn 1.2.2 / Python 3.11), com paridade de *conjunto* e de *ordem* confirmada
> contra `app/features.json`. Ver [Reprodutibilidade](#reprodutibilidade).

## Uso pretendido

Classificar, para o próximo dia útil (D-1), o **nível de risco de déficit energético**
do subsistema **SUDESTE** em três faixas — *Baixo*, *Medio*, *Alto* — a partir de
indicadores do sistema elétrico e do clima.

O app (`app/app.py`) lê um CSV com as colunas de entrada, alinha as 34 features
esperadas, executa a inferência e devolve a faixa de risco com uma sugestão
operacional (acionamento de térmicas, revisão de intercâmbio, monitoramento
intradiário etc.).

**Fora do escopo:** outros subsistemas, previsão horizonte > D-1, uso em tempo real
de operação, decisão automática de despacho, e qualquer leitura causal das features.

## Dados de treino

| Aspecto | Detalhe |
|---|---|
| Fonte energética | Snowflake `LAB_ONS.CORE_ONS` — `FATO_RISCO_ENERGETICO_DIARIO`, `FATO_BALANCO_ENERGETICO_DIARIO`, `FATO_RESERVATORIO_ENERGIA_DIARIO`, `FATO_FATOR_CAPACIDADE_DIARIO` |
| Fonte climática | INMET (CSV), bucket S3 `csvclima01`, região `us-east-2` |
| Subsistema | **SUDESTE** (filtro `NOM_SUBSISTEMA ILIKE '%SUDESTE%'` na query de extração) |
| Janela temporal | `2024-01-01` → `2025-08-31` (última data com dado disponível; `CFG.END_DATE = 2025-12-12` é o limite superior configurado na query, nunca alcançado pelos dados) |
| Granularidade | diária |
| Volume | **579 linhas × 51 colunas** após a engenharia de features (log do notebook: `Base consolidada: (579, 51)`) |

A data final `2025-08-31` é corroborada por dois caminhos independentes: (a) `app/df_demo.csv`,
exportado da base engineada, vai de `2025-08-29` a `2025-08-31`; (b) aritmética — 609 dias
corridos de `2024-01-01` a `2025-08-31` (inclusive) menos os 30 dias descartados pela janela
móvel de 30 dias (`PRECIP_SOMA_30d`) resultam em exatamente as 579 linhas do log.

### Rótulo (ground truth)

O alvo é `NIVEL_RISCO_CATEGORIA2`, **rederivado por regra** a partir do quantil de
`INTERCAMBIO_MW_MEDIO_DIA` da própria base:

- `INTERCAMBIO_MW_MEDIO_DIA <= q20` → **Alto**
- `q20 < intercâmbio <= q50` → **Medio**
- `intercâmbio > q50` → **Baixo**

Distribuição de classes resultante: `Baixo 49,9%`, `Medio 30,1%`, `Alto 20,0%`.

> A coluna oficial do ONS `NIVEL_RISCO_CATEGORIA` é lida na query, mas **não** é usada
> como alvo. O modelo aprende a regra de quantil acima, não a metodologia oficial do ONS.

## Features (34)

A ordem é posicional e faz parte do contrato do artefato — a lista abaixo está na
ordem exata de `feature_names_in_` e de `app/features.json`.

**Sistema / ONS (17)**

`INTERCAMBIO_MW_MEDIO_DIA_d-1`, `INTERCAMBIO_MW_MEDIO_DIA_7d`, `INTERCAMBIO_MW_MEDIO_DIA_min_7d`,
`CARGA_MWH_DIA_d-1`, `TOTAL_GERACAO_MWH_DIA_d-1`, `EAR_PERCENTUAL_d-1`, `FC_MEDIO_PCT_d-1`,
`INTERCAMBIO_MW_MEDIO_DIA_d-1_mean_14d`, `INTERCAMBIO_MW_MEDIO_DIA_d-1_std_14d`,
`CARGA_MWH_DIA_d-1_mean_14d`, `CARGA_MWH_DIA_d-1_std_14d`,
`EAR_PERCENTUAL_d-1_mean_14d`, `EAR_PERCENTUAL_d-1_std_14d`,
`IMPORT_RATIO_d-1`, `IMPORT_RATIO_d-1_mean_14d`, `IMPORT_RATIO_d-1_std_14d`, `STRESS_HIDRICO`

**Clima / INMET (9)**

`TEMP_MEDIA_C_d-1`, `TEMP_MAX_C_d-1`, `TEMP_MAX_C_7d`, `TEMP_AMPLITUDE_C_d-1`,
`TEMP_AMPLITUDE_C_7d`, `PRECIP_SOMA_7d`, `PRECIP_SOMA_30d`, `UMID_MEDIA_7d`, `VENTO_MEDIA_7d`

**Calendário e flags (8)**

`E_FINALDESEMANA`, `E_FERIADO`, `E_VESPERA_FERIADO`, `E_POS_FERIADO`,
`MES_sin`, `MES_cos`, `DOW_sin`, `DOW_cos`

> ⚠️ **O typo `TEMP_MEDIA_GO_C_7d` no notebook é load-bearing.** O notebook calcula a
> coluna `TEMP_MEDIA_C_7d` (e ela existe no CSV de demonstração `app/df_demo.csv`), mas a
> lista `CLIMA_KEEP` referencia `TEMP_MEDIA_GO_C_7d`; o filtro
> `feature_list = [c for c in feature_list if c in df.columns]` descarta esse nome e o
> modelo treinado/publicado fica com **34** features (sem `TEMP_MEDIA_C_7d`).
> **Corrigir o typo sem retreinar** faz o notebook produzir **35** features e quebra a
> paridade — de conjunto e de ordem — com o artefato publicado e com `app/features.json`.

### Importâncias (impureza, verificadas no artefato)

| # | Feature | Importância |
|---|---|---|
| 1 | `INTERCAMBIO_MW_MEDIO_DIA_d-1` | 0.1482 |
| 2 | `IMPORT_RATIO_d-1` | 0.1333 |
| 3 | `INTERCAMBIO_MW_MEDIO_DIA_7d` | 0.0659 |
| 4 | `STRESS_HIDRICO` | 0.0471 |
| 5 | `INTERCAMBIO_MW_MEDIO_DIA_min_7d` | 0.0441 |

As features de calendário são residuais (`MES_cos` 0.0109, `DOW_sin` 0.0101, `MES_sin` 0.0078,
`DOW_cos` 0.0045, `E_FINALDESEMANA` 0.0059) e as três flags de feriado recebem importância
**exatamente zero** (`E_FERIADO`, `E_VESPERA_FERIADO`, `E_POS_FERIADO`).

## Métricas

> **Estas métricas são alegações históricas do autor, registradas nas saídas do
> notebook (`notebook/notebook.ipynb`, execução de 2025-10-17). Elas não foram
> re-verificadas neste repositório** — reexecutar o notebook exige acesso ao Snowflake
> `LAB_ONS` e ao bucket S3 com as credenciais originais.

| Avaliação | QWK | ACC | Origem |
|---|---|---|---|
| Holdout (último fold do `TimeSeriesSplit`, `test_size=130`, `gap=7`) | **0.747** | **0.831** | log do notebook: `RandomForest: QWK=0.747 | ACC=0.831` |
| Time-series CV (4 folds) | **0.593 ± 0.141** | 0.710 ± 0.068 | log do notebook: `[TS-CV] QWK médio = 0.593 ± 0.141` |

Comparativo do mesmo holdout no notebook: RandomForest QWK 0.747 · XGBoost QWK 0.667.

**Leitura honesta dos números:** o QWK de validação cruzada (0.593) fica bem abaixo do
holdout (0.747) — o holdout é uma única janela de 130 dias e é otimista. O número de
cross-validation é a estimativa mais confiável para operação contínua.

Ressalva adicional: a célula de comparação do notebook avaliou
`RandomForest(n_estimators=900, max_depth=6)`, enquanto o artefato publicado carrega os
parâmetros selecionados pelo grid search temporal (`600/8/5/sqrt`). As métricas acima
**não** foram medidas especificamente sobre o artefato publicado.

## Limitações

- **Um único subsistema.** Treinado e avaliado apenas no SUDESTE; não há evidência de
  generalização para SE/CO, Norte, Nordeste ou Sul.
- **Método de árvores, modelo único.** RandomForest de 600 árvores — sem rede neural e
  sem ensemble de algoritmos distintos. A explicação é local (SHAP/impureza) e o modelo
  não extrapola fora da faixa histórica de registros.
- **Base de treino pequena.** 579 observações diárias para 34 features e 3 classes:
  variância alta e risco de overfitting, apesar do grid search temporal.
- **Alvo autorregulado.** O ground truth é o quantil do intercâmbio, não a classificação
  oficial do ONS — o modelo reproduz o regime da regra quantílica.
- **Sem monitoramento de concept drift.** Não existe pipeline que detecte mudança de
  regime, retreine o modelo ou alerte sobre queda de performance.
- **Probabilidades não calibradas.** `class_weight="balanced"` altera o prior das classes;
  as probabilidades por classe não devem ser lidas como frequência observada.
- **Artefato travado em versão.** Carrega sob scikit-learn 1.2.2 (e também sob 1.1.x, com `InconsistentVersionWarning`); a partir de **1.3** a carga falha (ver abaixo).
- **Contrato de features fixo.** Exatamente 34 nomes, em ordem posicional; qualquer
  mudança na lista quebra a inferência silenciosamente.

## Versão

| Item | Valor |
|---|---|
| scikit-learn do artefato | **1.2.2** (carga estrita verificada; falha a partir de **1.3** — verificado em 1.3.0, 1.3.1 e 1.3.2) |
| Treinamento | notebook executado em 2025-10-17; o log do conector Snowflake registra **Python 3.12.9** no kernel do autor — como o scikit-learn 1.2.2 não publica wheel `cp312`, o ambiente de treino original não é reconstruível a partir do notebook (build local ou outra máquina) |
| Ambiente de serving suportado | **Python 3.11** + pins de `app/requirements.txt` (única combinação verificada nesta auditoria) |

Motivo do travamento: a partir do scikit-learn 1.3 a estrutura do nó da árvore passa a
esperar o campo `missing_go_to_left`, e `joblib.load()` falha com
`ValueError: node array from the pickle has an incompatible dtype`. Em 1.2.2 a carga é
estrita e a inferência funciona.

## Reprodutibilidade

Dependências fixadas em [`app/requirements.txt`](app/requirements.txt) (Python 3.11):

```
scikit-learn==1.2.2
numpy==1.24.4
pandas==2.0.3
joblib==1.3.2
streamlit==1.35.0
altair==5.2.0
shap==0.44.0
xgboost==2.0.3
```

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r app/requirements.txt
```

**Reetraining** (requer credenciais de Snowflake `LAB_ONS` e do bucket S3 `csvclima01`):
`notebook/notebook.ipynb`. O notebook grava `modelo_RandomForest_grid.joblib` e
`features.json`; o artefato publicado está em `app/`.

> ⚠️ **Colisão de nome de arquivo ao reexecutar o notebook.** A célula de comparação de
> modelos faz `joblib.dump(model, "modelo_RandomForest.joblib")` com os parâmetros
> `n_estimators=900, max_depth=6` — o **mesmo nome** do artefato publicado, que carrega os
> parâmetros selecionados pelo grid search temporal (`600/8/5/sqrt`). Reexecutar o notebook
> sobrescreve o artefato publicado por um modelo diferente. Renomeie a saída antes de
> reexecutar.

**Auditoria do artefato e checagem de paridade** (sem dependências além das pinadas):

```bash
python - <<'PY'
import json, joblib
m = joblib.load("app/modelo_RandomForest.joblib")
feats = json.load(open("app/features.json"))
fn = list(m.feature_names_in_)
print("features.json:", len(feats), "| feature_names_in_:", len(fn))
print("SET match:  ", set(fn) == set(feats))
print("ORDER match:", fn == feats)
print("classes_:", list(m.classes_))
PY
```

Saída esperada: `features.json: 34 | feature_names_in_: 34`, `SET match: True`,
`ORDER match: True`, `classes_: [0, 1, 2]`.
