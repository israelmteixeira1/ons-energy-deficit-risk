# ONS Energy Deficit Risk

## Visão Geral

Classificação de risco de déficit energético para o subsistema SE do setor elétrico brasileiro (ONS - Operador Nacional do Sistema Elétrico), combinando modelagem de dados com dbt Core sobre Snowflake e um modelo de Machine Learning servido em Streamlit.

## Arquitetura

- **Data Warehouse**: Snowflake (LAB_ONS)
- **Transformação**: dbt Core 1.10.11
- **Adapter**: dbt-snowflake 1.10.2
- **Ambiente**: Python Virtual Environment (venv)
- **Versionamento**: Git/GitHub

## Pré-requisitos

- Python 3.9 ou superior (o dbt Core 1.10.11 exige Python >= 3.9)
- pip (gerenciador de pacotes Python)
- Acesso ao Snowflake com credenciais válidas
- Git instalado

## Instalação

### 1. Clone o repositório

git clone https://github.com/israelmteixeira1/ons-energy-deficit-risk.git
cd ons-energy-deficit-risk

### 2. Crie e ative o ambiente virtual

python3 -m venv .venv
source .venv/bin/activate

### 3. Instale as dependências do dbt

pip install dbt-core==1.10.11 dbt-snowflake==1.10.2


### Passo 4: Verifique a instalação do dbt

dbt --version


### Passo 5: Configure o profiles.yml

#### 5.1. Crie o diretório .dbt

mkdir -p ~/.dbt

#### 5.2. Crie o arquivo profiles.yml

**Linux/macOS:**

nano ~/.dbt/profiles.yml

#### 5.3. Cole essas informações no arquivo profiles.yml

```
ons_ml:
  target: dev
  outputs:
    dev:
      type: snowflake
      account: -
      user: -
      password: -
      role: ACCOUNTADMIN
      database: LAB_ONS
      warehouse: LAB_WH
      schema: STAGING_ONS
      threads: 4
```


      
#### 5.4. Salve o arquivo

- Pressione `Ctrl + X`
- Pressione `Y` para confirmar
- Pressione `Enter`


#### 5.5. Verifique se o arquivo foi criado corretamente

cat ~/.dbt/profiles.yml

### Passo 6: Teste a conexão com Snowflake

dbt debug

**Resultado esperado** (no final da saída):

Connection test: [OK connection ok]

All checks passed!


## Estrutura do Projeto

**Diretórios:**
- `.venv/` - Ambiente virtual Python (não versionado)
- `dags/` - DAGs do Airflow para orquestração
- `dbt_packages/` - Pacotes dbt instalados (não versionado)
- `macros/` - Macros SQL reutilizáveis
- `models/` - Modelos dbt
  - `staging/` - Camada de staging (dados brutos normalizados)
  - `core/` - Modelos finais (fatos e dimensões)

**Arquivos:**
- `.gitignore` - Arquivos ignorados pelo Git
- `dbt_project.yml` - Configuração principal do projeto
- `package-lock.yml` - Lock de versões de pacotes
- `packages.yml` - Dependências de pacotes dbt
- `README.md` - Este arquivo
- `MODEL_CARD.md` - Ficha do modelo (dados, features, métricas, limitações)
- `LICENSE` - Licença MIT



## Comandos Principais do dbt

### Instalação e Configuração

Instalar/atualizar pacotes dbt
dbt deps

Verificar conexão e configuração
dbt debug

Executar todos os modelos
dbt run

Executar modelo específico
dbt run --select nome_do_modelo

Executar modelos de uma camada
dbt run --select staging.*

Limpar arquivos compilados
dbt clean

Executar todos os testes
dbt test  




# App — Classificador de Risco de Déficit Energético (Streamlit)

## Visão geral
Aplicativo web em Streamlit para classificar o risco de déficit energético em três níveis (*Baixo*, *Médio*, *Alto*) usando um modelo **Random Forest** já treinado. O app:
- Carrega o modelo e a lista de features;
- Lê um CSV demonstrativo;
- Alinha as colunas esperadas e executa a inferência;
- Exibe o nível de risco e recomendações de ação.

## Estrutura do projeto
```
app/
├─ app.py
├─ requirements.txt
├─ modelo_RandomForest.joblib
├─ features.json
└─ df_demo.csv
```

## Pré-requisitos
- **Python** 3.11 — versão exigida pelas dependências fixadas em `app/requirements.txt`

## Instalação
Crie um ambiente virtual e instale as dependências:
```bash
python3.11 -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r app/requirements.txt
```

## Execução
O app resolve `features.json`, `modelo_RandomForest.joblib` e `df_demo.csv` a partir do
diretório atual — execute a partir de `app/`:
```bash
cd app
streamlit run app.py
```
O navegador abrirá em `http://localhost:8501`. O painel lateral mostra o status de **carregamento do modelo**.


## Como usar
1. Verifique no **sidebar** se o modelo carregou com sucesso.
2. Clique em **“▶️ Prever com base em D-1”**.
3. Serão exibidos: **Risco predito** (*Baixo*, *Médio*, *Alto*) e o **playbook** correspondente.

> Mapeamento de rótulos: `Baixo → 0`, `Medio → 1`, `Alto → 2`.
