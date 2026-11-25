import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def app_function(name):
    source = (Path(__file__).resolve().parents[1] / 'app' / 'app.py').read_text()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {'pd': pd, 'np': np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<app>', 'exec'), namespace)
    return namespace[name]


def test_missing_feature_is_rejected():
    with pytest.raises(ValueError, match='missing'):
        app_function('align_and_sanitize')(pd.DataFrame({'a': [1]}), ['a', 'b'])


def test_latest_date_selected_instead_of_august():
    app = Path(__file__).resolve().parents[1] / 'app' / 'app.py'
    # The shipped sample has only August; substitute a later row and exercise selection.
    tree = ast.parse(app.read_text())
    start = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'dt_col' for t in n.targets))
    end = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'drop_cols' for t in n.targets))
    namespace = {'pd': pd, 'df_in': pd.DataFrame({'DATA': ['2025-08-31', '2025-09-01'], 'NOM_SUBSISTEMA': ['SUDESTE/CENTRO-OESTE'] * 2})}
    exec(compile(ast.Module(body=tree.body[start:end], type_ignores=[]), '<selection>', 'exec'), namespace)
    assert str(namespace['target_row']['DATA'].iloc[0].date()) == '2025-09-01'


@pytest.mark.parametrize('classes', [[2, 1, 0], [0, 1], [0, 1, 3], None])
def test_inference_rejects_incompatible_model_classes(classes):
    import time

    class Model:
        classes_ = classes

        def predict(self, X):
            return np.array([0])

        def predict_proba(self, X):
            return np.array([[0.1, 0.2, 0.7]])

    source = (Path(__file__).resolve().parents[1] / 'app/app.py').read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'local_inference')
    ns = {'pd': pd, 'np': np, 'time': time, 'load_model': lambda _: Model(),
          'MODEL_PATH': 'unused', 'ORD_MAP': {'Baixo': 0, 'Medio': 1, 'Alto': 2},
          'IDX2LABEL': {0: 'Baixo', 1: 'Medio', 2: 'Alto'}}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<inference>', 'exec'), ns)
    with pytest.raises(ValueError, match='classes_'):
        ns['local_inference']([{'a': 1}])


def test_feature_manifest_matches_published_model():
    import json
    import joblib
    root = Path(__file__).resolve().parents[1] / 'app'
    features = json.loads((root / 'features.json').read_text())
    model = joblib.load(root / 'modelo_RandomForest.joblib')
    assert len(features) == 34
    assert features == list(model.feature_names_in_)
    assert list(model.classes_) == [0, 1, 2]


def test_explicit_csv_path_used(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    import json
    root = Path(__file__).resolve().parents[1]
    features = json.loads((root / 'app/features.json').read_text())
    row = {feature: 1 for feature in features}
    row['DATA'] = '2025-09-01'
    row['NOM_SUBSISTEMA'] = 'SUDESTE/CENTRO-OESTE'
    path = tmp_path / 'input.csv'
    pd.DataFrame([row]).to_csv(path, index=False)
    monkeypatch.setenv('DEMO_CSV', str(path))
    monkeypatch.setenv('FEATURES_JSON', str(root / 'app/features.json'))
    monkeypatch.setenv('MODEL_PATH', str(root / 'app/modelo_RandomForest.joblib'))
    app = AppTest.from_file(str(root / 'app/app.py'), default_timeout=60).run()
    assert not app.error
    assert not app.exception


def test_ui_does_not_recommend_operational_actions(monkeypatch):
    from streamlit.testing.v1 import AppTest
    root = Path(__file__).resolve().parents[1] / 'app'
    monkeypatch.setenv('DEMO_CSV', str(root / 'df_demo.csv'))
    monkeypatch.setenv('FEATURES_JSON', str(root / 'features.json'))
    monkeypatch.setenv('MODEL_PATH', str(root / 'modelo_RandomForest.joblib'))
    app = AppTest.from_file(str(root / 'app.py'), default_timeout=60).run()
    assert any('experimental' in item.value.lower() for item in app.warning)
    app.button[0].click().run(timeout=60)
    assert not app.exception
    assert not any('Sugestão operacional' in item.value for item in app.info)


def test_invalid_historical_context_does_not_break_inference(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    import json
    root = Path(__file__).resolve().parents[1] / 'app'
    row = {feature: 1 for feature in json.loads((root / 'features.json').read_text())}
    row['NOM_SUBSISTEMA'] = 'SUDESTE/CENTRO-OESTE'
    first = dict(row, DATA='2025-08-31')
    first[next(iter(row))] = None
    last = dict(row, DATA='2025-09-01')
    path = tmp_path / 'input.csv'
    pd.DataFrame([first, last]).to_csv(path, index=False)
    monkeypatch.setenv('DEMO_CSV', str(path))
    monkeypatch.setenv('FEATURES_JSON', str(root / 'features.json'))
    monkeypatch.setenv('MODEL_PATH', str(root / 'modelo_RandomForest.joblib'))
    app = AppTest.from_file(str(root / 'app.py'), default_timeout=60).run()
    app.button[0].click().run(timeout=60)
    assert not app.exception
    assert any('SHAP' in item.value for item in app.warning)


def test_manifest_cache_keeps_original_until_restart(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    root = Path(__file__).resolve().parents[1] / 'app'
    manifest = tmp_path / 'features.json'
    manifest.write_text((root / 'features.json').read_text())
    monkeypatch.setenv('DEMO_CSV', str(root / 'df_demo.csv'))
    monkeypatch.setenv('FEATURES_JSON', str(manifest))
    monkeypatch.setenv('MODEL_PATH', str(root / 'modelo_RandomForest.joblib'))
    app = AppTest.from_file(str(root / 'app.py'), default_timeout=60).run()
    assert not app.error and not app.exception
    manifest.write_text('[]')
    app.run()
    assert not app.error and not app.exception


def test_model_cache_keeps_original_after_replacement(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    root = Path(__file__).resolve().parents[1] / 'app'
    model = tmp_path / 'model.joblib'
    model.write_bytes((root / 'modelo_RandomForest.joblib').read_bytes())
    monkeypatch.setenv('DEMO_CSV', str(root / 'df_demo.csv'))
    monkeypatch.setenv('FEATURES_JSON', str(root / 'features.json'))
    monkeypatch.setenv('MODEL_PATH', str(model))
    app = AppTest.from_file(str(root / 'app.py'), default_timeout=60).run()
    assert not app.error and not app.exception
    model.write_bytes(b'not a model')
    app.button[0].click().run(timeout=60)
    assert not app.error and not app.exception
    assert app.session_state['_last_out'].loc[0, 'label'] == 'Medio'


def test_invalid_manifest_is_rejected_in_ui(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    root = Path(__file__).resolve().parents[1] / 'app'
    manifest = tmp_path / 'features.json'
    manifest.write_text('[]')
    monkeypatch.setenv('DEMO_CSV', str(root / 'df_demo.csv'))
    monkeypatch.setenv('FEATURES_JSON', str(manifest))
    monkeypatch.setenv('MODEL_PATH', str(root / 'modelo_RandomForest.joblib'))
    app = AppTest.from_file(str(root / 'app.py'), default_timeout=60).run()
    assert not app.exception
    assert app.error


def test_missing_date_is_rejected_in_ui(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    import json
    root = Path(__file__).resolve().parents[1] / 'app'
    row = {feature: 1 for feature in json.loads((root / 'features.json').read_text())}
    path = tmp_path / 'input.csv'
    pd.DataFrame([row]).to_csv(path, index=False)
    monkeypatch.setenv('DEMO_CSV', str(path))
    monkeypatch.setenv('FEATURES_JSON', str(root / 'features.json'))
    monkeypatch.setenv('MODEL_PATH', str(root / 'modelo_RandomForest.joblib'))
    app = AppTest.from_file(str(root / 'app.py'), default_timeout=60).run()
    assert not app.exception
    assert any('data' in item.value.lower() for item in app.error)


def test_invalid_date_reports_error_in_ui(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    import json
    root = Path(__file__).resolve().parents[1] / 'app'
    row = {feature: 1 for feature in json.loads((root / 'features.json').read_text())}
    row['DATA'] = 'invalid-date'
    path = tmp_path / 'input.csv'
    pd.DataFrame([row]).to_csv(path, index=False)
    monkeypatch.setenv('DEMO_CSV', str(path))
    monkeypatch.setenv('FEATURES_JSON', str(root / 'features.json'))
    monkeypatch.setenv('MODEL_PATH', str(root / 'modelo_RandomForest.joblib'))
    app = AppTest.from_file(str(root / 'app.py'), default_timeout=60).run()
    assert not app.exception
    assert any('data' in item.value.lower() for item in app.error)


def test_missing_feature_reports_error_in_ui(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    root = Path(__file__).resolve().parents[1]
    path = tmp_path / 'input.csv'
    pd.DataFrame([{'DATA': '2025-09-01', 'NOM_SUBSISTEMA': 'SUDESTE/CENTRO-OESTE'}]).to_csv(path, index=False)
    monkeypatch.setenv('DEMO_CSV', str(path))
    monkeypatch.setenv('FEATURES_JSON', str(root / 'app/features.json'))
    monkeypatch.setenv('MODEL_PATH', str(root / 'app/modelo_RandomForest.joblib'))
    app = AppTest.from_file(str(root / 'app/app.py'), default_timeout=60).run()
    assert not app.exception
    assert any('missing' in error.value for error in app.error)


@pytest.mark.parametrize('value', [None, 'bad', np.inf])
def test_invalid_value_is_rejected(value):
    with pytest.raises(ValueError, match='invalid'):
        app_function('align_and_sanitize')(pd.DataFrame({'a': [value]}), ['a'])


@pytest.mark.parametrize("subsystem", [None, "", "SUL", "NORDESTE"])
def test_ui_rejects_missing_or_out_of_scope_subsystem(tmp_path, monkeypatch, subsystem):
    from streamlit.testing.v1 import AppTest
    root = Path(__file__).resolve().parents[1] / "app"
    data = pd.read_csv(root / "df_demo.csv")
    if subsystem is None:
        data = data.drop(columns="NOM_SUBSISTEMA")
    else:
        data.loc[0, "NOM_SUBSISTEMA"] = subsystem
    path = tmp_path / "input.csv"
    data.to_csv(path, index=False)
    monkeypatch.setenv("DEMO_CSV", str(path))
    monkeypatch.setenv("FEATURES_JSON", str(root / "features.json"))
    monkeypatch.setenv("MODEL_PATH", str(root / "modelo_RandomForest.joblib"))
    app = AppTest.from_file(str(root / "app.py"), default_timeout=60).run()
    assert not app.exception
    assert any("subsistema" in item.value.lower() for item in app.error)
    assert not app.button


def test_demo_numeric_predictions_and_latest_ui(monkeypatch):
    import json
    import joblib
    import time
    from streamlit.testing.v1 import AppTest
    root = Path(__file__).resolve().parents[1] / "app"
    features = json.loads((root / "features.json").read_text())
    demo = pd.read_csv(root / "df_demo.csv")
    assert demo.DATA_REFERENCIA.tolist() == ["2025-08-29", "2025-08-30", "2025-08-31"]
    model = joblib.load(root / "modelo_RandomForest.joblib")
    source = ast.parse((root / "app.py").read_text())
    node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "local_inference")
    ns = {"pd": pd, "np": np, "time": time, "load_model": lambda _: model, "MODEL_PATH": "unused",
          "ORD_MAP": {"Baixo": 0, "Medio": 1, "Alto": 2}, "IDX2LABEL": {0: "Baixo", 1: "Medio", 2: "Alto"}}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<inference>", "exec"), ns)
    X = app_function("align_and_sanitize")(demo, features)
    result = ns["local_inference"](X.to_dict(orient="records"))
    expected = [[0.284885952025689, 0.5929667058294782, 0.12214734214483251],
                [0.2578164356735052, 0.6036339328745841, 0.13854963145191027],
                [0.31041450939909215, 0.5911759420316226, 0.0984095485692849]]
    assert result["predictions"] == [1, 1, 1]
    assert result["labels"] == ["Medio"] * 3
    np.testing.assert_allclose(result["proba"], expected, rtol=0, atol=1e-10)
    monkeypatch.setenv("DEMO_CSV", str(root / "df_demo.csv"))
    monkeypatch.setenv("FEATURES_JSON", str(root / "features.json"))
    monkeypatch.setenv("MODEL_PATH", str(root / "modelo_RandomForest.joblib"))
    app = AppTest.from_file(str(root / "app.py"), default_timeout=60).run()
    app.button[0].click().run(timeout=60)
    assert not app.exception
    out = app.session_state["_last_out"]
    assert out.loc[0, "label"] == "Medio"
    np.testing.assert_allclose(out.loc[0, ["p_Baixo", "p_Medio", "p_Alto"]].astype(float), expected[-1], rtol=0, atol=1e-10)
    display = app.dataframe[0].value
    assert str(display.loc[0, "DATA_REFERENCIA"].date()) == "2025-08-31"
    assert display.loc[0, "Faixa estimada"] == "Medio"
