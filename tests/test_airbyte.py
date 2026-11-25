import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


def sync_function(requests, sleep):
    path = Path(__file__).resolve().parents[1] / 'dags/airbyte_multi_sync_dag.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'trigger_airbyte_sync')
    namespace = {'requests': requests, 'time': SimpleNamespace(sleep=sleep), 'Variable': SimpleNamespace(get=lambda _: 'http://airbyte')}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['trigger_airbyte_sync']


def test_no_automatic_retry_can_retrigger_remote_jobs():
    path = Path(__file__).resolve().parents[1] / 'dags/airbyte_multi_sync_dag.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'default_args' for t in n.targets))
    values = {k.value: v.value for k, v in zip(node.value.keys, node.value.values) if isinstance(v, ast.Constant)}
    assert values['retries'] == 0


def test_running_job_has_bounded_polling_and_http_timeouts():
    class Response:
        def __init__(self, job): self.job = job
        def raise_for_status(self): pass
        def json(self): return {'job': self.job}
    class Client:
        calls = 0
        def post(self, url, **kwargs):
            assert kwargs.get('timeout') == 30
            self.calls += 1
            return Response({'id': 1} if url.endswith('/sync') else {'status': 'running'})
    client = Client()
    with pytest.raises(TimeoutError):
        sync_function(client, lambda _: None)('connection')
    assert client.calls <= 361


def test_cancelled_job_fails_without_polling_again():
    class Response:
        def __init__(self, job): self.job = job
        def raise_for_status(self): pass
        def json(self): return {'job': self.job}
    class Client:
        def post(self, url, **kwargs):
            return Response({'id': 1} if url.endswith('/sync') else {'status': 'cancelled'})
    def no_sleep(_):
        raise AssertionError('cancelled job was polled again')
    with pytest.raises(RuntimeError, match='cancelled'):
        sync_function(Client(), no_sleep)('connection')


@pytest.mark.parametrize("failure", ["request", "status"])
def test_polling_http_failure_identifies_remote_job(failure):
    import requests
    class Response:
        def __init__(self, job): self.job = job
        def json(self): return {"job": self.job}
        def raise_for_status(self):
            if self.job is None:
                raise requests.HTTPError("HTTP 503")
    class Client:
        exceptions = requests.exceptions
        def post(self, url, **kwargs):
            if url.endswith("/sync"):
                return Response({"id": 987})
            if failure == "request":
                raise requests.Timeout("poll timed out")
            return Response(None)
    with pytest.raises(RuntimeError, match=r"connection.*987") as exc:
        sync_function(Client(), lambda _: None)("connection")
    assert isinstance(exc.value.__cause__, requests.RequestException)


@pytest.mark.parametrize('payload', ['invalid_json', {}, {'job': None}, {'job': {}}, {'job': {'status': []}}])
def test_malformed_polling_response_identifies_remote_job(payload):
    import requests

    class Response:
        def __init__(self, body): self.body = body
        def raise_for_status(self): pass
        def json(self):
            if self.body == 'invalid_json':
                raise ValueError('malformed JSON')
            return self.body

    class Client:
        exceptions = requests.exceptions
        def post(self, url, **kwargs):
            return Response({'job': {'id': 987}} if url.endswith('/sync') else payload)

    with pytest.raises(RuntimeError, match=r'connection.*987.*estado remoto desconhecido') as exc:
        sync_function(Client(), lambda _: pytest.fail('must not retry'))('connection')
    assert exc.value.__cause__ is not None
