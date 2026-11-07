import json
import threading
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from vacv import server


@pytest.fixture
def local_server(monkeypatch, tmp_path):
    fake = SimpleNamespace(summary={"external_data_used": False}, web_root=tmp_path, run_dir=tmp_path,
                           predict=lambda smiles: {"predicted_si": 12.5}, batch=lambda text:{"received":text})
    monkeypatch.setattr(server, "Workbench", lambda path: fake)
    instance = server.make_server(tmp_path, 0)
    worker = threading.Thread(target=instance.serve_forever, daemon=True)
    worker.start()
    yield f"http://127.0.0.1:{instance.server_port}"
    instance.shutdown()
    instance.server_close()
    worker.join(timeout=2)


def test_http_prediction_and_source_isolation(local_server):
    with urlopen(local_server + "/api/summary", timeout=3) as response:
        assert json.load(response)["external_data_used"] is False
        assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
    req = Request(local_server + "/api/predict", data=b'{"smiles":"CCO"}', headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=3) as response:
        assert json.load(response)["predicted_si"] == 12.5
    for path in ("/../data/raw/competition/train.csv", "/download/model.joblib", "/download/../../train.csv"):
        with pytest.raises(HTTPError) as error:
            urlopen(local_server + path, timeout=3)
        assert error.value.code == 404


@pytest.mark.parametrize("headers,data,status", [
    ({"Content-Type": "application/json", "Origin": "https://unrelated.example"}, b'{}', 403),
    ({"Content-Type": "application/json", "Host": "unrelated.example"}, b'{}', 403),
    ({"Content-Type": "text/plain"}, b'{}', 415),
    ({"Content-Type": "application/json"}, b'broken', 400),
    ({"Content-Type": "application/json"}, b'[]', 400),
])
def test_http_rejects_invalid_requests(local_server, headers, data, status):
    with pytest.raises(HTTPError) as error:
        urlopen(Request(local_server + "/api/predict", data=data, headers=headers), timeout=3)
    assert error.value.code == status


def test_http_batch_route_and_origin_guard(local_server):
    payload=json.dumps({'csv':'SMILES\nCCO\n'}).encode()
    request=Request(local_server+'/api/batch',data=payload,headers={'Content-Type':'application/json'})
    with urlopen(request,timeout=3) as response:
        assert json.load(response)['received']=='SMILES\nCCO\n'
    request=Request(local_server+'/api/batch',data=payload,headers={'Content-Type':'application/json','Origin':'https://unrelated.example'})
    with pytest.raises(HTTPError) as error:
        urlopen(request,timeout=3)
    assert error.value.code==403
