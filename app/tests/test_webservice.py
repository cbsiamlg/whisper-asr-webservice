import pytest
from starlette.testclient import TestClient
from webservice import app


@pytest.fixture
def client():
    return TestClient(app)


def test_healthx(client):
    response = client.get("/healthx")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_healthz(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.text == "OK"
