import pytest
from starlette.testclient import TestClient
from webservice import app


@pytest.fixture
def client():
    return TestClient(app)


def test_liveness(client):
    response = client.get("/liveness")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness(client):
    response = client.get("/readiness")
    assert response.status_code == 200
    assert response.text == "OK"
