import pytest
from unittest.mock import MagicMock

class PaymentService:
    def __init__(self, db, gateway_client):
        self.db = db
        self.gateway = gateway_client

    def process_payment(self, user_id, amount):
        if amount <= 0:
            raise ValueError("Amount must be positive")
        txn_id = self.gateway.charge(user_id, amount)
        self.db["payments"][txn_id] = {"user": user_id, "amount": amount}
        return txn_id

def test_process_payment_success(fake_db):
    mock_gateway = MagicMock()
    mock_gateway.charge.return_value = "txn123"
    service = PaymentService(fake_db, mock_gateway)

    txn_id = service.process_payment("u1", 50.0)
    assert txn_id == "txn123"
    assert fake_db["payments"]["txn123"]["amount"] == 50.0

def test_process_payment_invalid_amount(fake_db):
    service = PaymentService(fake_db, MagicMock())
    with pytest.raises(ValueError):
        service.process_payment("u1", 0)
