import torch


def pytest_sessionstart(session):
    # Small CPU checks are faster and more predictable without oversubscription.
    torch.set_num_threads(2)
