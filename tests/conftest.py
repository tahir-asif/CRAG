import pytest

from app import main


@pytest.fixture(autouse=True)
def reset_stub_store():
    main._STUB_STORE.clear()
    yield
    main._STUB_STORE.clear()


def pytest_addoption(parser):
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="Run live tests that hit external APIs (Groq, etc.)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--live"):
        return  # --live passed: run everything
    skip_live = pytest.mark.skip(reason="Needs --live flag to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
