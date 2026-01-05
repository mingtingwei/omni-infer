import sys
from pathlib import Path
import types
import pytest
from omni.adaptors.sglang.utils import ConditionalTNGScope

_THIS_FILE = Path(__file__).resolve()

repo_root = None
for parent in _THIS_FILE.parents:
    if parent.name == "omni":
        repo_root = parent.parent
        break

if repo_root is None:
    raise RuntimeError("Cannot locate repo root (directory containing 'omni')")

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

if "torchair" not in sys.modules:
    fake_torchair = types.ModuleType("torchair")
    fake_torchair.scope = types.SimpleNamespace()
    sys.modules["torchair"] = fake_torchair

# Fake context manager used to mock torchair scopes
class FakeContext:
    def __init__(self, name, record, args):
        self.name = name
        self.record = record
        self.args = args

    def __enter__(self):
        self.record.append(f"enter:{self.name}:{self.args}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.record.append(f"exit:{self.name}:{self.args}")

@pytest.fixture
def scope_record():
    return []


@pytest.fixture
def mock_tng_scope(monkeypatch, scope_record):
    """
    Patch torchair.scope APIs used inside omni.adaptors.sglang.utils
    """

    class FakeScopeAPI:
        def super_kernel(self, scope, options):
            return FakeContext(
                "super_kernel", scope_record, (scope, options)
            )

        def npu_stream_switch(self, stream_id):
            return FakeContext(
                "stream_switch", scope_record, (stream_id,)
            )

        def limit_core_num(self, aic, aiv):
            return FakeContext(
                "limit_core", scope_record, (aic, aiv)
            )

    monkeypatch.setattr(
        "omni.adaptors.sglang.utils.tng.scope",
        FakeScopeAPI(),
        raising=True,
    )

    return scope_record

# Tests
def test_empty_scope(mock_tng_scope):
    with ConditionalTNGScope():
        pass

    assert mock_tng_scope == []


def test_super_kernel_only(mock_tng_scope):
    with ConditionalTNGScope(
        super_kernel=True,
        scope="test_scope",
        options="stream-fusion=1",
    ):
        pass

    assert mock_tng_scope == [
        "enter:super_kernel:('test_scope', 'stream-fusion=1')",
        "exit:super_kernel:('test_scope', 'stream-fusion=1')",
    ]


def test_core_limit_no_stream(mock_tng_scope):
    with ConditionalTNGScope(core_num="4|8"):
        pass

    assert mock_tng_scope == [
        "enter:limit_core:(4, 8)",
        "exit:limit_core:(4, 8)",
    ]


def test_multi_stream_only(mock_tng_scope):
    with ConditionalTNGScope(
        multi_stream=True,
        stream_id="s1",
    ):
        pass

    assert mock_tng_scope == [
        "enter:stream_switch:('s1',)",
        "exit:stream_switch:('s1',)",
    ]


def test_multi_stream_core_limit_910c(mock_tng_scope, monkeypatch):
    monkeypatch.setenv("ASCEND_PLATFORM", "A3")  # 910C

    with ConditionalTNGScope(
        multi_stream=True,
        stream_id="s1",
        core_num="4|8",
    ):
        pass

    assert mock_tng_scope == [
        "enter:stream_switch:('s1',)",
        "enter:limit_core:(20, 40)",
        "exit:limit_core:(20, 40)",
        "exit:stream_switch:('s1',)",
    ]


def test_multi_stream_core_limit_910b(mock_tng_scope, monkeypatch):
    monkeypatch.setenv("ASCEND_PLATFORM", "A2")  # 910B

    with ConditionalTNGScope(
        multi_stream=True,
        stream_id="s2",
        core_num="2|4",
    ):
        pass

    assert mock_tng_scope == [
        "enter:stream_switch:('s2',)",
        "enter:limit_core:(22, 44)",
        "exit:limit_core:(22, 44)",
        "exit:stream_switch:('s2',)",
    ]


def test_full_combination_order(mock_tng_scope, monkeypatch):
    monkeypatch.setenv("ASCEND_PLATFORM", "A2")

    with ConditionalTNGScope(
        super_kernel=True,
        scope="fusion",
        options="stream-fusion=1",
        multi_stream=True,
        stream_id="s3",
        core_num="2|4",
    ):
        pass

    assert mock_tng_scope == [
        "enter:super_kernel:('fusion', 'stream-fusion=1')",
        "enter:stream_switch:('s3',)",
        "enter:limit_core:(22, 44)",
        "exit:limit_core:(22, 44)",
        "exit:stream_switch:('s3',)",
        "exit:super_kernel:('fusion', 'stream-fusion=1')",
    ]


def test_exit_called_on_exception(mock_tng_scope):
    with pytest.raises(RuntimeError):
        with ConditionalTNGScope(
            super_kernel=True,
            scope="err_scope",
        ):
            raise RuntimeError("boom")

    assert any(x.startswith("exit:") for x in mock_tng_scope)
