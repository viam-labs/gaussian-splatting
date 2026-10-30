"""finalize stage with a fake data client. Never touches the Viam cloud."""

from __future__ import annotations

from pathlib import Path

import pytest

from gaussian_splatting.build import finalize
from gaussian_splatting.build.stages import StageError


class FakeDataClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    async def add_sequences_to_dataset(
        self, dataset_id: str, sequence_ids: list[str], timeout: float | None = None
    ) -> None:
        self.calls.append((dataset_id, list(sequence_ids)))


class FakeSession:
    def __init__(self, client: FakeDataClient) -> None:
        self._client = client
        self.closed = False

    @property
    def data_client(self) -> FakeDataClient:
        return self._client

    def close(self) -> None:
        self.closed = True


def test_finalize_adds_sequence_and_closes(tmp_path: Path) -> None:
    client = FakeDataClient()
    sessions: list[FakeSession] = []
    seen_creds: list[tuple[str, str]] = []

    async def factory(api_key: str, api_key_id: str) -> FakeSession:
        seen_creds.append((api_key, api_key_id))
        session = FakeSession(client)
        sessions.append(session)
        return session

    dotenv = tmp_path / ".env"
    dotenv.write_text("# org key\nexport VIAM_API_KEY='secret'\nVIAM_API_KEY_ID=\"id-1\"\n")
    finalize.run_finalize("ds-1", "seq-1", session_factory=factory, env={}, dotenv=dotenv)
    assert client.calls == [("ds-1", ["seq-1"])]
    assert seen_creds == [("secret", "id-1")]
    assert sessions[0].closed


def test_env_overrides_dotenv(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("VIAM_API_KEY=file\nVIAM_API_KEY_ID=file-id\n")
    assert finalize.credentials({"VIAM_API_KEY": "env", "VIAM_API_KEY_ID": "env-id"}, dotenv) == (
        "env",
        "env-id",
    )
    assert finalize.credentials({}, dotenv) == ("file", "file-id")


def test_missing_credentials(tmp_path: Path) -> None:
    with pytest.raises(StageError, match="VIAM_API_KEY"):
        finalize.credentials({}, tmp_path / "absent.env")


def test_empty_ids_rejected() -> None:
    import asyncio

    with pytest.raises(StageError):
        asyncio.run(finalize.add_sequence_to_dataset(FakeDataClient(), "", "seq"))
