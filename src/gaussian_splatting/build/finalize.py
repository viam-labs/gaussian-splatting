"""finalize stage (`--finalize`): add the sequence to the dataset with an organization API key.

Used when the module's own key lacked organization scope for `add_sequences_to_dataset`. Credentials come from
the environment (`VIAM_API_KEY`, `VIAM_API_KEY_ID`), optionally loaded from a `.env` file in the working dir.
The cloud client is injected so tests run against a fake and never touch the real cloud.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Protocol

from gaussian_splatting.build.stages import StageError

log = logging.getLogger("splat-build")

ENV_API_KEY = "VIAM_API_KEY"
ENV_API_KEY_ID = "VIAM_API_KEY_ID"


class SequenceDatasetClient(Protocol):
    async def add_sequences_to_dataset(
        self, dataset_id: str, sequence_ids: list[str], timeout: float | None = None
    ) -> None: ...


class Session(Protocol):
    """What we need from `ViamClient`: a data client and a way to close the connection."""

    @property
    def data_client(self) -> SequenceDatasetClient: ...

    def close(self) -> None: ...


SessionFactory = Callable[[str, str], Awaitable[Session]]


def load_env_file(path: Path) -> dict[str, str]:
    """Minimal .env reader (KEY=VALUE, '#' comments, optional quotes). The real environment wins."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip("'\"")
        values[key] = value
    return values


def credentials(env: dict[str, str] | None = None, dotenv: Path | None = Path(".env")) -> tuple[str, str]:
    merged: dict[str, str] = {}
    if dotenv is not None:
        merged.update(load_env_file(dotenv))
    merged.update(os.environ if env is None else env)
    key, key_id = merged.get(ENV_API_KEY), merged.get(ENV_API_KEY_ID)
    if not key or not key_id:
        raise StageError(
            f"--finalize needs {ENV_API_KEY} and {ENV_API_KEY_ID} (organization key) in the env or .env"
        )
    return key, key_id


async def open_viam_session(api_key: str, api_key_id: str) -> Session:
    from viam.app.viam_client import ViamClient
    from viam.rpc.dial import DialOptions

    client = await ViamClient.create_from_dial_options(DialOptions.with_api_key(api_key, api_key_id))
    return client  # ViamClient has `data_client` and `close()`; structural match of `Session`


async def add_sequence_to_dataset(client: SequenceDatasetClient, dataset_id: str, sequence_id: str) -> None:
    if not dataset_id or not sequence_id:
        raise StageError("--finalize needs both --dataset-id and --sequence-id")
    log.info("adding sequence %s to dataset %s", sequence_id, dataset_id)
    await client.add_sequences_to_dataset(dataset_id, [sequence_id])


async def finalize_async(
    dataset_id: str, sequence_id: str, api_key: str, api_key_id: str, session_factory: SessionFactory
) -> None:
    session = await session_factory(api_key, api_key_id)
    try:
        await add_sequence_to_dataset(session.data_client, dataset_id, sequence_id)
    finally:
        session.close()


def run_finalize(
    dataset_id: str,
    sequence_id: str,
    *,
    session_factory: SessionFactory = open_viam_session,
    env: dict[str, str] | None = None,
    dotenv: Path | None = Path(".env"),
) -> None:
    api_key, api_key_id = credentials(env, dotenv)
    asyncio.run(finalize_async(dataset_id, sequence_id, api_key, api_key_id, session_factory))
