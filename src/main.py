import asyncio

from viam.module.module import Module

import gaussian_splatting.module  # noqa: F401  (registers the models on import)


async def main() -> None:
    await Module.run_from_registry()


if __name__ == "__main__":
    asyncio.run(main())
