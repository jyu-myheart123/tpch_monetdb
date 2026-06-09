from collections.abc import Callable

from tpch_monetdb.misc.tpch.fasttest_proc import FasttestProc


class _FasttestHolder:
    def __init__(self) -> None:
        self._runners: dict[str, FasttestProc] = {}

    def get(self, key: str, factory: Callable[[], FasttestProc]) -> FasttestProc:
        runner = self._runners.get(key)
        if runner is None:
            runner = factory()
            self._runners[key] = runner
        return runner

    def keys(self) -> tuple[str, ...]:
        return tuple(self._runners.keys())

    def terminate(self, key: str, *, suppress_errors: bool = False) -> bool:
        runner = self._runners.pop(key, None)
        if runner is None:
            return False
        try:
            runner.terminate(suppress_errors=suppress_errors)
        except Exception:
            if not suppress_errors:
                raise
        return True

    def terminate_matching(
        self,
        predicate: Callable[[str], bool],
        *,
        suppress_errors: bool = True,
    ) -> tuple[str, ...]:
        failed: list[str] = []
        for key in list(self._runners.keys()):
            if not predicate(key):
                continue
            try:
                self.terminate(key, suppress_errors=suppress_errors)
            except Exception:
                failed.append(key)
        return tuple(failed)

    def terminate_all(self) -> None:
        self.terminate_matching(lambda _key: True, suppress_errors=True)


FastTestPool = _FasttestHolder()
