import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

from tpch_monetdb.llm_cache import utils
from tpch_monetdb.llm_cache.git_snapshotter import GitSnapshotter
from tpch_monetdb.misc.tpch.compiler import Compiler, build_id as read_build_id

logger = logging.getLogger(__name__)


class CachedCompiler(Compiler):
    def __init__(
        self,
        args: Dict,
        git_snapshotter: Optional[GitSnapshotter] = None,
        compile_cache_dir: Optional[Path] = None,
    ) -> None:
        super().__init__(**args)
        self.args = args
        self.git_snapshotter = git_snapshotter
        self.cache_dir = compile_cache_dir

        # create cache dir if needed
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # make 777
            try:
                self.cache_dir.chmod(0o777)
            except PermissionError:
                pass
        return None

    def build(self) -> Optional[str]:
        # forward to cache function. This is only to override the build function of the parent class, which is called by FasttestProc. The actual caching logic is implemented in build_cached, which is called by this function.
        cached_result, used_cache, compile_key_hash = self.build_cached()
        return cached_result

    def build_cached(
        self,
        skip_cache: bool = False,
        current_git_snapshot: Optional[str] = None,
        only_from_cache: bool = False,
    ) -> Tuple[str | None, bool, str]:
        """
        Build with caching support. Returns if the result was from cache.
        This is going beyond the original def build() by returning a tuple
        of (output, from_cache).
        """

        is_cached, cached_result, cache_path, compile_key_hash = (
            self._check_answer_from_cache(current_git_snapshot)
        )
        if is_cached and not skip_cache:
            return cached_result, True, compile_key_hash

        if only_from_cache:
            raise Exception(
                f"Result not found in cache for key {compile_key_hash} and only_from_cache is set. Cache path: {cache_path}"
            )

        # call normal build
        output = super().build()

        # store output in cache only if compilation succeeded (output is None)
        # Don't cache error results to allow AI to fix the code and retry
        if cache_path is not None and output is None:
            if self._artifacts_available():
                utils.dump_pickle(
                    cache_path,
                    CompileCacheType(
                        outputs=output,
                        artifact_fingerprints=self._artifact_fingerprints(),
                    ),
                )
                logger.debug(f"Saved compile result to cache: {cache_path}")
            else:
                logger.info(
                    "Compilation reported success but build artifacts are missing; not caching."
                )
        elif output is not None:
            logger.info(f"Compilation failed, not caching error result: {output[:100]}...")

        return output, False, compile_key_hash

    def _expected_artifact_paths(self) -> Dict[str, Path]:
        paths = {"app": self.workdir / self.app_name}
        paths.update(
            {
                f"lib{lib}.so": self.build_dir_path / f"lib{lib}.so"
                for lib in sorted(self.libs.keys())
            }
        )
        return paths

    def _artifacts_available(self) -> bool:
        for path in self._expected_artifact_paths().values():
            if not path.exists():
                return False
        return True

    def _artifact_fingerprint(self, path: Path) -> str:
        artifact_build_id = read_build_id(path)
        if artifact_build_id is not None:
            return f"build-id:{artifact_build_id}"
        stat = path.stat()
        return f"stat:{stat.st_mtime_ns}:{stat.st_size}"

    def _artifact_fingerprints(self) -> Dict[str, str]:
        return {
            name: self._artifact_fingerprint(path)
            for name, path in self._expected_artifact_paths().items()
        }

    def _cached_artifacts_match(self, cached: "CompileCacheType") -> bool:
        expected = getattr(cached, "artifact_fingerprints", None)
        if not expected:
            logger.info(
                "Compile cache metadata hit without artifact fingerprints; rebuilding."
            )
            return False
        current = self._artifact_fingerprints()
        if current != expected:
            logger.info(
                "Compile cache metadata hit, but local build artifact fingerprints differ; rebuilding."
            )
            return False
        return True

    def _check_answer_from_cache(
        self, current_git_snapshot: Optional[str] = None
    ) -> Tuple[bool, Optional[str], Optional[Path], str]:
        """Return cached compilation metadata only when local artifacts match."""
        if self.git_snapshotter is None and current_git_snapshot is None:
            logger.warning(
                "Can't determine current code version (GitSnapshotter is None); "
                "skipping compile cache lookup."
            )
            return False, None, None, ""

        if self.git_snapshotter is not None and self.git_snapshotter.is_dirty():
            logger.info(
                "Working tree is dirty; skipping compile cache lookup and cache writes."
            )
            return False, None, None, ""

        # fetch git hash
        if current_git_snapshot is not None:
            assert self.git_snapshotter is None, (
                "Cannot provide current_git_snapshot if git_snapshotter is set"
            )
            git_hash = current_git_snapshot
        else:
            assert self.git_snapshotter is not None, (
                "git_snapshotter must be set to fetch git hash"
            )
            git_hash = self.git_snapshotter.current_hash

        if self.cache_dir is None:
            logger.info(
                "Cache directory not configured; skipping compile cache lookup."
            )
            return False, None, None, ""

        hash_payload = dict(self.args)
        hash_payload.pop("working_dir", None)
        hash_payload.update(
            {
                "snapshotter_hash": git_hash,
                "cxx_flags": self.extra_cxxflags,
            }
        )
        compile_key_hash = utils.sha256(utils.stable_json(hash_payload))
        cache_path = _cache_path_for_hash(self.cache_dir, compile_key_hash)

        if not cache_path.exists():
            logger.info(f"No matching compile cache found at {cache_path=}")
            return False, None, cache_path, compile_key_hash

        cached: Optional[CompileCacheType] = utils.load_pickle(
            cache_path, CompileCacheType
        )
        assert cached is not None
        if not self._artifacts_available():
            logger.info(
                "Compile cache metadata hit, but local build artifacts are missing; rebuilding."
            )
            return False, None, cache_path, compile_key_hash
        if not self._cached_artifacts_match(cached):
            return False, None, cache_path, compile_key_hash
        logger.debug(f"Loaded compile result from cache: {cache_path}")
        return True, cached.outputs, cache_path, compile_key_hash


def _cache_path_for_hash(cache_dir: Path, hash: str) -> Path:
    return cache_dir / f"{hash}.pkl"


class CompileCacheType:
    def __init__(
        self,
        outputs: Optional[str],
        artifact_fingerprints: Optional[Dict[str, str]] = None,
    ) -> None:
        self.outputs = outputs
        self.artifact_fingerprints = artifact_fingerprints
        return None
