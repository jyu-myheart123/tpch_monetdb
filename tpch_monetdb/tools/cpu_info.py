import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from agents.run_context import RunContextWrapper
from agents.tool import FunctionTool
from pydantic import BaseModel, Field

from tpch_monetdb.llm_cache.git_snapshotter import GitSnapshotter
from tpch_monetdb.tools.function_tool_args import load_function_tool_args
from tpch_monetdb.tools.litellm_shell import CHARS_PER_TOKEN, MAX_TOOL_RESULT_TOKENS
from tpch_monetdb.tools.sandbox import SandboxConfig, sandbox_shell_async
from tpch_monetdb.utils.wandb_stats_logging import WandbRunHook

logger = logging.getLogger(__name__)


class CpuInfoArgs(BaseModel):
    timeout_ms: int | None = Field(
        2000, description="Timeout in milliseconds for each probe command"
    )


class CpuInfoTool:
    def __init__(
        self,
        cwd: Path,
        cache_dir: Path,
        git_snapshotter: Optional[GitSnapshotter] = None,
        wandb_metrics_hook: Optional[WandbRunHook] = None,
        max_output_tokens: int = MAX_TOOL_RESULT_TOKENS,
    ) -> None:
        """Collect read-only CPU topology and ISA evidence for optimization decisions."""
        self.cwd = cwd
        self.cache_dir = cache_dir
        self.git_snapshotter = git_snapshotter
        self.wandb_metrics_hook = wandb_metrics_hook
        self.max_output_tokens = max_output_tokens
        return None

    async def _run_probe(self, command: str, timeout_ms: int | None) -> dict[str, Any]:
        """Run a single read-only probe command inside the sandbox."""
        tmp_root = os.environ.get("TMPDIR") or tempfile.gettempdir()
        cfg = SandboxConfig(
            writable_roots=[str(self.cwd)],
            cwd=str(self.cwd),
            tmp_root=tmp_root,
            fail_if_unavailable=True,
            nproc=None,
        )
        proc = await sandbox_shell_async(
            command,
            cfg=cfg,
            env=os.environ.copy(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        timed_out = False
        try:
            timeout = (timeout_ms or 0) / 1000 or None
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            stdout_bytes, stderr_bytes = await proc.communicate()
            timed_out = True
        stdout = stdout_bytes.decode("utf-8", errors="ignore")
        stderr = stderr_bytes.decode("utf-8", errors="ignore")
        return {
            "command": command,
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": getattr(proc, "returncode", None),
            "timed_out": timed_out,
        }

    def _truncate(self, text: str) -> str:
        """Clamp oversized raw probe output to the shared tool token budget."""
        # 计算截断阈值：最多保留 max_output_tokens 的文本
        # 每个 token 约等于 4 个字符（大致估计）
        max_chars = self.max_output_tokens * 4
        
        if len(text) <= max_chars:
            # 文本在限制内，直接返回
            return text
        
        # 文本过长，需要截断：保留开头和结尾，中间用省略号连接
        head_chars = max_chars // 3  # 开头占 1/3
        tail_chars = max_chars // 3  # 结尾占 1/3
        # 中间预留给省略号提示
        
        head = text[:head_chars]
        tail = text[-tail_chars:]
        
        # 在安全的行边界处进行截断，避免切断中间的单词或行
        # 找到最后一个换行符，在那里切断
        last_newline_in_head = head.rfind('\n')
        if last_newline_in_head > 0:
            head = head[:last_newline_in_head]
        
        first_newline_in_tail = tail.find('\n')
        if first_newline_in_tail >= 0:
            tail = tail[first_newline_in_tail + 1:]
        
        truncation_notice = (
            f"\n... (total {len(text)} chars, truncated; "
            f"showing first {len(head)} and last {len(tail)} chars) ...\n"
        )
        
        return head + truncation_notice + tail

    def _parse_cpuinfo_flags(self, text: str) -> list[str]:
        """Extract ISA flags from /proc/cpuinfo when present."""
        # /proc/cpuinfo 中包含 CPU 功能标志，通常在 "flags" 或 "Features" 行
        # 我们需要从这些行中提取 ISA 功能标志
        
        flags = []
        
        # 逐行扫描 cpuinfo 输出
        for line in text.splitlines():
            line = line.strip()
            # 寻找标志行："flags" 或 "Features"
            # Linux 一般用 "flags:"，ARM64 用 "Features:"
            if line.startswith("flags") or line.startswith("Features"):
                # 例如："flags : fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm cpuid_fault epb pti ssbd ibrs ibpb stibp tpr_shadow vnmi flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid cqm_llc cqm_occup_llc rdt_a rdseed adx smap clflushopt clwb intel_pt sha_ni xsaveopt xsavec xgetbv1 xsaves cqm_mbm_total cqm_mbm_local dtherm ida arat pln pts vnmi pti"
                # 冒号后面是所有 flag，用空格分隔
                if ':' in line:
                    flags_str = line.split(':', 1)[1].strip()
                    # 以空格分隔，拆成单个 flag
                    flags = flags_str.split()
                break  # 一般只有一个 flags 行，找到就可以停止
        
        return flags  # 保持原始顺序以便测试断言匹配

    def _parse_lscpu_summary(self, text: str) -> dict[str, str]:
        """Extract stable key/value fields from lscpu output."""
        # lscpu 输出是键值对格式：
        # 例如：
        #   Architecture:                    x86_64
        #   CPU op-mode(s):                  32-bit, 64-bit
        #   Model name:                      Intel(R) Xeon(R) Platinum 8375C CPU @ 3.50GHz
        #   L1d cache:                       32K
        #   L1i cache:                       32K
        #   L2 cache:                        512K
        #   L3 cache:                        55M
        #   NUMA node0 CPU(s):               0-55
        
        result = {}
        
        for line in text.splitlines():
            line = line.strip()
            if ':' not in line:
                # 跳过不包含冒号的行（可能是空行或其他格式）
                continue
            
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            # 只提取我们关心的字段，避免存储无关信息
            # 这样可以保持输出稳定和精简
            important_keys = {
                "Architecture",      # CPU 架构（x86_64, aarch64 等）
                "Model name",        # CPU 型号名称
                "L1d cache",        # L1 数据缓存
                "L1i cache",        # L1 指令缓存
                "L2 cache",         # L2 缓存
                "L3 cache",         # L3 缓存
                "NUMA nodes",       # NUMA 节点数
                "NUMA node(s)",     # lscpu output may use this exact key
                "CPU(s)",           # CPU 核心数
                "Stepping",         # CPU stepping
                "CPU MHz",          # CPU 主频
                "Flags",            # CPU 功能标志（另一种格式）
            }
            
            if key in important_keys:
                result[key] = value
        
        return result

    def _build_response(self, probes: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Normalize raw probe outputs into a stable JSON payload for the agent."""
        # 从三个探针的结果中提取相关信息，组装成结构化的 JSON 响应
        
        cpuinfo_output = probes.get("cpuinfo", {}).get("stdout", "")
        lscpu_output = probes.get("lscpu", {}).get("stdout", "")
        uname_output = probes.get("uname", {}).get("stdout", "").strip()
        
        # 解析 CPU 信息
        cpu_flags = self._parse_cpuinfo_flags(cpuinfo_output)
        lscpu_summary = self._parse_lscpu_summary(lscpu_output)
        
        # 识别支持的 ISA 扩展集合
        # 这些是我们关心的高性能计算相关的指令集
        isa_support = {
            "avx512f": "avx512f" in cpu_flags,  # AVX-512 Foundation
            "avx2": "avx2" in cpu_flags,        # AVX2
            "avx": "avx" in cpu_flags,          # AVX
            "sse4_2": "sse4_2" in cpu_flags,    # SSE 4.2
            "sse4_1": "sse4_1" in cpu_flags,    # SSE 4.1
            "neon": "neon" in cpu_flags,        # ARM NEON（ARM 架构）
            "asimd": "asimd" in cpu_flags,      # ARM Advanced SIMD（ARM64）
        }

        # 判断是否有真实硬件支持证据
        has_real_hardware_evidence = bool(cpu_flags or lscpu_summary)

        target_cpu_hint = "native" if has_real_hardware_evidence else None

        vectorization_order = ["avx512f", "avx2", "avx", "sse4_2", "sse4_1", "asimd", "neon"]
        vectorization_flags = [flag for flag in vectorization_order if flag in cpu_flags]

        # 如果 lscpu 或 cpuinfo 输出过长，进行截断
        cpuinfo_truncated = self._truncate(cpuinfo_output)
        lscpu_truncated = self._truncate(lscpu_output)

        # 组装最终的响应 JSON
        response = {
            "arch": lscpu_summary.get("Architecture", uname_output),
            "model_name": lscpu_summary.get("Model name"),
            "target_cpu_hint": target_cpu_hint,
            "vectorization_flags": vectorization_flags,
            "cache_summary": {
                "L1d": lscpu_summary.get("L1d cache"),
                "L1i": lscpu_summary.get("L1i cache"),
                "L2": lscpu_summary.get("L2 cache"),
                "L3": lscpu_summary.get("L3 cache"),
            },
            "vectorization_recommendation": (
                "vectorization_supported"
                if has_real_hardware_evidence
                else "vectorization_support_unclear"
            ),
            "cpu_count": lscpu_summary.get("CPU(s)"),
            "numa_nodes": lscpu_summary.get("NUMA nodes"),
            "isa_support": isa_support,
            "target_cpu_hint_native": target_cpu_hint,
            "flags_count": len(cpu_flags),
            "raw_cpuinfo": cpuinfo_truncated,
            "raw_lscpu": lscpu_truncated,
            "raw_uname": uname_output,
        }

        return response


def make_cpu_info_tool(
    cwd: Path,
    cache_dir: Path,
    git_snapshotter: Optional[GitSnapshotter] = None,
    wandb_metrics_hook: Optional[WandbRunHook] = None,
    max_output_tokens: int = MAX_TOOL_RESULT_TOKENS,
) -> FunctionTool:
    impl = CpuInfoTool(
        cwd=cwd,
        cache_dir=cache_dir,
        git_snapshotter=git_snapshotter,
        wandb_metrics_hook=wandb_metrics_hook,
        max_output_tokens=max_output_tokens,
    )

    async def on_invoke(ctx: RunContextWrapper[Any], args_json: str) -> str:
        del ctx
        try:
            parsed = load_function_tool_args(args_json)
            args = CpuInfoArgs.model_validate(parsed)
            return await impl(timeout_ms=args.timeout_ms)
        except json.JSONDecodeError as exc:
            return (
                f"Error: Invalid JSON format. {str(exc)}. "
                "Please ensure the arguments are valid JSON."
            )
        except Exception as exc:
            logger.exception("cpu_info tool failed")
            return f"Error collecting cpu info: {str(exc)}"

    return FunctionTool(
        name="cpu_info",
        description="Collects read-only CPU architecture, flags, cache, and NUMA information",
        params_json_schema=CpuInfoArgs.model_json_schema(),
        on_invoke_tool=on_invoke,
    )
