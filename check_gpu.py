import subprocess
from pathlib import Path
from typing import List

HOSTFILE = Path("/share/project/denghaoge/shilinlu/proj/simple_ursa/accelerate_configs/9_nodes_deepspeed.hostfile")
NVIDIA_SMI_QUERY = "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits"


def load_hosts(hostfile: Path) -> List[str]:
    if not hostfile.exists():
        raise FileNotFoundError(f"Hostfile not found: {hostfile}")
    with hostfile.open() as fh:
        return [line.strip() for line in fh if line.strip()]


def fetch_gpu_stats(host: str) -> str:
    cmd = f"ssh {host} \"{NVIDIA_SMI_QUERY}\""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip() or "Unknown error"
        header = f"[{host}]"
        body = f"  error -> {stderr}"
        return "\n".join((header, body))

    rows = []
    for line in result.stdout.strip().splitlines():
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            rows.append(f"  unexpected output -> {line}")
            continue
        idx, name, mem_used, mem_total, util = parts
        rows.append(f"  GPU {idx} {name} | {mem_used}/{mem_total} MiB | {util}%")
    body = "\n".join(rows) if rows else "  no GPU data"
    separator = "-" * 40
    header = f"[{host}]"
    return "\n".join((separator, header, body))


def main() -> None:
    hosts = load_hosts(HOSTFILE)
    for host in hosts:
        print(fetch_gpu_stats(host))


if __name__ == "__main__":
    main()
