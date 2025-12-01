# ------------------------------------------------------------------------
# Copyright (c) 2024-present, BAAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Utility to stop distributed cache jobs."""

import os
import shlex
import subprocess
from typing import Iterable


def _call_host(host: str, remote_cmd: str) -> int:
    """Execute a command on a remote host via SSH."""
    quoted_remote = shlex.quote(remote_cmd)
    full_cmd = f"ssh {host} {quoted_remote}"
    return subprocess.call(full_cmd, shell=True)


def _iter_hosts(hostfile_path: str) -> Iterable[str]:
    with open(hostfile_path, "r", encoding="utf-8") as handle:
        for line in handle:
            entry = line.strip()
            if entry and not entry.startswith("#"):
                yield entry


if __name__ == "__main__":
    repo_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    hostfile = "/share/project/denghaoge/shilinlu/proj/simple_ursa/accelerate_configs/shilin_10node.hostfile"

    cache_script = os.path.join(repo_root, "data_process", "cache_videos.py")
    cache_launcher = os.path.join(repo_root, "data_process", "cache.sh")

    kill_patterns = {
        cache_script,
        cache_launcher,
        "cache_decord320.py",
        "cache.sh",
    }

    for host in _iter_hosts(hostfile):
        for pattern in kill_patterns:
            remote_cmd = f"pkill -f {shlex.quote(pattern)} || true"
            print(f"[{host}] stopping processes matching '{pattern}'")
            _call_host(host, remote_cmd)
