"""Run a Python script in a child process and kill it above an RSS cap.

usage: python run_capped.py <cap_gb> <script.py> [args...]
The watchdog lives in this parent process, so a long C++ call in the child
(which holds the GIL) cannot stop it.
"""
import subprocess
import sys
import time

import psutil

cap = float(sys.argv[1]) * 2**30
cmd = [sys.executable, "-u", *sys.argv[2:]]
child = subprocess.Popen(cmd)
proc = psutil.Process(child.pid)
peak = 0
while child.poll() is None:
    try:
        rss = proc.memory_info().rss
    except psutil.Error:
        break
    peak = max(peak, rss)
    if rss > cap:
        print(f"[run_capped] RSS {rss / 2**30:.2f} GB above cap, killing", flush=True)
        child.kill()
        child.wait()
        sys.exit(99)
    time.sleep(0.2)
print(f"[run_capped] exit {child.returncode}, peak RSS {peak / 2**30:.2f} GB", flush=True)
sys.exit(child.returncode)
