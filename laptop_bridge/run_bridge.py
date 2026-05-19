# Spawn camera_bridge.py + franka_bridge.py as subprocesses, forward SIGINT.
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def main():
    procs = []
    for script in ("camera_bridge.py", "franka_bridge.py"):
        p = subprocess.Popen([PY, os.path.join(HERE, script)])
        print(f"[run_bridge] started {script} pid={p.pid}")
        procs.append((script, p))
        time.sleep(1.0)

    stop = {"flag": False}

    def _shutdown(signum, frame):
        if stop["flag"]:
            return
        stop["flag"] = True
        print("[run_bridge] SIGINT — stopping children")
        for _, p in procs:
            try:
                p.send_signal(signal.SIGINT)
            except Exception:
                pass

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        while not stop["flag"]:
            for name, p in procs:
                rc = p.poll()
                if rc is not None:
                    print(f"[run_bridge] {name} exited rc={rc} — shutting down")
                    _shutdown(None, None)
                    break
            time.sleep(0.5)
    finally:
        deadline = time.monotonic() + 5.0
        for name, p in procs:
            try:
                p.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                print(f"[run_bridge] killing {name}")
                p.kill()


if __name__ == "__main__":
    main()
