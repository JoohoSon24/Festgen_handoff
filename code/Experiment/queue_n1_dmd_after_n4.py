"""Hold the N1 DMD run until N4 GT-GAN has a completion marker."""
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/root/Diffusion 3')
EXP = ROOT / 'Experiment'
RUN = ROOT / 'FID/student90_n1_dmd_after_n4_20261026'
N4_RUN = ROOT / 'FID/student90_n1_budget_extension_20261025'
MARKER = N4_RUN / 'n4_extension_completed.json'
STATUS = RUN / 'queue_status.json'
SCRIPT = EXP / 'run_n1_dmd_after_n4.py'


def write(value):
    RUN.mkdir(parents=True, exist_ok=True)
    value['updated_utc'] = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    temp = STATUS.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(STATUS)


def main():
    if (RUN / 'completed.json').exists():
        write({'status': 'already_completed'})
        return
    write({'status': 'queued', 'waits_for': str(MARKER), 'next': 'N1 DMD-inspired fine-tune after N4 extra 1,000 updates'})
    while not MARKER.exists():
        n4_status_path = N4_RUN / 'status.json'
        if n4_status_path.exists():
            n4_status = json.loads(n4_status_path.read_text())
            if n4_status.get('status') == 'failed':
                write({'status': 'waiting_for_n4_recovery', 'n4_status': n4_status.get('status'),
                       'reason': 'N4 failed; DMD will not start without the completion marker.'})
                return
        write({'status': 'queued', 'waits_for': str(MARKER), 'next': 'N1 DMD-inspired fine-tune after N4 extra 1,000 updates'})
        time.sleep(30)

    write({'status': 'waiting_for_gpu_lock', 'n4_completion_marker': str(MARKER)})
    with (EXP / 'gt_gan.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        write({'status': 'running', 'n4_completion_marker': str(MARKER), 'pid': os.getpid()})
        log = RUN / 'queue.log'
        with log.open('a', buffering=1) as output:
            result = subprocess.run([sys.executable, '-u', str(SCRIPT)], cwd=ROOT,
                                    stdout=output, stderr=subprocess.STDOUT, check=False)
        write({'status': 'completed' if result.returncode == 0 else 'failed',
               'returncode': result.returncode, 'log': str(log)})
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
