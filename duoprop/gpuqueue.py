"""Tiny GPU job queue: dispatches jobs from runs/jobs.txt to GPUs with free capacity.

jobs.txt lines:  <weight> <name> <shell command>      ('#' lines ignored; file is re-read every poll,
so jobs can be appended at any time).  Each GPU has capacity 2; GPU processes not started by the
queue count as weight 1 each.  Logs go to runs/logs/<name>.log, state to runs/queue_state.json.
"""
import json
import os
import re
import shutil
import subprocess
import time

WAIT = re.compile(r'^until (.*?); do sleep \d+; done;\s*(.*)$', re.S)


def ready(cmd):
    """Jobs of the form 'until COND; do sleep N; done; REST' are launched (as REST) only once COND holds,
    so waiting jobs never hold a GPU slot."""
    m = WAIT.match(cmd)
    if not m:
        return True, cmd
    ok = subprocess.run(['bash', '-c', m.group(1)], cwd=ROOT, capture_output=True).returncode == 0
    return ok, m.group(2)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOBS = os.path.join(ROOT, 'runs', 'jobs.txt')
STATE = os.path.join(ROOT, 'runs', 'queue_state.json')
LOGS = os.path.join(ROOT, 'runs', 'logs')
CAP = 2
CAP_FILE = os.path.join(ROOT, 'runs', 'queue_cap')  # optional: per-GPU capacity, re-read every poll


GPUS_FILE = os.path.join(ROOT, 'runs', 'queue_gpus')  # optional: comma-separated GPU indices the queue may use, re-read every poll


def allowed(ngpu):
    try:
        gs = [int(x) for x in open(GPUS_FILE).read().replace(',', ' ').split()]
        return [g for g in gs if 0 <= g < ngpu]
    except (OSError, ValueError):
        return list(range(ngpu))


MIN_FREE_DISK = 300e9  # no new jobs below this much free disk (cache builds can be hundreds of GB)


def capacity():
    if shutil.disk_usage(ROOT).free < MIN_FREE_DISK:
        return 0
    try:
        return int(open(CAP_FILE).read().strip())
    except (OSError, ValueError):
        return CAP


def gpu_procs():
    uu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader']).decode()
    idx = {u.strip(): int(i) for i, u in (l.split(',') for l in uu.strip().split('\n'))}
    out = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader']).decode()
    procs = []
    for l in out.strip().split('\n'):
        if l.strip():
            pid, u = l.split(',')
            procs.append((int(pid), idx[u.strip()]))
    return procs, len(idx)


def ancestors(pid):
    seen = set()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        try:
            with open(f'/proc/{pid}/stat') as f:
                pid = int(f.read().rsplit(')', 1)[1].split()[1])
        except OSError:
            break
    return seen


def main():
    os.makedirs(LOGS, exist_ok=True)
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    running = {}  # name -> (Popen, gpu, weight)
    while True:
        for name, (p, g, w) in list(running.items()):
            rc = p.poll()
            if rc is not None:
                state[name].update(status='done' if rc == 0 else 'failed', rc=rc, end=time.strftime('%m-%d %H:%M'))
                del running[name]
        jobs = []
        if os.path.exists(JOBS):
            for line in open(JOBS):
                line = line.strip()
                if line and not line.startswith('#'):
                    w, name, cmd = line.split(None, 2)
                    jobs.append((int(w), name, cmd))
        procs, ngpu = gpu_procs()
        load = [0] * ngpu
        ours = {p.pid for p, _, _ in running.values()}
        for pid, g in procs:
            if not (ancestors(pid) & ours):
                load[g] += 1
        for p, g, w in running.values():
            load[g] += w
        for w, name, cmd in jobs:
            if name in state:
                continue
            free = [g for g in allowed(ngpu) if load[g] + w <= capacity()]
            if not free:
                continue  # backfill: smaller jobs further down may still fit
            ok, cmd = ready(cmd)
            if not ok:
                continue
            g = min(free, key=lambda x: load[x])
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
            log = open(os.path.join(LOGS, f'{name}.log'), 'a')
            p = subprocess.Popen(['bash', '-c', cmd], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            running[name] = (p, g, w)
            load[g] += w
            state[name] = dict(status='running', gpu=g, start=time.strftime('%m-%d %H:%M'), cmd=cmd)
        json.dump(state, open(STATE, 'w'), indent=1)
        time.sleep(20)


if __name__ == '__main__':
    main()
