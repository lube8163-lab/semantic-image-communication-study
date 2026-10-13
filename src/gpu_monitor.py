"""Read-only NVIDIA temperature checks and optional existing CoolingDesk telemetry.

Public-export adapter. It does not change fans, clocks or cooling policies.
The measured V100 run used the optional fresh CoolingDesk GPU/HBM checks.
"""
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

def check(role='feature'):
    selector = os.environ.get('SC_' + role.upper() + '_GPU', os.environ.get('CUDA_VISIBLE_DEVICES', '0')).split(',')[0]
    limit = 70 if role == 'feature' else 80
    fields = ['uuid', 'name', 'temperature.gpu', 'memory.used', 'power.draw', 'utilization.gpu']
    row = subprocess.check_output(['nvidia-smi', '-i', selector, '--query-gpu=' + ','.join(fields), '--format=csv,noheader,nounits'], text=True).strip()
    values = [v.strip() for v in row.split(',')]
    if len(values) != len(fields) or float(values[2]) >= limit:
        raise RuntimeError('GPU temperature limit reached or ambiguous telemetry')
    result = dict(zip(fields, values))
    snapshot = os.environ.get('SC_COOLING_STATE')
    if snapshot and role == 'feature':
        for attempt in range(5):
            try:
                record = json.loads(Path(snapshot).read_text(encoding='utf-8-sig'))
                break
            except (PermissionError, FileNotFoundError, json.JSONDecodeError):
                if attempt == 4:
                    raise
                time.sleep(0.05)
        state = record['State']
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(record['At'])).total_seconds()
        gpu = next(g for g in state['Snapshot']['Gpus'] if g['Id'] == values[0])
        if age > 30 or age < -5 or not state['MotherboardActive'] or state['Plan']['Emergency']:
            raise RuntimeError('Cooling telemetry stale, inactive or emergency')
        if max(gpu['Temperature'], gpu.get('MemoryTemperature') or 0) >= 70:
            raise RuntimeError('GPU/HBM temperature limit reached')
        result['cooling_age_seconds'] = age
    return result
