import json
from datetime import datetime, timedelta, timezone
import pytest
import gpu_monitor

def prepare(monkeypatch, tmp_path, age=0, hbm=40, emergency=False, board=True):
    monkeypatch.setenv('SC_FEATURE_GPU', '0')
    monkeypatch.setattr(gpu_monitor.subprocess, 'check_output', lambda *args, **kwargs: 'test-id, Test GPU, 40, 1000, 50, 0')
    path = tmp_path / 'cooling.json'
    path.write_text(json.dumps({'At': (datetime.now(timezone.utc) - timedelta(seconds=age)).isoformat(), 'State': {'MotherboardActive': board, 'Plan': {'Emergency': emergency}, 'Snapshot': {'Gpus': [{'Id': 'test-id', 'Temperature': 40, 'MemoryTemperature': hbm}]}}}), encoding='utf-8')
    monkeypatch.setenv('SC_COOLING_STATE', str(path))

def test_fresh_matching_hbm_telemetry_is_accepted(monkeypatch, tmp_path):
    prepare(monkeypatch, tmp_path)
    assert gpu_monitor.check()['cooling_age_seconds'] < 1

@pytest.mark.parametrize('conditions', [{'age': 31}, {'hbm': 70}, {'emergency': True}, {'board': False}])
def test_original_hard_stops_are_preserved(monkeypatch, tmp_path, conditions):
    prepare(monkeypatch, tmp_path, **conditions)
    with pytest.raises(RuntimeError):
        gpu_monitor.check()

def test_hot_nvidia_telemetry_stops_without_coolingdesk(monkeypatch):
    monkeypatch.delenv('SC_COOLING_STATE', raising=False)
    monkeypatch.setattr(gpu_monitor.subprocess, 'check_output', lambda *args, **kwargs: 'test-id, Test GPU, 70, 1000, 50, 0')
    with pytest.raises(RuntimeError):
        gpu_monitor.check()
