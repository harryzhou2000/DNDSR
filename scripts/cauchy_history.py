#!/usr/bin/env python3
"""Serve archived Cauchy history with Python's standard library only."""
import runpy
from pathlib import Path

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve(
    ).parents[1] / 'python/DNDSR/Cauchy/monitor/history_service.py'), run_name='__main__')
