#!/usr/bin/env python3
import runpy
from pathlib import Path
runpy.run_path(Path(__file__).with_name("render_figure.py"), run_name="__main__")
