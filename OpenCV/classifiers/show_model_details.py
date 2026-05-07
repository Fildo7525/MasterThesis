from pathlib import Path
import sys

from svm_pretrain import SVMDetector
from gmm_pretrain import GMMDetector
from isolation_forest_pretrain import IsolationForestDetector

import joblib

def help():
    return """
Program to show isides of the detector generated pretrain joblib file
python show_model_details.py <path to model>
"""

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(help())
        exit(1)


    model_path = Path(sys.argv[1]).absolute()

    meta = joblib.load(model_path)

    pipeline = meta["pipeline"]
    band_indices       = meta["band_indices"]
    vegetation_indices = meta["vegetation_indices"]

    print(f"""
    Model: {model_path}

    Pipeline: {pipeline}
    Bands: {band_indices}
    VIs: {vegetation_indices}
    """)
