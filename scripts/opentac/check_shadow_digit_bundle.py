"""Verify the portable 5D DIGIT calibration/results bundle without Isaac or CUDA."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    folder = root / 'docs/results/shadow_digit_area_rate_20261024'
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    for relative, expected in manifest['files'].items():
        path = root / relative
        raw = path.read_bytes()
        if len(raw) != expected['bytes'] or hashlib.sha256(raw).hexdigest() != expected['sha256']:
            raise ValueError(f'Calibration bundle checksum mismatch: {relative}')
    model = json.loads((root / 'env_configs/opentac/shadow_digit_key_top_force_5d_area_rate.v1.json').read_text(encoding='utf-8'))
    if (model['schema'] != manifest['calibration_schema'] or
            model['feature_names'] != manifest['feature_names'] or
            model['sensor_sha256'] != manifest['sensor_sha256'] or
            model['training_seeds'] != manifest['training_seeds'] or
            model['held_out_seeds'] != manifest['held_out_seeds'] or
            model['frame_labels_included'] is not False):
        raise ValueError('Frozen force model differs from the migration manifest')
    print(f"Verified {len(manifest['files'])} bundle files")
    print(f"calibration_version={model['schema']}")
    print(f"sensor_sha256={model['sensor_sha256']}")
    print('No trajectory HDF5 files are required to run the frozen estimator.')


if __name__ == '__main__':
    main()
