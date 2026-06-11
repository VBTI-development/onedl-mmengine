from __future__ import annotations
import argparse
import tarfile
import zipfile
from pathlib import Path

EXPECTED_FILES = {
    'mmengine/dist/__init__.py',
    'mmengine/dist/dist.py',
    'mmengine/dist/utils.py',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('dist_dir', nargs='?', default='dist')
    return parser.parse_args()


def require_one(paths: list[Path], pattern: str) -> Path:
    if len(paths) != 1:
        raise SystemExit(
            f'Expected exactly one {pattern} artifact, found {len(paths)}')
    return paths[0]


def verify_wheel(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as zf:
        names = set(zf.namelist())

    missing = EXPECTED_FILES - names
    if missing:
        raise SystemExit(f'{wheel.name} missing {sorted(missing)}')


def verify_sdist(sdist: Path) -> None:
    with tarfile.open(sdist) as tf:
        names = set(tf.getnames())

    prefix = sdist.name.removesuffix('.tar.gz')
    missing = {
        name
        for name in EXPECTED_FILES if f'{prefix}/{name}' not in names
    }
    if missing:
        raise SystemExit(f'{sdist.name} missing {sorted(missing)}')


def main() -> None:
    args = parse_args()
    dist_dir = Path(args.dist_dir)
    wheel = require_one(sorted(dist_dir.glob('onedl_mmengine-*.whl')), 'wheel')
    sdist = require_one(
        sorted(dist_dir.glob('onedl_mmengine-*.tar.gz')), 'sdist')
    verify_wheel(wheel)
    verify_sdist(sdist)


if __name__ == '__main__':
    main()
