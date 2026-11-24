"""Restore the optional raw data without overwriting conflicting local files."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory

MANIFEST = Path('data-bundles/validation-data.json')


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            result.update(chunk)
    return result.hexdigest()


def load_manifest(root):
    return json.loads((root / MANIFEST).read_text())


def figure_path(name):
    path = Path(name)
    if len(path.parts) < 2 or path.parts[0] != 'figures' or '..' in path.parts:
        raise ValueError(f'Unsafe data path: {name}')
    return path


def restore(root, archive):
    manifest = load_manifest(root)
    if sha256(archive) != manifest['sha256']:
        raise ValueError('Archive checksum does not match the validation manifest')

    records = {record['path']: record for record in manifest['files']}
    destinations = {}
    targets = set()
    root = root.resolve()
    for name, record in records.items():
        member = figure_path(name)
        relative = figure_path(record['restore_path']) if 'restore_path' in record else member
        target = root / relative
        if target.resolve() != target:
            raise ValueError(f'Unsafe data destination: {relative}')
        if target in targets:
            raise ValueError(f'Duplicate data destination: {relative}')
        if target.exists() and (not target.is_file() or sha256(target) != record['sha256']):
            raise ValueError(f'Will not overwrite different local data: {relative}')
        destinations[name] = target
        targets.add(target)

    # Read only the expected regular files. Verify everything before restoring.
    with TemporaryDirectory(prefix='.validation-restore-', dir=root) as temporary:
        staging = Path(temporary)
        pending = {}
        seen = set()
        with tarfile.open(archive, 'r|gz') as bundle:
            for entry in bundle:
                record = records.get(entry.name)
                if (record is None or entry.name in seen or not entry.isfile()
                        or entry.size != record['bytes']):
                    raise ValueError(f'Unexpected archive entry: {entry.name}')
                seen.add(entry.name)
                contents = bundle.extractfile(entry)
                destination = destinations[entry.name]
                saved = None if destination.is_file() else staging / str(len(seen))
                result = hashlib.sha256()
                with contents, (saved.open('wb') if saved else nullcontext()) as stream:
                    for chunk in iter(lambda: contents.read(2**20), b''):
                        result.update(chunk)
                        if stream is not None:
                            stream.write(chunk)
                if result.hexdigest() != record['sha256']:
                    raise ValueError(f'Data checksum mismatch: {entry.name}')
                if saved is not None:
                    pending[entry.name] = saved
        if seen != records.keys():
            raise ValueError('Bundle is missing files from the validation manifest')

        for name, saved in pending.items():
            destination = destinations[name]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.resolve() != destination:
                raise ValueError(f'Unsafe data destination: {name}')
            destination.hardlink_to(saved)
    return len(records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path, help='Path to the data archive listed in the manifest')
    args = parser.parse_args()
    try:
        count = restore(Path(__file__).resolve().parents[1], args.archive)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f'Verified and restored {count} files.')
