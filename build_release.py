"""Package only application source, never local datasets or recordings."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

FILES = (
    'main.py', 'virtual_camera.py', 'runtime.py', 'download_model.py',
    'requirements.txt', 'requirements-virtual.txt', 'README.md', '.gitignore',
    'run_virtual_camera.bat', 'tests/test_app.py', 'build_release.py',
)


def main():
    root = Path(__file__).parent
    destination = root / 'dist' / 'hand-motion-source.zip'
    destination.parent.mkdir(exist_ok=True)
    with ZipFile(destination, 'w', ZIP_DEFLATED) as archive:
        for name in FILES:
            archive.write(root / name, f'hand-motion/{name}')
    print(destination)


if __name__ == '__main__':
    main()
