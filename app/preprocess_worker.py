"""Dispatch preprocessing; create UI previews from final rendering images only."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dehaze import main as dehaze_main
from enhance_lowlight import main as lowlight_main
from ges.cli import read
import cv2

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--method', choices=['dehaze', 'lowlight'], default='dehaze')
    options, arguments = parser.parse_known_args()
    code = (lowlight_main if options.method == 'lowlight' else dehaze_main)(arguments)
    if code == 0:
        try:
            folder = Path(arguments[arguments.index('--output') + 1])
            previews = folder / 'previews'; previews.mkdir(exist_ok=True)
            for path in sorted(folder.glob('*.png')):
                image = read(path)
                scale = min(1., 640 / max(image.shape[:2]))
                image = cv2.resize(image, (max(1, round(image.shape[1]*scale)), max(1, round(image.shape[0]*scale))))
                ok, data = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if not ok: raise ValueError(f'Cannot generate preview: {path.name}')
                data.tofile(previews / (path.stem + '.jpg'))
                print(f'Preview ready: {path.name}', flush=True)
        except Exception as exc:
            print(f'Error: {exc}', file=sys.stderr)
            code = 1
    raise SystemExit(code)
