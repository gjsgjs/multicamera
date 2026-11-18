from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ges.cli import main, read
import cv2

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    status = main(sys.argv[1:])
    if status == 0:
        try:
            target = Path(sys.argv[sys.argv.index('-o') + 1])
            image = read(target)
            scale = min(1., 1800 / max(image.shape[:2]))
            preview = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))))
            ok, data = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ok:
                data.tofile(target.parent / 'preview.jpg')
        except Exception as exc:
            print(f'Preview warning: {exc}', flush=True)
    raise SystemExit(status)
