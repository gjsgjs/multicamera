"""Download the edge model from the author's repository. No executables are run."""
from pathlib import Path
import urllib.request

root=Path(__file__).resolve().parent/'models'
root.mkdir(exist_ok=True)
for name in ('deploy.prototxt','hed_pretrained_bsds.caffemodel'):
    path=root/name
    if path.exists():
        print(f'Already present: {path}'); continue
    url='https://raw.githubusercontent.com/flowerDuo/GES-GSP-Stitching/master/Code/model/'+name
    data=urllib.request.urlopen(url,timeout=120).read()
    path.write_bytes(data)
    print(f'Downloaded: {path} ({len(data)} bytes)')
