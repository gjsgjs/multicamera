import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dehaze import dehaze_image, main
import cv2
import numpy as np


class DehazeTests(unittest.TestCase):
    def test_zero_strength_preserves_pixels_and_constant_images_are_finite(self):
        image=np.random.default_rng(6).integers(0,256,(49,71,3),dtype=np.uint8)
        result,_=dehaze_image(image,[.8,.85,.9],omega=0)
        np.testing.assert_array_equal(result,image)
        for value in (0,127,255):
            original=np.full((30,40,3),value,np.uint8)
            processed,stats=dehaze_image(original,[.9,.9,.9])
            self.assertEqual(processed.shape,original.shape)
            self.assertEqual(processed.dtype,np.uint8)
            self.assertTrue(np.isfinite(stats['transmission_mean']))

    def test_recovers_synthetic_scattering_model(self):
        clear=np.random.default_rng(2).uniform(0,1,(100,120,3)).astype(np.float32)
        clear[::5,::5]=0  # Every patch satisfies the dark-channel assumption.
        hazy=np.uint8(np.rint((.4*clear+.6*.8)*255))
        recovered,_=dehaze_image(hazy,[.8,.8,.8],omega=1,tmin=.1)
        before=np.mean((hazy.astype(float)/255-clear)**2)
        after=np.mean((recovered.astype(float)/255-clear)**2)
        self.assertLess(after,before*.05)

    def test_batch_excludes_ground_truth_and_keeps_names_sizes_and_source_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            source=Path(root)/'fog'; source.mkdir()
            image=np.random.default_rng(3).integers(100,240,(50,70,3),dtype=np.uint8)
            _,data=cv2.imencode('.png',image)
            for name in ('00.png','01.png','ground_truth.png'):
                data.tofile(source/name)
            original=(source/'00.png').read_bytes()
            self.assertEqual(main(['--input',str(source)]),0)
            output=source.with_name('fog_dehazed')
            self.assertEqual(sorted(p.name for p in output.glob('*.png')),['00.png','01.png'])
            self.assertEqual((source/'00.png').read_bytes(),original)
            for p in output.glob('*.png'):
                self.assertEqual(cv2.imdecode(np.fromfile(p,np.uint8),cv2.IMREAD_COLOR).shape,image.shape)
            self.assertTrue((output/'dehaze-report.json').exists())
            self.assertTrue((output/'diagnostics/comparison.jpg').exists())


if __name__=='__main__': unittest.main()
