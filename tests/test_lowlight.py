import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from enhance_lowlight import enhance_image, refine_illumination, main, write_png
from ges import calibrate, render


class LowLightTests(unittest.TestCase):
    def test_black_highlights_and_disabled_enhancement(self):
        for value in (0,255):
            source=np.full((32,48,3),value,np.uint8)
            np.testing.assert_array_equal(enhance_image(source),source)
        source=np.random.default_rng(9).integers(0,256,(40,50,3),dtype=np.uint8)
        np.testing.assert_array_equal(enhance_image(source,strength=0),source)
        flat=enhance_image(np.full((32,48,3),8,np.uint8),denoise=0)
        self.assertGreater(flat.mean(),8)
        self.assertLess(flat.max(),24)  # Near black must not turn into middle gray.

    def test_no_spatial_warp_and_color_ratios_preserved(self):
        source=np.zeros((80,112,3),np.uint8)
        source[12:31,15:43]=[12,24,36]
        source[48:70,77:100]=[40,20,10]
        result=enhance_image(source,denoise=0)
        self.assertEqual(result.shape,source.shape)
        np.testing.assert_array_equal(np.any(result>0,axis=2),np.any(source>0,axis=2))
        # A common per-pixel RGB gain preserves hue up to uint8 rounding.
        gains=result[source.min(axis=2)>0].astype(float)/source[source.min(axis=2)>0]
        self.assertLess(np.ptp(gains,axis=1).max(),.11)

    def test_illumination_flat_field_and_invalid_parameters(self):
        illumination=refine_illumination(np.full((24,32),.07,np.float32))
        np.testing.assert_allclose(illumination,.07,atol=1e-6)
        image=np.zeros((32,40,3),np.uint8)
        for options in ({'strength':float('nan')},{'max_gain':.9},{'denoise':-1},{'work_size':16}):
            with self.assertRaises(ValueError): enhance_image(image,**options)

    def test_registration_helpers_never_become_rendered_textures(self):
        world=cv2.GaussianBlur(np.random.default_rng(16).integers(0,256,(180,300,3),dtype=np.uint8),(3,3),0)
        helpers=[world[:,:220].copy(),world[:,80:].copy()]
        natural=[np.full_like(im,[13,23,33]) for im in helpers]
        with patch('ges.model.structures',return_value=[]) as edges, contextlib.redirect_stdout(io.StringIO()):
            model=calibrate(natural,matching_images=helpers,edges='canny',cell=64)
        self.assertEqual(model['report']['matching_views'],'separate_photometric_views')
        self.assertGreater(model['report']['matches'],15)
        self.assertIs(edges.call_args_list[0].args[0],natural[0])
        self.assertIs(edges.call_args_list[1].args[0],natural[1])
        result=render(model,natural)
        occupied=result[np.any(result>0,axis=2)]
        self.assertGreater(len(occupied),40000)
        np.testing.assert_array_equal(occupied,np.tile([13,23,33],(len(occupied),1)))
        for bad in ([helpers[0]],[None,helpers[1]],[helpers[0][:90],helpers[1]]):
            with self.assertRaisesRegex(ValueError,'Matching views'): calibrate(natural,matching_images=bad)

    def test_batch_separates_helpers_excludes_truth_and_preserves_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'source'; source.mkdir()
            output=Path(folder)/'natural'
            image=np.random.default_rng(4).integers(0,70,(32,48,3),dtype=np.uint8)
            for name in ('00.png','01.png','ground_truth.png'): write_png(source/name,image)
            before={p.name:p.read_bytes() for p in source.iterdir()}
            with contextlib.redirect_stdout(io.StringIO()):
                status=main(['--input',str(source),'--output',str(output)])
            self.assertEqual(status,0)
            self.assertEqual(sorted(p.name for p in output.glob('*.png')),['00.png','01.png'])
            self.assertEqual(sorted(p.name for p in (output/'registration').glob('*.png')),['00.png','01.png'])
            self.assertEqual(before,{p.name:p.read_bytes() for p in source.iterdir()})
            report=json.loads((output/'enhancement-report.json').read_text(encoding='utf-8'))
            self.assertTrue(report['complete'])
            self.assertEqual(len(report['images']),2)
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(['--input',str(source),'--output',str(source)]),1)
                self.assertEqual(main(['--input',str(source),'--output',str(output)]),1)


if __name__=='__main__': unittest.main()
