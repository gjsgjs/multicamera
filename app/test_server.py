import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from server import Server

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ges
import cv2
import numpy as np


class AppTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.server = Server(('127.0.0.1', 0), Path(self.folder.name))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join()
        self.folder.cleanup()

    def call(self, path, method='GET', data=None, headers=None):
        headers = headers or {}
        if isinstance(data, dict):
            data = json.dumps(data).encode()
            headers['Content-Type'] = 'application/json'
        request = urllib.request.Request(self.base + path, data=data, method=method, headers=headers)
        return urllib.request.urlopen(request, timeout=10)

    def create_and_upload(self, mode):
        with self.call('/api/jobs', 'POST', {'files': ['left.png', 'right.png'], 'mode': mode, 'columns': 2}) as response:
            job = json.load(response)
        for index in range(2):
            image = np.full((48, 64, 3), 40 + index * 100, np.uint8)
            ok, data = cv2.imencode('.png', image)
            self.assertTrue(ok)
            with self.call(f"/api/jobs/{job['id']}/files/{index}", 'PUT', data.tobytes()) as response:
                self.assertEqual(response.status, 200)
        return job

    def run_job(self, job):
        with self.call(f"/api/jobs/{job['id']}/run", 'POST', {}) as response:
            self.assertEqual(response.status, 200)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            with self.call(f"/api/jobs/{job['id']}") as response:
                current = json.load(response)
            if current['status'] in ('success', 'failed'):
                return current
            time.sleep(.1)
        self.fail('Background stitch did not complete in time')

    def preprocess_job(self, job, method='dehaze'):
        with self.call(f"/api/jobs/{job['id']}/preprocess", 'POST', {'method': method}) as response:
            self.assertEqual(response.status, 200)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            with self.call(f"/api/jobs/{job['id']}") as response:
                current = json.load(response)
            if current['status'] in ('ready', 'preprocess_failed'):
                return current
            time.sleep(.1)
        self.fail('Preprocessing did not complete in time')

    def test_preprocessing_preview_then_stitch_uses_processed_pixels(self):
        uploaded = self.create_and_upload('auto')
        job = self.preprocess_job(uploaded)
        self.assertEqual(job['status'], 'ready', job)
        self.assertEqual(job['processed']['method'], 'dehaze')
        self.assertEqual(len(job['processed']['images']), 2)
        processed = []
        for item in job['processed']['images']:
            with self.call(item['preview']) as response:
                self.assertEqual(response.headers['Content-Type'], 'image/jpeg')
            with self.call(item['full']) as response:
                processed.append(cv2.imdecode(np.frombuffer(response.read(), np.uint8), cv2.IMREAD_COLOR))
        self.assertFalse(np.array_equal(processed[0], np.full((48, 64, 3), 40, np.uint8)))
        folder = self.server.store.jobs[job['id']]['folder']
        original = (folder/'inputs'/'0000.png').read_bytes()
        with self.call(f"/api/jobs/{job['id']}/run", 'POST', {'mode': 'grid', 'columns': 2, 'blend': 'sharp', 'use_processed': True}) as response:
            self.assertEqual(response.status, 200)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            with self.call(f"/api/jobs/{job['id']}") as response:
                finished = json.load(response)
            if finished['status'] in ('success', 'failed'): break
            time.sleep(.1)
        self.assertEqual(finished['status'], 'success', finished)
        with self.call(finished['result']['download']) as response:
            result = cv2.imdecode(np.frombuffer(response.read(), np.uint8), cv2.IMREAD_COLOR)
        np.testing.assert_array_equal(result, np.hstack(processed))
        self.assertEqual((folder/'inputs'/'0000.png').read_bytes(), original)

    def test_lowlight_serves_natural_images_and_stitches_their_pixels(self):
        from enhance_lowlight import enhance_image
        job = self.preprocess_job(self.create_and_upload('auto'), 'lowlight')
        self.assertEqual(job['status'], 'ready', job)
        self.assertEqual(job['processed']['method'], 'lowlight')
        self.assertEqual(job['preprocess_method'], 'lowlight')
        folder = self.server.store.jobs[job['id']]['folder']
        natural = []
        for index, item in enumerate(job['processed']['images']):
            expected = enhance_image(np.full((48,64,3),40+index*100,np.uint8))
            with self.call(item['full']) as response:
                image = cv2.imdecode(np.frombuffer(response.read(),np.uint8),cv2.IMREAD_COLOR)
            np.testing.assert_array_equal(image,expected)
            natural.append(image)
            helper = cv2.imdecode(np.fromfile(folder/'processed'/'registration'/f'{index:04d}.png',np.uint8),cv2.IMREAD_COLOR)
            self.assertFalse(np.array_equal(image,helper))
            with self.call(item['preview']) as response:
                ok, thumbnail = cv2.imencode('.jpg',expected,[cv2.IMWRITE_JPEG_QUALITY,90])
                self.assertTrue(ok)
                self.assertEqual(response.read(),thumbnail.tobytes())
        with self.call(f"/api/jobs/{job['id']}/run",'POST',{'mode':'grid','columns':2,'use_processed':True}) as response:
            self.assertEqual(response.status,200)
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            with self.call(f"/api/jobs/{job['id']}") as response: finished=json.load(response)
            if finished['status'] in ('success','failed'): break
            time.sleep(.1)
        self.assertEqual(finished['status'],'success',finished)
        with self.call(finished['result']['download']) as response:
            result=cv2.imdecode(np.frombuffer(response.read(),np.uint8),cv2.IMREAD_COLOR)
        np.testing.assert_array_equal(result,np.hstack(natural))

    def test_explicit_original_selection_does_not_use_previous_processing(self):
        job=self.preprocess_job(self.create_and_upload('grid'),'lowlight')
        self.assertEqual(job['status'],'ready',job)
        with self.call(f"/api/jobs/{job['id']}/run",'POST',{'use_processed':False}) as response:
            self.assertEqual(response.status,200)
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            with self.call(f"/api/jobs/{job['id']}") as response: finished=json.load(response)
            if finished['status'] in ('success','failed'): break
            time.sleep(.1)
        self.assertEqual(finished['status'],'success',finished)
        with self.call(finished['result']['download']) as response:
            image=cv2.imdecode(np.frombuffer(response.read(),np.uint8),cv2.IMREAD_COLOR)
        np.testing.assert_array_equal(image[:,:64],40)
        np.testing.assert_array_equal(image[:,64:],140)

    def test_preprocess_failure_is_reported_and_cannot_stitch(self):
        with self.call('/api/jobs', 'POST', {'files': ['a.png', 'b.png']}) as response:
            job = json.load(response)
        for index in range(2):
            with self.call(f"/api/jobs/{job['id']}/files/{index}", 'PUT', b'not an image') as response:
                self.assertEqual(response.status, 200)
        failed = self.preprocess_job(job, 'lowlight')
        self.assertEqual(failed['status'], 'preprocess_failed')
        self.assertTrue(failed['error'])
        self.assertIsNone(failed['processed'])
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.call(f"/api/jobs/{job['id']}/run", 'POST', {'use_processed': True})
        self.assertEqual(error.exception.code, 400)

    def test_processed_input_requires_completed_preprocessing(self):
        job = self.create_and_upload('grid')
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.call(f"/api/jobs/{job['id']}/run", 'POST', {'use_processed': True})
        self.assertEqual(error.exception.code, 400)
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.call(f"/api/jobs/{job['id']}/preprocess", 'POST', {'method': 'unknown'})
        self.assertEqual(error.exception.code, 400)

    def test_success_serves_full_resolution_png_and_preview(self):
        job = self.run_job(self.create_and_upload('grid'))
        self.assertEqual(job['status'], 'success', job)
        self.assertEqual((job['result']['width'], job['result']['height']), (128, 48))
        with self.call(job['result']['download']) as response:
            self.assertIn('attachment', response.headers['Content-Disposition'])
            image = cv2.imdecode(np.frombuffer(response.read(), np.uint8), cv2.IMREAD_COLOR)
        np.testing.assert_array_equal(image[:, :64], 40)
        np.testing.assert_array_equal(image[:, 64:], 140)
        with self.call(job['result']['preview']) as response:
            self.assertEqual(response.headers['Content-Type'], 'image/jpeg')
        self.assertTrue((Path(job['result']['output']).parent / 'status.json').exists())

    def test_failure_exposes_real_error_and_log(self):
        job = self.run_job(self.create_and_upload('auto'))
        self.assertEqual(job['status'], 'failed')
        self.assertIn('disconnected', job['error'])
        self.assertTrue(any('Error:' in line for line in job['logs']))
        self.assertIsNone(job['result'])

    def test_validation_and_page_assets(self):
        for asset in ('/', '/app.js', '/style.css'):
            with self.call(asset) as response:
                self.assertEqual(response.status, 200)
                self.assertGreater(len(response.read()), 100)
        for data in ({'files': ['single.png']}, {'files': ['../a.png', 'b.png']},
                     {'files': ['a.png', 'a.png']}):
            with self.assertRaises(urllib.error.HTTPError) as error:
                self.call('/api/jobs', 'POST', data)
            self.assertEqual(error.exception.code, 400)
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.call('/api/jobs', 'POST', {'files': ['a.png', 'b.png']}, {'Origin': 'https://example.com'})
        self.assertEqual(error.exception.code, 403)
        with self.call('/api/jobs', 'POST', {'files': ['a.png', 'b.png']}) as response:
            job = json.load(response)
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.call(f"/api/jobs/{job['id']}/run", 'POST', {})
        self.assertEqual(error.exception.code, 400)


if __name__ == '__main__':
    unittest.main()
