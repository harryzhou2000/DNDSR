"""Stdlib-only history protocol tests; no native solver/binding imports."""
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
MODULES = ROOT / 'python/DNDSR/Cauchy/monitor'


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, MODULES / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


service, catalog = load('history_service'), load('catalog')


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.csv = self.root / 'history.csv'
        self.csv.write_text('iteration,status,time,res_rho\n1,0,0,1\n2,0,0,')
        self.index = service.HistoryIndex()

    def tearDown(self):
        self.temp.cleanup()

    def test_partial_append_rotation_filter_and_reset(self):
        first = self.index.query(self.csv, {'limit': '1'})
        self.assertEqual(first['totalRows'], 1)
        with self.csv.open('a') as f:
            f.write('.5\n3,2,1,.1\n')
        second = self.index.query(
            self.csv, {'cursor': first['nextCursor'], 'kind': 'nonlinear'})
        self.assertEqual(second['rows'][0]['iteration'], 2)
        self.assertFalse(second['hasMore'])
        (self.root / 'history_1.csv').write_text('iteration,status,time,res_rho\n4,0,2,.01\n')
        third = self.index.query(self.csv, {'cursor': second['nextCursor']})
        self.assertEqual(third['rows'][0]['iteration'], 4)
        selected = self.index.query(
            self.csv, {'start': '2', 'end': '3', 'columns': 'res_rho'})
        self.assertEqual(selected['rows'], [{'res_rho': .5}, {'res_rho': .1}])
        self.csv.write_text('iteration,status,time,res_rho\n9,0,0,2\n')
        reset = self.index.query(self.csv, {'cursor': third['nextCursor']})
        self.assertTrue(reset['reset'])
        self.assertEqual(reset['rows'][0]['iteration'], 9)

    def test_sparse_pages_and_bad_queries(self):
        self.csv.write_text('iteration,status,time,res_rho\n' +
                            ''.join(f'{i},0,{i},1\n' for i in range(1001)))
        a = self.index.query(self.csv, {'limit': '600'})
        b = self.index.query(
            self.csv, {'cursor': a['nextCursor'], 'limit': '600'})
        self.assertEqual(a['indices'] + b['indices'], list(range(1001)))
        self.assertEqual(
            len(self.index.series[str(self.csv)]['segments'][0]['offsets']), 4)
        restarted = service.HistoryIndex().query(
            self.csv, {'cursor': b['nextCursor']})
        self.assertTrue(restarted['reset'])
        self.assertEqual(restarted['indices'][0], 0)
        for args in ({'limit': '0'}, {'limit': '-1'}, {'cursor': 'bad'}, {'start': 'nan'}, {'columns': 'missing'}):
            with self.assertRaises(ValueError):
                self.index.query(self.csv, args)

    def test_remote_hub_never_reads_csv_and_recovers(self):
        self.csv.write_text(
            'iteration,status,time,res_rho\n1,0,0,1\n2,0,0,.5\n')
        config = self.root / 'resolved.json'
        config.write_text('{}')
        manifest = self.root / 'run-run-manifest.json'
        manifest.write_text(json.dumps({'resolvedConfig': str(config), 'domains': [
            {'index': 0, 'history': str(self.csv), 'boundaryIntegrations': [
                {'canonicalName': 'Wall', 'csv': str(self.csv)}]}]}))
        server = service.serve(service.Companion(
            manifest, 'test-token'), '127.0.0.1', 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f'http://127.0.0.1:{server.server_port}'
        try:
            with self.assertRaises(HTTPError) as failure:
                urlopen(url + '/api/v1/history/query')
            self.assertEqual(failure.exception.code, 401)
            hub = catalog.RunCatalog(
                registry_dir=self.root/'registry', state_file=self.root/'hub.json')
            with patch.object(catalog, '_read_json_safe', side_effect=AssertionError('remote hub accessed disk')):
                rid = hub.attach(url=url, token='test-token')['runId']
                data = hub.get_history(rid, max_points=0)
                self.assertEqual(data['totalRows'], 2)
                self.assertEqual(data['source'], 'http')
                with self.csv.open('a') as f:
                    f.write('3,0,0,.25\n')
                self.assertEqual(hub.get_history(rid)['totalRows'], 3)
                self.assertEqual(hub.get_boundary_history(rid)
                                 ['series']['totalRows'], 3)
                self.csv.write_text('iteration,status,time,res_rho\n8,0,0,2\n')
                self.assertEqual(hub.get_history(
                    rid)['columns']['iteration'], [8])
            reloaded = catalog.RunCatalog(
                registry_dir=self.root/'registry', state_file=self.root/'hub.json')
            self.assertEqual(reloaded.get_history(
                rid)['columns']['iteration'], [8])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
