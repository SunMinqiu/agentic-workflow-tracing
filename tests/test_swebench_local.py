import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('swebench_local', ROOT / 'scripts/swebench_local.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class SwebenchLocalTests(unittest.TestCase):
    def test_routes_all_models_to_vllm(self):
        for url in ['http://host:8000', 'http://host:8000/v1/', 'http://host:8000/v1/chat/completions']:
            cfg = m.endpoint_config({'VLLM_URL': url, 'VLLM_SERVED_MODEL': 'test-model'})
            self.assertEqual(cfg['Providers'][0]['api_base_url'], 'http://host:8000/v1/chat/completions')
            self.assertEqual(set(cfg['Router'].values()), {'vllm,test-model'})
        env = m.agent_env({'VLLM_SERVED_MODEL': 'test-model', 'ANTHROPIC_API_KEY': 'unwanted',
                           'ANTHROPIC_DEFAULT_HAIKU_MODEL': 'hosted-model', 'CLAUDECODE': '1'})
        self.assertNotIn('ANTHROPIC_API_KEY', env)
        self.assertNotIn('CLAUDECODE', env)
        self.assertEqual(env['ANTHROPIC_DEFAULT_HAIKU_MODEL'], 'vllm,test-model')

    def test_dirty_workspace_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / 'test'
            subprocess.run(['git', 'init', '-q', str(repo)], check=True)
            (repo / 'source').write_text('original')
            subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True)
            subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Test', '-c',
                            'user.email=test@example.invalid', 'commit', '-qm', 'base'], check=True)
            task = {'alias': 'test', 'base_commit': m.git(repo, 'rev-parse', 'HEAD')}
            self.assertEqual(m.prepare(task, root), repo)
            (repo / 'source').write_text('user changes')
            with self.assertRaisesRegex(ValueError, 'contains changes'):
                m.prepare(task, root)
            self.assertEqual((repo / 'source').read_text(), 'user changes')

    def test_task_metadata_and_prompts(self):
        tasks = json.loads((m.TASKS / 'manifest.json').read_text())
        self.assertEqual([t['alias'] for t in tasks], ['django', 'sphinx', 'nodebb'])
        for task in tasks:
            ctx = json.loads((m.TASKS / task['alias'] / 'task_context.json').read_text())
            self.assertEqual(ctx['base_commit'], task['base_commit'])
            prompt = (m.TASKS / task['alias'] / 'prompt.txt').read_text()
            self.assertIn('# Issue', prompt)
            self.assertNotIn('diff --git', prompt)


if __name__ == '__main__':
    unittest.main()
