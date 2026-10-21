"""Validate both BCC lost-callback signatures without requiring a Linux kernel."""
import ast
from pathlib import Path
import unittest


class LostCallbackTests(unittest.TestCase):
    def test_old_and_new_bcc_signatures(self):
        path = Path(__file__).resolve().parents[1]/'src/agent_io_tracing/tracing/bcc_tracer.py'
        tree = ast.parse(path.read_text())
        callback = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'handle_lost')
        namespace = {'lost_events':[0]}
        exec(compile(ast.Module(body=[callback],type_ignores=[]),str(path),'exec'),namespace)
        namespace['handle_lost'](7)
        namespace['handle_lost'](2,3)
        self.assertEqual(namespace['lost_events'],[10])


if __name__ == '__main__':
    unittest.main()
