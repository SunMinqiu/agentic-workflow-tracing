import asyncio
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from aiohttp import web, ClientSession

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('proxy', ROOT/'scripts/swebench_proxy.py')
proxy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proxy)


class ProxyTests(unittest.IsolatedAsyncioTestCase):
    async def test_serial_stream_capture(self):
        active = 0
        maximum = 0
        seen = []
        async def upstream(request):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            seen.append(await request.json())
            await asyncio.sleep(0.02)
            active -= 1
            events = [{'choices':[{'delta':{'reasoning':'thinking'}}]},
                      {'choices':[{'delta':{'content':'answer'}}]},
                      {'choices':[], 'usage':{'prompt_tokens':8,'completion_tokens':2},
                       'metrics':{'time_to_first_token_ms':10}}]
            return web.Response(text=''.join('data: '+json.dumps(e)+'\n\n' for e in events)+'data: [DONE]\n\n', content_type='text/event-stream')
        app = web.Application()
        app.router.add_post('/v1/chat/completions', upstream)
        up_runner = web.AppRunner(app)
        await up_runner.setup()
        up_site = web.TCPSite(up_runner, '127.0.0.1', 0)
        await up_site.start()
        port = up_site._server.sockets[0].getsockname()[1]
        try:
            with tempfile.TemporaryDirectory() as directory:
                local = web.Application()
                local['args'] = SimpleNamespace(upstream=f'http://127.0.0.1:{port}',output=directory)
                local['serial_lock'] = asyncio.Lock()
                local['cache_salt'] = 'one-run'
                local.router.add_post('/v1/chat/completions', proxy.forward)
                runner = web.AppRunner(local)
                await runner.setup()
                site = web.TCPSite(runner, '127.0.0.1', 0)
                await site.start()
                local_port = site._server.sockets[0].getsockname()[1]
                try:
                    async with ClientSession() as session:
                        async def request():
                            async with session.post(f'http://127.0.0.1:{local_port}/v1/chat/completions',json={'stream':True,'messages':[]}) as response:
                                self.assertIn('answer',await response.text())
                        await asyncio.gather(request(),request())
                        async with session.post(f'http://127.0.0.1:{local_port}/v1/chat/completions', json={
                            'stream': True, 'messages': [{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/png;base64,AA=='}}]}]
                        }) as response:
                            self.assertEqual(response.status,415)
                    rows = [json.loads(l) for l in (Path(directory)/'requests.jsonl').read_text().splitlines()]
                    self.assertEqual(maximum,1)
                    self.assertEqual(len(rows),2)
                    self.assertTrue(all(r['first_token_ms'] is not None for r in rows))
                    self.assertTrue(all(r['usage']['completion_tokens']==2 for r in rows))
                    self.assertTrue(all(r['server_metrics']['time_to_first_token_ms']==10 for r in rows))
                    self.assertEqual([r['cache_salt'] for r in seen],['one-run','one-run'])
                finally:
                    await runner.cleanup()
        finally:
            await up_runner.cleanup()


if __name__ == '__main__':
    unittest.main()
