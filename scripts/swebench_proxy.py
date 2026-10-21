#!/usr/bin/env python3
"""Record OpenAI streaming request timing between CCR and vLLM."""
import argparse
import asyncio
import json
from pathlib import Path
import time
import uuid
from aiohttp import web, ClientSession, ClientTimeout


def has_token(event):
    for choice in event.get('choices', []):
        delta = choice.get('delta', {})
        if delta.get('content') or delta.get('reasoning_content') or delta.get('reasoning'):
            return True
        for call in delta.get('tool_calls', []):
            if call.get('function', {}).get('arguments') or call.get('function', {}).get('name'):
                return True
    return False


async def forward(request):
    async with request.app['serial_lock']:
        return await forward_serial(request)


async def forward_serial(request):
    args = request.app['args']
    rid = uuid.uuid4().hex
    body = await request.read()
    start = time.perf_counter()
    record = {'request_id': rid, 'start_unix': time.time(), 'path': request.path,
              'first_token_ms': None, 'last_token_ms': None, 'token_events': 0}
    payload = json.loads(body)
    for message in payload.get('messages', []):
        content = message.get('content', [])
        if isinstance(content, list) and any(part.get('type') in {
            'image', 'image_url', 'input_image', 'input_audio', 'audio', 'video', 'video_url'
        } for part in content if isinstance(part, dict)):
            raise web.HTTPUnsupportedMediaType(text='This benchmark accepts text-only model inputs')
    payload['cache_salt'] = request.app['cache_salt']
    record['request_kind'] = 'session_title' if b'Generate a concise, sentence-case title' in body else 'agent'
    if payload.get('stream'):
        payload.setdefault('stream_options', {})['include_usage'] = True
        body = json.dumps(payload).encode()
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / (rid + '.request.json')).write_bytes(body)
    pending = b''
    try:
        async with ClientSession(timeout=ClientTimeout(total=None)) as session:
            async with session.request(request.method, args.upstream.rstrip('/') + request.path_qs,
                                       data=body, headers={'Content-Type': 'application/json',
                                           'Authorization': request.headers.get('Authorization', ''),
                                           'X-Request-Id': rid}) as upstream:
                record['status'] = upstream.status
                response = web.StreamResponse(status=upstream.status, headers={
                    'Content-Type': upstream.headers.get('Content-Type', 'application/json')})
                await response.prepare(request)
                with (dest / (rid + '.response')).open('wb') as output:
                    async for chunk in upstream.content.iter_any():
                        now = (time.perf_counter() - start) * 1000
                        output.write(chunk)
                        pending += chunk
                        while b'\n' in pending:
                            line, pending = pending.split(b'\n', 1)
                            if not line.startswith(b'data:'):
                                continue
                            try:
                                event = json.loads(line[5:].strip())
                            except (ValueError, UnicodeError):
                                continue
                            if has_token(event):
                                if record['first_token_ms'] is None:
                                    record['first_token_ms'] = now
                                record['last_token_ms'] = now
                                record['token_events'] += 1
                            if event.get('usage'):
                                record['usage'] = event['usage']
                            if event.get('metrics'):
                                record['server_metrics'] = event['metrics']
                        await response.write(chunk)
                await response.write_eof()
                return response
    except Exception as exc:
        record['error'] = str(exc)
        raise
    finally:
        record['duration_ms'] = (time.perf_counter() - start) * 1000
        with (dest / 'requests.jsonl').open('a') as output:
            output.write(json.dumps(record) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', required=True)
    parser.add_argument('--port', type=int, default=18081)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    app = web.Application(client_max_size=32 * 1024 * 1024)
    app['args'] = args
    app['serial_lock'] = asyncio.Lock()
    app['cache_salt'] = uuid.uuid4().hex
    app.router.add_route('*', '/{path:.*}', forward)
    web.run_app(app, host='127.0.0.1', port=args.port, print=None)
