import json
import os
import queue
import subprocess
import sys
import threading

import pytest

pytest.importorskip("fastmcp")


def test_stdio_server_serves_tools_with_clean_stdout():
    env = dict(os.environ, DATAVACUUM_CONTEXT='builtin:demo2')
    p = subprocess.Popen([sys.executable, '-m', 'datavac.llm.mcp_server'], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True, encoding='utf-8')
    stdout_lines: queue.Queue = queue.Queue()
    threading.Thread(target=lambda: [stdout_lines.put(l) for l in p.stdout], daemon=True).start() # type: ignore
    threading.Thread(target=lambda: [None for _ in p.stderr], daemon=True).start() # type: ignore

    def send(msg): p.stdin.write(json.dumps(msg) + '\n'); p.stdin.flush() # type: ignore
    def result_for(id_):
        while True:
            msg = json.loads(stdout_lines.get(timeout=120)) # a non-JSON line on stdout fails here
            if msg.get('id') == id_: return msg['result']

    try:
        send({'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {
            'protocolVersion': '2025-06-18', 'capabilities': {}, 'clientInfo': {'name': 'test', 'version': '0'}}})
        result_for(1)
        send({'jsonrpc': '2.0', 'method': 'notifications/initialized'})
        send({'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'})
        assert {t['name'] for t in result_for(2)['tools']} == {'list_mgoas', 'describe_mg', 'describe_an', 'readonly_sql'}

        send({'jsonrpc': '2.0', 'id': 3, 'method': 'tools/call',
              'params': {'name': 'describe_mg', 'arguments': {'mg_name': 'IdVg'}}})
        text = result_for(3)['content'][0]['text']
        assert 'CREATE TABLE' in text
    finally:
        p.stdin.close() # type: ignore
        p.wait(timeout=30)
