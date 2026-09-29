"""Offline RAG regression tests. No credentials, API calls or artifact writes."""
import ast
import json
import os
from pathlib import Path
import time
import traceback
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import faiss
import numpy as np


def load_app_logic():
    source = Path('app.py').read_text(encoding='utf-8')
    env = dict(os=os, json=json, np=np, faiss=faiss, time=time, traceback=traceback,
               client=Mock(), session_histories={}, HISTORY_TTL=1800,
               VECTOR_PATH='data/vector_data.npy', INDEX_PATH='data/index.faiss',
               RESERVE_VECTOR_PATH='data/reserve_vector_data.npy',
               RESERVE_INDEX_PATH='data/reserve_index.faiss',
               EMBED_MODEL='text-embedding-3-small')
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    for node in functions:
        node.decorator_list = []
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'app.py', 'exec'), env)
    # Execute the actual data/index loading code, excluding external service startup.
    begin = source.index('# 通常用')
    end = source.index('SPREADSHEET_ID =')
    exec(compile(source[begin:end], 'app.py', 'exec'), env)
    return env


class RagTests(unittest.TestCase):
    def setUp(self):
        self.e = load_app_logic()

    def request(self, question, reserve, indices=None):
        e = self.e
        e.update(request=SimpleNamespace(
                     is_json=True,
                     get_json=lambda **_kwargs: {'question': question, 'session_id': 'test'}),
                 jsonify=lambda x: x, GREETING_PATTERNS=[], base_prompt='SYSTEM',
                 expand_query=Mock(return_value='normal expanded'),
                 expand_reserve_query=Mock(return_value='reserve expanded'),
                 get_embedding=Mock(return_value=np.zeros(1536, dtype='float32')),
                 pf_matcher=Mock(), log_chat_history=Mock())
        e['pf_matcher'].format_match_info.return_value = ''
        nfaq = len(e['reserve_faq_questions' if reserve else 'faq_questions'])
        ids = indices if indices is not None else [0, nfaq, -1, -1, -1, -1, -1]
        e['index'] = Mock()
        e['reserve_index'] = Mock()
        for idx in (e['index'], e['reserve_index']):
            idx.search.return_value = (np.zeros((1, 7)), np.array([ids]))
        e['client'].chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='回答'))])
        result = e['chat']()
        return result

    def assert_route(self, question, reserve):
        result = self.request(question, reserve)
        e = self.e
        self.assertEqual(result['response'], '回答')
        e['reserve_index' if reserve else 'index'].search.assert_called_once()
        e['index' if reserve else 'reserve_index'].search.assert_not_called()
        e['expand_reserve_query' if reserve else 'expand_query'].assert_called_once_with(question, [])
        call = e['client'].chat.completions.create.call_args.kwargs
        self.assertEqual(call['model'], 'gpt-4o')
        prompt = call['messages'][1]['content']
        prefix = 'reserve_' if reserve else ''
        self.assertIn('Q: ' + e[prefix+'faq_questions'][0], prompt)
        self.assertIn('A: ' + e[prefix+'faq_answers'][0], prompt)
        self.assertIn('【参考知識】' + e[prefix+'knowledge_contents'][0], prompt)
        self.assertIn('ユーザーの質問: ' + question, prompt)
        self.assertNotIn('expanded', prompt)
        if reserve:
            e['pf_matcher'].match.assert_not_called()
        else:
            e['pf_matcher'].match.assert_called_once_with(question, [])
        e['log_chat_history'].assert_called_once()

    def test_reservation(self):
        self.assert_route('製造予約はどこから行えますか？', True)

    def test_login(self):
        self.assert_route('予約システムにログインできません', True)

    def test_product(self):
        self.assert_route('ドリップバッグの最小ロットはいくつですか？', False)

    def test_negative_indices(self):
        self.request('予約について', True, [-1]*7)
        prompt = self.e['client'].chat.completions.create.call_args.kwargs['messages'][1]['content']
        self.assertNotIn('【参考知識】', prompt)
        self.assertNotIn('Q:', prompt)

    def test_empty_results(self):
        self.e['metadata_note'] = ''
        result = self.request('予約について', True, [-1]*7)
        self.assertIn('当社は', result['response'])
        self.e['client'].chat.completions.create.assert_not_called()

    def test_blank_question(self):
        result = self.request('   ', False)
        self.assertEqual(result[1], 400)
        self.e['get_embedding'].assert_not_called()

    def test_history_ttl_and_limit(self):
        e = self.e
        for i in range(12):
            e['add_to_session_history']('s', 'user', str(i))
        self.assertEqual(len(e['get_session_history']('s')), 10)
        e['session_histories']['s']['last_active'] = time.time() - 1801
        self.assertEqual(e['get_session_history']('s'), [])

    def test_embedding_contract(self):
        e = self.e
        e['client'].embeddings.create.return_value = SimpleNamespace(data=[SimpleNamespace(embedding=[1.0]*1536)])
        vector = e['get_embedding']('質問')
        self.assertEqual(vector.dtype, np.float32)
        self.assertEqual(vector.shape, (1536,))
        e['client'].embeddings.create.assert_called_once_with(model='text-embedding-3-small', input=['質問'])
        with self.assertRaises(ValueError):
            e['get_embedding'](' ')
        e['client'].embeddings.create.side_effect = RuntimeError('mock API failure')
        with self.assertRaises(RuntimeError):
            e['get_embedding']('質問')

    def test_normal_artifacts(self):
        self.check_artifacts('')

    def test_reserve_artifacts(self):
        self.check_artifacts('reserve_')

    def check_artifacts(self, prefix):
        e = self.e
        idx, vectors = e[prefix+'index'], e[prefix+'vector_data']
        corpus, flags = e[prefix+'search_corpus'], e[prefix+'source_flags']
        self.assertEqual(idx.ntotal, len(corpus))
        self.assertEqual(vectors.shape, (len(corpus), 1536))
        np.testing.assert_array_equal(idx.reconstruct_n(0, idx.ntotal), vectors)
        nf = len(e[prefix+'faq_questions'])
        self.assertEqual(flags[:nf], ['faq']*nf)
        for i, text in enumerate(e[prefix+'knowledge_contents']):
            self.assertEqual(corpus[nf+i], text)
            self.assertEqual(flags[nf+i], 'knowledge')
        for i, question in enumerate(e[prefix+'faq_questions']):
            expected = question + ' ' + e[prefix+'faq_answers'][i] if prefix else question
            self.assertEqual(corpus[i], expected)

    def test_index_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            self.e['validate_search_index'](np.zeros((1, 2)), SimpleNamespace(ntotal=2, d=2), ['x'], 'test')

    def test_embedding_defined_before_startup_generation(self):
        tree = ast.parse(Path('app.py').read_text(encoding='utf-8'))
        definition = next(n.lineno for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_embedding')
        for node in tree.body:
            if isinstance(node, ast.If):
                for call in ast.walk(node):
                    if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'get_embedding':
                        self.assertLess(definition, call.lineno)

    def test_real_faiss_results_reach_prompt(self):
        for reserve in [False, True]:
            self.e = load_app_logic()
            e = self.e
            prefix = 'reserve_' if reserve else ''
            real_index = e[prefix+'index']
            vector = e[prefix+'vector_data'][len(e[prefix+'faq_questions'])]
            self.request('予約について' if reserve else '製品について', reserve)
            e[prefix+'index'] = real_index
            e['get_embedding'].return_value = vector
            e['chat']()
            prompt = e['client'].chat.completions.create.call_args.kwargs['messages'][1]['content']
            self.assertIn(e[prefix+'knowledge_contents'][0], prompt)

    def test_faiss_small_and_empty(self):
        idx = faiss.IndexFlatL2(2)
        self.assertTrue((idx.search(np.zeros((1, 2), dtype='float32'), 7)[1] == -1).all())
        idx.add(np.zeros((1, 2), dtype='float32'))
        self.assertEqual(idx.search(np.zeros((1, 2), dtype='float32'), 7)[1].tolist(), [[0, -1, -1, -1, -1, -1, -1]])

    def test_expander_fallbacks(self):
        for filename, name in [('query_expander.py', 'expand_query'), ('expand_reserve_query.py', 'expand_reserve_query')]:
            tree = ast.parse(Path(filename).read_text(encoding='utf-8'))
            api = Mock()
            env = {'openai': api}
            exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[]), filename, 'exec'), env)
            for content in ['', '  ', None]:
                api.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
                self.assertEqual(env[name]('原文', [{'role': 'user', 'content': '前の質問'}]), '原文')
            api.chat.completions.create.side_effect = RuntimeError('mock failure')
            self.assertEqual(env[name]('原文', [{'role': 'user', 'content': '前の質問'}]), '原文')


if __name__ == '__main__':
    unittest.main()
