import pytest
from datavac.util.lazydict import CategorizedLazyDict
from datavac.util.lazydict import _default_splitter, _default_joiner
from datavac.util.lazydict import FunctionLazyDict

def test_default_splitter_and_joiner():
    key = "foo bar"
    category, subkey = _default_splitter(key)
    assert category == "foo"
    assert subkey == "bar"
    joined = _default_joiner(category, subkey)
    assert joined == key

def test_getitem_and_lazy_loading():
    calls = []
    def getter_a():
        calls.append('a')
        return {'x': 1, 'y': 2}
    def getter_b():
        calls.append('b')
        return {'z': 3}
    d = CategorizedLazyDict({'a': getter_a, 'b': getter_b})
    # Not loaded yet
    assert calls == []
    assert d['a x'] == 1
    assert calls == ['a']
    assert d['a y'] == 2
    # Should not call getter_a again
    assert calls == ['a']
    assert d['b z'] == 3
    assert calls == ['a', 'b']

def test_iter_and_len():
    def getter_a():
        return {'x': 1, 'y': 2}
    def getter_b():
        return {'z': 3}
    d = CategorizedLazyDict({'a': getter_a, 'b': getter_b})
    keys = set(d)
    assert keys == {'a x', 'a y', 'b z'}
    assert len(d) == 3

def test_keyerror_on_missing_category():
    d = CategorizedLazyDict({'a': lambda: {'x': 1}})
    with pytest.raises(KeyError):
        _ = d['b y']

def test_keyerror_on_missing_subkey():
    d = CategorizedLazyDict({'a': lambda: {'x': 1}})
    with pytest.raises(KeyError):
        _ = d['a y']

def test_custom_splitter_and_joiner():
    def splitter(key):
        return key.split(':')
    def joiner(category, subkey):
        return f"{category}:{subkey}"
    d = CategorizedLazyDict(
        {'foo': lambda: {'bar': 42}},
        category_splitter=splitter,
        category_joiner=joiner
    )
    assert d['foo:bar'] == 42
    assert set(d) == {'foo:bar'}
    
def test_contains_existing_key():
    d = CategorizedLazyDict({'a': lambda: {'x': 1, 'y': 2}})
    assert 'a x' in d
    assert 'a y' in d

def test_contains_nonexistent_category():
    d = CategorizedLazyDict({'a': lambda: {'x': 1}})
    assert 'b x' not in d

def test_contains_nonexistent_subkey():
    d = CategorizedLazyDict({'a': lambda: {'x': 1}})
    assert 'a y' not in d

def test_contains_triggers_lazy_loading_once():
    calls = []
    def getter():
        calls.append('called')
        return {'x': 1}
    d = CategorizedLazyDict({'a': getter})
    assert 'a x' in d
    assert calls == ['called']
    # Should not call again
    assert 'a x' in d
    assert calls == ['called']

def test_contains_with_custom_splitter_and_joiner():
    def splitter(key):
        return key.split(':')
    def joiner(category, subkey):
        return f"{category}:{subkey}"
    d = CategorizedLazyDict(
        {'foo': lambda: {'bar': 42}},
        category_splitter=splitter,
        category_joiner=joiner
    )
    assert 'foo:bar' in d
    assert 'foo:baz' not in d
    
def test_functionlazydict_getitem_and_lazy_loading():
    calls = []
    def getter(key):
        calls.append(key)
        return key.upper()
    def keylister():
        return ['a', 'b']
    d = FunctionLazyDict(getter, keylister)
    # Not loaded yet
    assert calls == []
    assert d['a'] == 'A'
    assert calls == ['a']
    assert d['b'] == 'B'
    assert calls == ['a', 'b']
    # Should not call getter again for already loaded keys
    assert d['a'] == 'A'
    assert calls == ['a', 'b']

def test_functionlazydict_iter_and_len():
    def getter(key):
        return key.upper()
    def keylister():
        return ['x', 'y', 'z']
    d = FunctionLazyDict(getter, keylister)
    keys = set(d)
    assert keys == {'x', 'y', 'z'}
    assert len(d) == 3

def test_functionlazydict_contains():
    def getter(key):
        return key
    def keylister():
        return ['foo', 'bar']
    d = FunctionLazyDict(getter, keylister)
    assert 'foo' in d
    assert 'bar' in d
    assert 'baz' not in d

def test_functionlazydict_keyerror_on_missing_key():
    def getter(key):
        if key == 'a':
            return 1
        raise KeyError(key)
    def keylister():
        return ['a']
    d = FunctionLazyDict(getter, keylister)
    with pytest.raises(KeyError):
        _ = d['b']

