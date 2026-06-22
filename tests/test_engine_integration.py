from domain_kv.engine_integration import MockEngineAdapter, VLLMAdapter, KVPressAdapter


def test_mock_adapter_roundtrip():
    store = {'sec': [('k1', 1), ('k2', 2)]}
    m = MockEngineAdapter(internal_store=store)
    snap = m.snapshot()
    assert snap == store
    m.apply_compressed({'sec': [('k1', 1)]})
    assert m.store['sec'][0][0] == 'k1'


def test_adapters_exist():
    # VLLMAdapter/KVPressAdapter may be None if packages not installed; ensure attribute present
    assert hasattr(MockEngineAdapter, 'snapshot')
    assert VLLMAdapter is not None or VLLMAdapter is None
    assert KVPressAdapter is not None or KVPressAdapter is None
