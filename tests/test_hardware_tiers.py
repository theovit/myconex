import ast, pathlib

def _load_tier_defs():
    # hardware.py imports psutil and may call subprocess at module level, making a
    # direct import fragile in CI. We parse the source as AST instead to extract
    # the literal TIER_DEFINITIONS dict without executing the module.
    src = (pathlib.Path(__file__).parent.parent / "core/classifier/hardware.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "TIER_DEFINITIONS":
                    return ast.literal_eval(node.value)
    raise ValueError("TIER_DEFINITIONS not found")

def test_tier_models_match_mesh_config():
    defs = _load_tier_defs()
    assert defs["T1"]["ollama_model"] == "llama3.1:70b"
    assert defs["T2"]["ollama_model"] == "qwen3:8b"
    assert defs["T3"]["ollama_model"] == "qwen3:4b"
    assert defs["T4"]["ollama_model"] == "qwen3:0.6b"
