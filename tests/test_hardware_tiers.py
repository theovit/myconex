import ast, pathlib

def _load_tier_defs():
    src = pathlib.Path("core/classifier/hardware.py").read_text()
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
