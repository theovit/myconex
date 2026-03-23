"""
MYCONEX integration adapters for external AI frameworks.

Packages:
  flash-moe      github.com/danveloper/flash-moe      — C/Metal MoE inference engine
  hermes-agent   github.com/NousResearch/hermes-agent  — multi-provider agent framework

The primary integration surface is `moe_hermes_integration.HermesMoEAgent`,
which replaces the default Ollama/llama3.1:8b backend.
"""
