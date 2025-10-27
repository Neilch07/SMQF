"""Fallback stub for numexpr to avoid binary incompatibilities with numpy 2.x.

Pandas treats numexpr as an optional dependency. Raising ImportError here lets
pandas gracefully disable numexpr acceleration instead of crashing when the
compiled wheel mismatches the local numpy version.
"""
raise ImportError("numexpr is disabled due to local binary incompatibility")
