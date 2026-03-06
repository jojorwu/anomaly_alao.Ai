import unittest
from pathlib import Path
from ast_analyzer import ASTAnalyzer
from ast_transformer import ASTTransformer

class TestV9Optimizations(unittest.TestCase):
    def setUp(self):
        self.analyzer = ASTAnalyzer()
        self.transformer = ASTTransformer()
        self.test_dir = Path("tests/temp_v9")
        self.test_dir.mkdir(exist_ok=True)

    def tearDown(self):
        for f in self.test_dir.glob("*"):
            f.unlink()
        if self.test_dir.exists():
            self.test_dir.rmdir()

    def test_comparison_inversion(self):
        script = """
if not (a == b) then end
if not (c ~= d) then end
if not (x < y) then end
if not (x > y) then end
if not (x <= y) then end
if not (x >= y) then end
"""
        path = self.test_dir / "inversion.lua"
        path.write_text(script)
        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn("if a ~= b then end", new_content)
        self.assertIn("if c == d then end", new_content)
        self.assertIn("if x >= y then end", new_content)
        self.assertIn("if x <= y then end", new_content)
        self.assertIn("if x > y then end", new_content)
        self.assertIn("if x < y then end", new_content)

    def test_expanded_nested_inverse_calls(self):
        script = """
local a = math.deg(math.rad(x))
local b = math.rad(math.deg(y))
local c = math.abs(math.sqrt(z))
local d = math.abs(math.exp(w))
"""
        path = self.test_dir / "nested_expanded.lua"
        path.write_text(script)
        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn("local a = x", new_content)
        self.assertIn("local b = y", new_content)
        self.assertIn("local c = math.sqrt(z)", new_content)
        self.assertIn("local d = math.exp(w)", new_content)

    def test_math_identities(self):
        script = """
local a = math.min(x, x)
local b = math.max(y, y)
local c = math.min(z, math.huge)
local d = math.max(w, -math.huge)
"""
        path = self.test_dir / "math_id.lua"
        path.write_text(script)
        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn("local a = x", new_content)
        self.assertIn("local b = y", new_content)
        self.assertIn("local c = z", new_content)
        self.assertIn("local d = w", new_content)

    def test_string_identities(self):
        script = """
local s = "test"
local a = s .. ""
local b = "" .. s
local c = string.sub(s, 1)
"""
        path = self.test_dir / "string_id.lua"
        path.write_text(script)
        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn("local a = s", new_content)
        self.assertIn("local b = s", new_content)
        self.assertIn("local c = s", new_content)

    def test_table_remove_expanded(self):
        script = """
table.remove(t, #t)
"""
        path = self.test_dir / "table_rem.lua"
        path.write_text(script)
        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn("t[#t] = nil", new_content)

if __name__ == "__main__":
    unittest.main()
