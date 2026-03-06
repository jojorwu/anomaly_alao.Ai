import unittest
from pathlib import Path
from ast_analyzer import ASTAnalyzer
from ast_transformer import ASTTransformer

class TestV8Optimizations(unittest.TestCase):
    def setUp(self):
        self.analyzer = ASTAnalyzer()
        self.transformer = ASTTransformer()
        self.test_dir = Path("tests/temp_v8")
        self.test_dir.mkdir(exist_ok=True)

    def tearDown(self):
        for f in self.test_dir.glob("*"):
            f.unlink()
        if self.test_dir.exists():
            self.test_dir.rmdir()

    def test_logical_identity(self):
        script = """
if x and true then end
if false and y then end
if a or false then end
if true or b then end
local bool1 = true
if bool1 and true then end
local bool2 = not not bool1
"""
        path = self.test_dir / "logic.lua"
        path.write_text(script)
        # Note: fix_yellow not needed since logical_identity is GREEN
        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn("if x then end", new_content)
        self.assertIn("if false then end", new_content)
        self.assertIn("if a then end", new_content)
        self.assertIn("if true then end", new_content)
        self.assertIn("local bool2 = bool1", new_content)

    def test_nested_redundant_calls(self):
        script = """
local a = math.abs(math.abs(x))
local b = math.floor(math.floor(y))
local c = tostring(tostring(z))
local d = tonumber(tonumber(w))
"""
        path = self.test_dir / "nested.lua"
        path.write_text(script)
        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn("local a = math.abs(x)", new_content)
        self.assertIn("local b = math.floor(y)", new_content)
        self.assertIn("local c = tostring(z)", new_content)
        self.assertIn("local d = tonumber(w)", new_content)

    def test_table_literal_simplification(self):
        script = """
local t = { [1] = "a", [2] = "b", [3] = "c" }
local t2 = { "a", [2] = "b" }
"""
        path = self.test_dir / "table.lua"
        path.write_text(script)
        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn('local t = { "a", "b", "c" }', new_content)
        self.assertIn('local t2 = { "a", "b" }', new_content)

    def test_unused_ipairs_value(self):
        script = """
for i, v in ipairs(t) do
    print(i)
end
"""
        path = self.test_dir / "ipairs.lua"
        path.write_text(script)
        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn("for i = 1, #t do", new_content)
        self.assertNotIn("local v = t[i]", new_content)

    def test_bit_identity(self):
        script = """
local a = bit.band(x, 0)
local b = bit.bor(y, 0)
local c = bit.bxor(z, 0)
local d = bit.lshift(w, 0)
local e = bit.band(x, 0xFFFFFFFF)
"""
        path = self.test_dir / "bit.lua"
        path.write_text(script)
        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn("local a = 0", new_content)
        self.assertIn("local b = y", new_content)
        self.assertIn("local c = z", new_content)
        self.assertIn("local d = w", new_content)
        self.assertIn("local e = x", new_content)

if __name__ == "__main__":
    unittest.main()
