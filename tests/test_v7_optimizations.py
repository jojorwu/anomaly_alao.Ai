import unittest
from pathlib import Path
from ast_analyzer import ASTAnalyzer
from ast_transformer import ASTTransformer

class TestV7Optimizations(unittest.TestCase):
    def setUp(self):
        self.analyzer = ASTAnalyzer()
        self.transformer = ASTTransformer()
        self.test_dir = Path("tests/temp_v7")
        self.test_dir.mkdir(exist_ok=True)

    def tearDown(self):
        for f in self.test_dir.glob("*"):
            f.unlink()
        if self.test_dir.exists():
            self.test_dir.rmdir()

    def test_bit_folding(self):
        script = """
local a = bit.band(0xF0, 0x0F)
local b = bit.bor(1, 2, 4)
local c = bit.bxor(7, 3)
local d = bit.bnot(0)
local e = bit.lshift(1, 4)
local f = bit.rshift(16, 2)
local g = bit.arshift(-16, 2)
local h = bit.rol(1, 1)
local i = bit.ror(2, 1)
"""
        path = self.test_dir / "bit.lua"
        path.write_text(script)
        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn("local a = 0", new_content)
        self.assertIn("local b = 7", new_content)
        self.assertIn("local c = 4", new_content)
        self.assertIn("local d = -1", new_content)
        self.assertIn("local e = 16", new_content)
        self.assertIn("local f = 4", new_content)
        self.assertIn("local g = -4", new_content)
        self.assertIn("local h = 2", new_content)
        self.assertIn("local i = 1", new_content)

    def test_string_conversion_folding(self):
        script = """
local a = string.len("test")
local b = string.byte("ABC", 2)
local c = string.char(65, 66, 67)
local d = tonumber("123.45")
local e = tostring(100)
local f = tostring("already")
"""
        path = self.test_dir / "string.lua"
        path.write_text(script)
        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn("local a = 4", new_content)
        self.assertIn("local b = 66", new_content)
        self.assertIn('local c = "ABC"', new_content)
        self.assertIn("local d = 123.45", new_content)
        self.assertIn('local e = "100"', new_content)
        self.assertIn('local f = "already"', new_content)

    def test_math_pow_identity(self):
        script = """
local a = math.pow(x, 1)
local b = math.pow(y, 0)
local c = x ^ 1
local d = x ^ 0
"""
        path = self.test_dir / "pow.lua"
        path.write_text(script)
        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn("local a = x", new_content)
        self.assertIn("local b = 1", new_content)
        self.assertIn("local c = x", new_content)
        self.assertIn("local d = 1", new_content)

    def test_algebraic_simplification(self):
        script = """
local x = math.random()
local a = x + 0
local b = 0 + x
local c = x - 0
local d = x * 1
local e = 1 * x
local f = x / 1
local g = string.upper("test")
local h = string.lower("TEST")
"""
        path = self.test_dir / "algebra.lua"
        path.write_text(script)
        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn("local a = x", new_content)
        self.assertIn("local b = x", new_content)
        self.assertIn("local c = x", new_content)
        self.assertIn("local d = x", new_content)
        self.assertIn("local e = x", new_content)
        self.assertIn("local f = x", new_content)
        self.assertIn('local g = "TEST"', new_content)
        self.assertIn('local h = "test"', new_content)

    def test_starts_with(self):
        script = """
if string.find(s, "prefix") == 1 then end
if 1 == string.find(s, "p") then end
"""
        path = self.test_dir / "starts.lua"
        path.write_text(script)
        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn('if string.sub(s, 1, 6) == "prefix" then end', new_content)
        self.assertIn('if string.byte(s) == 112 then end', new_content)

if __name__ == "__main__":
    unittest.main()
