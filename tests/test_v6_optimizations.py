import unittest
from pathlib import Path
from ast_analyzer import ASTAnalyzer
from ast_transformer import ASTTransformer

class TestV6Optimizations(unittest.TestCase):
    def setUp(self):
        self.analyzer = ASTAnalyzer()
        self.transformer = ASTTransformer()
        self.test_dir = Path("tests/temp_v6")
        self.test_dir.mkdir(exist_ok=True)

    def tearDown(self):
        for f in self.test_dir.glob("*"):
            f.unlink()
        if self.test_dir.exists():
            self.test_dir.rmdir()

    def test_infer_type_expanded(self):
        script = "local x = math.sin(1)\nlocal y = tonumber(x)"
        path = self.test_dir / "infer.lua"
        path.write_text(script)

        modified, new_content, _ = self.transformer.transform_file(path, backup=False, fix_yellow=True)
        self.assertTrue(modified)
        self.assertIn("local y = x", new_content)

    def test_constant_folding_uminus(self):
        script = "local x = -1 + 2\nlocal y = -math.pi"
        path = self.test_dir / "fold_uminus.lua"
        path.write_text(script)

        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn("local x = 1", new_content)
        # -math.pi -> -3.141592654
        self.assertIn("-3.141592654", new_content)

    def test_constant_folding_math(self):
        script = """
local a = math.abs(-10)
local b = math.floor(1.5)
local c = math.ceil(1.1)
local d = math.sqrt(16)
local e = math.min(1, 2, 3)
local f = math.max(1, 2, 3)
"""
        path = self.test_dir / "fold_math.lua"
        path.write_text(script)

        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn("local a = 10", new_content)
        self.assertIn("local b = 1", new_content)
        self.assertIn("local c = 2", new_content)
        self.assertIn("local d = 4", new_content)
        self.assertIn("local e = 1", new_content)
        self.assertIn("local f = 3", new_content)

    def test_math_pow_neg(self):
        script = """
local a = math.pow(x, -1)
local b = math.pow(y, -2)
local c = math.pow(z, -0.5)
"""
        path = self.test_dir / "pow_neg.lua"
        path.write_text(script)

        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn("1/x", new_content)
        self.assertIn("1/(y*y)", new_content)
        self.assertIn("1/math.sqrt(z)", new_content)

    def test_math_fmod(self):
        script = "local r = math.fmod(x, 10)"
        path = self.test_dir / "fmod.lua"
        path.write_text(script)

        # fmod is YELLOW
        modified, new_content, _ = self.transformer.transform_file(path, backup=False, fix_yellow=True)
        self.assertTrue(modified)
        self.assertIn("x % 10", new_content)

    def test_table_insert_getn(self):
        script = "table.insert(t, table.getn(t) + 1, v)"
        path = self.test_dir / "insert_getn.lua"
        path.write_text(script)

        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn("t[#t+1] = v", new_content)

if __name__ == "__main__":
    unittest.main()
