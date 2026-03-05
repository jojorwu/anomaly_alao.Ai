import unittest
from pathlib import Path
from ast_analyzer import ASTAnalyzer
from ast_transformer import ASTTransformer

class TestV5Optimizations(unittest.TestCase):
    def setUp(self):
        self.analyzer = ASTAnalyzer()
        self.transformer = ASTTransformer()
        self.test_dir = Path("tests/temp_v5")
        self.test_dir.mkdir(exist_ok=True)

    def tearDown(self):
        for f in self.test_dir.glob("*"):
            f.unlink()
        if self.test_dir.exists():
            self.test_dir.rmdir()

    def test_math_deg_rad(self):
        script = "local d = math.deg(x)\nlocal r = math.rad(y)"
        path = self.test_dir / "degrad.lua"
        path.write_text(script)

        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn("x * 57.29577951", new_content)
        self.assertIn("y * 0.0174532925", new_content)

    def test_math_random_0_1(self):
        script = "local r = math.random(0, 1)"
        path = self.test_dir / "rand01.lua"
        path.write_text(script)

        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn("math.random()", new_content)

    def test_constant_folding_deg_rad(self):
        script = "local d = math.deg(1)\nlocal r = math.rad(180)"
        path = self.test_dir / "fold_degrad.lua"
        path.write_text(script)

        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn("local d = 57.29577951", new_content)
        self.assertIn("local r = 3.141592654", new_content)

    def test_string_rep_2(self):
        script = 'local s = string.rep("abc", 2)'
        path = self.test_dir / "rep2.lua"
        path.write_text(script)

        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn('"abc" .. "abc"', new_content)

if __name__ == "__main__":
    unittest.main()
