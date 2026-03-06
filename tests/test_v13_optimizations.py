import unittest
from pathlib import Path
from ast_analyzer import ASTAnalyzer
from ast_transformer import ASTTransformer

class TestV13Optimizations(unittest.TestCase):
    def setUp(self):
        self.analyzer = ASTAnalyzer()
        self.transformer = ASTTransformer()
        self.test_dir = Path("tests/temp_v13")
        self.test_dir.mkdir(exist_ok=True)

    def tearDown(self):
        for f in self.test_dir.glob("*"):
            f.unlink()
        if self.test_dir.exists():
            self.test_dir.rmdir()

    def transform(self, script):
        path = self.test_dir / "test.lua"
        path.write_text(script)
        modified, new_content, _ = self.transformer.transform_file(path, backup=False, fix_yellow=True)
        return new_content if modified else script

    def test_nested_index_simple_expr(self):
        # x^2 -> x*x should now work for nested index
        script = "local a = db.actor.health ^ 2"
        expected = "local a = db.actor.health*db.actor.health"
        self.assertEqual(self.transform(script), expected)

    def test_enhanced_constant_folding(self):
        # math.sin
        self.assertIn("local a = 0", self.transform("local a = math.sin(0)"))
        # math.log
        self.assertIn("local a = 0", self.transform("local a = math.log(1)"))
        # string.rep
        self.assertIn('local a = "abcabc"', self.transform('local a = string.rep("abc", 2)'))

    def test_algebraic_0_minus_x(self):
        script = "local a = 0 - x"
        expected = "local a = - x"
        self.assertEqual(self.transform(script), expected)

    def test_math_abs_comparisons(self):
        # math.abs(x) > 0 -> x ~= 0
        self.assertIn("if x ~= 0 then end", self.transform("if math.abs(x) > 0 then end"))
        # math.abs(x) <= 0 -> x == 0
        self.assertIn("if x == 0 then end", self.transform("if math.abs(x) <= 0 then end"))
        # math.abs(x) ~= 0 -> x ~= 0
        self.assertIn("if x ~= 0 then end", self.transform("if math.abs(x) ~= 0 then end"))
        # math.abs(x) == 0 -> x == 0
        self.assertIn("if x == 0 then end", self.transform("if math.abs(x) == 0 then end"))

    def test_expo_to_sqrt(self):
        script = "local a = x ^ 0.5"
        expected = "local a = math.sqrt(x)"
        self.assertEqual(self.transform(script), expected)

    def test_sqrt_abs_pow_identity(self):
        # (math.sqrt(x))^2 -> x
        self.assertEqual(self.transform("local a = math.sqrt(x) ^ 2"), "local a = x")
        self.assertEqual(self.transform("local a = math.pow(math.sqrt(x), 2)"), "local a = x")

        # (math.abs(x))^2 -> x*x
        self.assertEqual(self.transform("local a = math.abs(x) ^ 2"), "local a = x*x")
        self.assertEqual(self.transform("local a = math.pow(math.abs(x), 2)"), "local a = x*x")

if __name__ == "__main__":
    unittest.main()
