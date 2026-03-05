import unittest
from pathlib import Path
from ast_analyzer import ASTAnalyzer
from ast_transformer import ASTTransformer

class TestV4Optimizations(unittest.TestCase):
    def setUp(self):
        self.analyzer = ASTAnalyzer()
        self.transformer = ASTTransformer()
        self.test_dir = Path("tests/temp_v4")
        self.test_dir.mkdir(exist_ok=True)

    def tearDown(self):
        for f in self.test_dir.glob("*"):
            f.unlink()
        if self.test_dir.exists():
            self.test_dir.rmdir()

    def test_unused_loop_value(self):
        script = """
        for k, v in pairs(t) do
            print(k)
        end
        """
        path = self.test_dir / "unused_loop.lua"
        path.write_text(script)
        findings = self.analyzer.analyze_file(path)
        self.assertTrue(any(f.pattern_name == 'unused_loop_variable' and f.details.get('var_name') == 'v' for f in findings))

    def test_math_atan2_to_atan(self):
        script = "local a = math.atan2(y, 1)"
        path = self.test_dir / "atan2.lua"
        path.write_text(script)

        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn("math.atan(y)", new_content)

    def test_math_mod_to_percent(self):
        script = "local m = math.mod(x, y)"
        path = self.test_dir / "mod.lua"
        path.write_text(script)

        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn("x % y", new_content)

    def test_math_log_base_e(self):
        script = "local l = math.log(x, math.exp(1))"
        path = self.test_dir / "log.lua"
        path.write_text(script)

        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn("math.log(x)", new_content)

    def test_constant_folding_pi(self):
        script = "local x = 2 * math.pi"
        path = self.test_dir / "pi.lua"
        path.write_text(script)

        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        # 2 * 3.141592653589793 = 6.283185307179586
        self.assertIn("6.283185", new_content)

if __name__ == "__main__":
    unittest.main()
