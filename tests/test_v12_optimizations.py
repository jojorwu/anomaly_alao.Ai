import unittest
from pathlib import Path
from ast_analyzer import ASTAnalyzer
from ast_transformer import ASTTransformer

class TestV12Optimizations(unittest.TestCase):
    def setUp(self):
        self.analyzer = ASTAnalyzer()
        self.transformer = ASTTransformer()
        self.test_dir = Path("tests/temp_v12")
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

    def test_math_log10_div(self):
        script = "local a = math.log(x) / math.log(10)"
        expected = "local a = math.log10(x)"
        self.assertEqual(self.transform(script), expected)

    def test_math_sqrt_mult_self(self):
        script = "local a = math.sqrt(x) * math.sqrt(x)"
        expected = "local a = x"
        self.assertEqual(self.transform(script), expected)

    def test_string_empty_checks(self):
        script = "if #s == 0 then end"
        self.assertIn("if s == '' then end", self.transform(script))

        script = "if string.len(s) == 0 then end"
        self.assertIn("if s == '' then end", self.transform(script))

    def test_string_sub_negative_indices_expanded(self):
        script = "local a = string.sub(s, 1, #s)"
        # Note: string_sub_identity might also trigger, but let's check negative index first
        # Actually string_sub_identity is GREEN too.
        # string.sub(s, 1, #s) -> string.sub(s, 1, -1) -> s (via identity)
        res = self.transform(script)
        self.assertTrue("string.sub(s, 1, -1)" in res or "local a = s" in res)

        script = "local a = string.sub(s, 2, #s - 1)"
        self.assertIn("string.sub(s, 2, -2)", self.transform(script))

        script = "local a = string.sub(s, 1, #s - 2)"
        self.assertIn("string.sub(s, 1, -3)", self.transform(script))

        script = "if 0 == #s then end"
        self.assertIn("if s == '' then end", self.transform(script))

    def test_logical_redundancy_nil(self):
        script = "if x == nil or not x then end"
        self.assertIn("if not x then end", self.transform(script))

    def test_logical_redundancy_not_nil(self):
        script = "if x ~= nil and x then end"
        self.assertIn("if x then end", self.transform(script))

    def test_constant_folding_string_char(self):
        script = "local a = string.char(65, 66, 67)"
        self.assertIn('local a = "ABC"', self.transform(script))

    def test_constant_folding_math_exp(self):
        script = "local a = math.exp(0)"
        self.assertIn('local a = 1', self.transform(script))

if __name__ == "__main__":
    unittest.main()
