
import unittest
from pathlib import Path
import tempfile
import os
from ast_analyzer import ASTAnalyzer
from ast_transformer import ASTTransformer

class TestV17Optimizations(unittest.TestCase):
    def setUp(self):
        self.analyzer = ASTAnalyzer()
        self.transformer = ASTTransformer()

    def transform(self, code):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lua', delete=False) as f:
            f.write(code)
            tmp_path = Path(f.name)
        try:
            modified, new_code, _ = self.transformer.transform_file(tmp_path, backup=False)
            return new_code if modified else code
        finally:
            if tmp_path.exists():
                os.remove(tmp_path)

    def test_pairs_to_next_auto(self):
        code = "for k, v in pairs(t) do end"
        new_code = self.transform(code)
        self.assertIn("for k, v in next, t, nil do", new_code)

    def test_string_sub_identity_expanded(self):
        code = "local s1 = string.sub(s, 1, -1); local s2 = string.sub(s, 1, #s)"
        new_code = self.transform(code)
        self.assertIn("local s1 = s", new_code)
        self.assertIn("local s2 = s", new_code)

    def test_abs_sqrt_cmp_negative(self):
        code = "if math.abs(x) < -1 or math.sqrt(x) == -0.5 then end"
        new_code = self.transform(code)
        self.assertIn("if false or false then", new_code)

    def test_trig_inverse(self):
        code = "local a = math.asin(math.sin(x)); local b = math.acos(math.cos(y))"
        new_code = self.transform(code)
        self.assertIn("local a = x", new_code)
        self.assertIn("local b = y", new_code)

    def test_bool_assignment_simplification(self):
        code = "if cond then x = true else x = false end"
        new_code = self.transform(code)
        self.assertIn("x = not not (cond)", new_code)

    def test_bool_assignment_inverted(self):
        code = "if cond then x = false else x = true end"
        new_code = self.transform(code)
        self.assertIn("x = not (cond)", new_code)

if __name__ == '__main__':
    unittest.main()
