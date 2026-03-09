
import unittest
from pathlib import Path
import tempfile
import os
from ast_analyzer import ASTAnalyzer
from ast_transformer import ASTTransformer

class TestV16Optimizations(unittest.TestCase):
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

    def test_vector_set_chain(self):
        code = "local v = vector():set(1, 2, 3)"
        new_code = self.transform(code)
        self.assertIn("vector(1, 2, 3)", new_code)

    def test_vector_set_chain_single(self):
        code = "local v = vector():set(v2)"
        new_code = self.transform(code)
        self.assertIn("vector(v2)", new_code)

    def test_math_tan_identity(self):
        code = "local t = math.sin(x) / math.cos(x)"
        new_code = self.transform(code)
        self.assertIn("math.tan(x)", new_code)

    def test_comparison_identity_true(self):
        code = "if x == x then end"
        new_code = self.transform(code)
        self.assertIn("if true then", new_code)

    def test_comparison_identity_false(self):
        code = "if x ~= x then end"
        new_code = self.transform(code)
        self.assertIn("if false then", new_code)

    def test_vector_copy_identity(self):
        code = "v:set(v)"
        new_code = self.transform(code)
        self.assertEqual(new_code.strip(), "v")

    def test_vector_copy_identity_expanded(self):
        code = "v:set(v.x, v.y, v.z)"
        new_code = self.transform(code)
        self.assertEqual(new_code.strip(), "v")

    def test_math_tan_atan_identity(self):
        code = "local x = math.atan(math.tan(angle))"
        new_code = self.transform(code)
        self.assertIn("local x = angle", new_code)

    def test_string_empty_check_gt0(self):
        code = "if #s > 0 then end"
        new_code = self.transform(code)
        self.assertIn("s ~= ''", new_code)

    def test_string_empty_check_ge1(self):
        code = "if #s >= 1 then end"
        new_code = self.transform(code)
        self.assertIn("s ~= ''", new_code)

    def test_string_empty_check_lt1(self):
        code = "if #s < 1 then end"
        new_code = self.transform(code)
        self.assertIn("s == ''", new_code)

    def test_abs_cmp_negative(self):
        code = "if math.abs(x) < -1 then end"
        new_code = self.transform(code)
        self.assertIn("if false then", new_code)

if __name__ == '__main__':
    unittest.main()
