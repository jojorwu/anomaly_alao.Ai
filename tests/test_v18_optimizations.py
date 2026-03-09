
import unittest
from pathlib import Path
import tempfile
import os
from ast_analyzer import ASTAnalyzer
from ast_transformer import ASTTransformer

class TestV18Optimizations(unittest.TestCase):
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

    def test_table_sort_redundant(self):
        code = "table.sort(t, function(a,b) return a < b end)"
        new_code = self.transform(code)
        self.assertEqual(new_code.strip(), "table.sort(t)")

    def test_logical_absorption(self):
        code = "local x = (a and b) or a"
        new_code = self.transform(code)
        self.assertIn("local x = a", new_code)

        code = "local x = a or (a and b)"
        new_code = self.transform(code)
        self.assertIn("local x = a", new_code)

    def test_logical_absorption_nested(self):
        code = "local x = (a or b) and a"
        new_code = self.transform(code)
        self.assertIn("local x = a", new_code)

    def test_vector_inverse_chain(self):
        code = "v:add(v2):sub(v2)"
        new_code = self.transform(code)
        self.assertEqual(new_code.strip(), "v")

    def test_vector_inverse_chain_mul_div(self):
        code = "v:mul(s):div(s)"
        new_code = self.transform(code)
        self.assertEqual(new_code.strip(), "v")

if __name__ == '__main__':
    unittest.main()
