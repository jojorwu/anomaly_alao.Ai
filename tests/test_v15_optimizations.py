
import unittest
from pathlib import Path
import tempfile
import os
from ast_analyzer import ASTAnalyzer
from ast_transformer import ASTTransformer

class TestV15Optimizations(unittest.TestCase):
    def setUp(self):
        self.analyzer = ASTAnalyzer()
        self.transformer = ASTTransformer()

    def test_vector_method_single_arg(self):
        code = "v:add(1, 1, 1)"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lua', delete=False) as f:
            f.write(code)
            tmp_path = Path(f.name)
        try:
            modified, new_code, _ = self.transformer.transform_file(tmp_path)
            self.assertTrue(modified)
            self.assertIn("v:add(1)", new_code)
        finally:
            os.remove(tmp_path)

    def test_vector_method_copy(self):
        code = "v:set(v2.x, v2.y, v2.z)"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lua', delete=False) as f:
            f.write(code)
            tmp_path = Path(f.name)
        try:
            modified, new_code, _ = self.transformer.transform_file(tmp_path)
            self.assertTrue(modified)
            self.assertIn("v:set(v2)", new_code)
        finally:
            os.remove(tmp_path)

    def test_vector_init_zero(self):
        code = "local v = vector(0, 0, 0)"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lua', delete=False) as f:
            f.write(code)
            tmp_path = Path(f.name)
        try:
            modified, new_code, _ = self.transformer.transform_file(tmp_path)
            self.assertTrue(modified)
            self.assertIn("vector()", new_code)
        finally:
            os.remove(tmp_path)

    def test_metric_comparison_sqr(self):
        code = "if a:distance_to(b) < a:distance_to(c) then end"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lua', delete=False) as f:
            f.write(code)
            tmp_path = Path(f.name)
        try:
            modified, new_code, _ = self.transformer.transform_file(tmp_path)
            self.assertTrue(modified)
            self.assertIn("a:distance_to_sqr(b) < a:distance_to_sqr(c)", new_code)
        finally:
            os.remove(tmp_path)

    def test_metric_direct_access(self):
        code = "local d = obj1:position():distance_to(obj2:position())"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lua', delete=False) as f:
            f.write(code)
            tmp_path = Path(f.name)
        try:
            modified, new_code, _ = self.transformer.transform_file(tmp_path)
            self.assertTrue(modified)
            self.assertIn("obj1:distance_to(obj2)", new_code)
        finally:
            os.remove(tmp_path)

    def test_math_clamp_reversed(self):
        code = "local c = math.max(0, math.min(100, x))"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lua', delete=False) as f:
            f.write(code)
            tmp_path = Path(f.name)
        try:
            modified, new_code, _ = self.transformer.transform_file(tmp_path)
            self.assertTrue(modified)
            self.assertIn("clamp(x, 0, 100)", new_code)
        finally:
            os.remove(tmp_path)

    def test_redundant_nested_tonumber_tostring(self):
        code = "local n = tonumber(tostring(x))"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lua', delete=False) as f:
            f.write(code)
            tmp_path = Path(f.name)
        try:
            modified, new_code, _ = self.transformer.transform_file(tmp_path)
            self.assertTrue(modified)
            self.assertIn("tonumber(x)", new_code)
        finally:
            os.remove(tmp_path)

    def test_vector_redundant_op(self):
        code = "v:add(0)"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lua', delete=False) as f:
            f.write(code)
            tmp_path = Path(f.name)
        try:
            modified, new_code, _ = self.transformer.transform_file(tmp_path)
            self.assertTrue(modified)
            self.assertEqual(new_code.strip(), "v")
        finally:
            os.remove(tmp_path)

    def test_vector_mul_zero(self):
        code = "v:mul(0)"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lua', delete=False) as f:
            f.write(code)
            tmp_path = Path(f.name)
        try:
            modified, new_code, _ = self.transformer.transform_file(tmp_path)
            self.assertTrue(modified)
            self.assertIn("v:set(0)", new_code)
        finally:
            os.remove(tmp_path)

    def test_bit_multi_arg_deduplicate(self):
        code = "bit.band(x, x, y)"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lua', delete=False) as f:
            f.write(code)
            tmp_path = Path(f.name)
        try:
            modified, new_code, _ = self.transformer.transform_file(tmp_path)
            self.assertTrue(modified)
            self.assertIn("bit.band(x, y)", new_code)
        finally:
            os.remove(tmp_path)

    def test_bit_bxor_cancellation(self):
        code = "bit.bxor(x, y, x)"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lua', delete=False) as f:
            f.write(code)
            tmp_path = Path(f.name)
        try:
            modified, new_code, _ = self.transformer.transform_file(tmp_path)
            self.assertTrue(modified)
            self.assertEqual(new_code.strip(), "y")
        finally:
            os.remove(tmp_path)

    def test_bit_multi_arg_zero(self):
        code = "bit.band(x, 0, y)"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lua', delete=False) as f:
            f.write(code)
            tmp_path = Path(f.name)
        try:
            modified, new_code, _ = self.transformer.transform_file(tmp_path)
            self.assertTrue(modified)
            self.assertEqual(new_code.strip(), "0")
        finally:
            os.remove(tmp_path)

if __name__ == '__main__':
    unittest.main()
