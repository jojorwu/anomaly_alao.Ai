import unittest
from pathlib import Path
from ast_analyzer import ASTAnalyzer
from ast_transformer import ASTTransformer

class TestV10Optimizations(unittest.TestCase):
    def setUp(self):
        self.analyzer = ASTAnalyzer()
        self.transformer = ASTTransformer()
        self.test_dir = Path("tests/temp_v10")
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

    def test_math_pow_general(self):
        script = "local a = math.pow(x, y)"
        expected = "local a = x ^ y"
        self.assertEqual(self.transform(script), expected)

    def test_table_concat_complex_arg(self):
        # Regression test for fragile comma searching in table.concat
        script = 'local s = table.concat(get_items(1, 2), "")'
        expected = 'local s = table.concat(get_items(1, 2))'
        self.assertEqual(self.transform(script), expected)

    def test_math_pow_precedence(self):
        script = "local a = math.pow(x + y, z * w)"
        expected = "local a = (x + y) ^ (z * w)"
        self.assertEqual(self.transform(script), expected)

    def test_bit_identities_same_arg(self):
        script = """
local a = bit.bxor(x, x)
local b = bit.band(y, y)
local c = bit.bor(z, z)
"""
        new_content = self.transform(script)
        self.assertIn("local a = 0", new_content)
        self.assertIn("local b = y", new_content)
        self.assertIn("local c = z", new_content)

    def test_table_concat_default_sep(self):
        script = 'local s = table.concat(t, "")'
        expected = 'local s = table.concat(t)'
        self.assertEqual(self.transform(script), expected)

    def test_inverted_return_bool(self):
        script = "if cond then return false else return true end"
        # We expect return not (cond) to be safe with precedence
        new_content = self.transform(script)
        self.assertIn("return not (cond)", new_content)

    def test_inverted_return_bool_complex(self):
        script = "if a + b > 0 then return false else return true end"
        new_content = self.transform(script)
        self.assertIn("return not (a + b > 0)", new_content)

    def test_inverted_return_bool_precedence(self):
        # Regression test for precedence: not a + b -> not (a + b)
        script = "if a + b then return false else return true end"
        new_content = self.transform(script)
        self.assertIn("return not (a + b)", new_content)

    def test_string_sub_all(self):
        script = 'local s = "hello"; local s2 = string.sub(s, 1, -1)'
        new_content = self.transform(script)
        self.assertIn('local s2 = s', new_content)

    def test_nested_redundant_string_calls(self):
        script = """
local a = string.lower(string.lower(s))
local b = string.upper(string.upper(s))
"""
        new_content = self.transform(script)
        self.assertIn("local a = string.lower(s)", new_content)
        self.assertIn("local b = string.upper(s)", new_content)

    def test_math_log_exp(self):
        script = "local a = math.log(math.exp(x))"
        new_content = self.transform(script)
        self.assertIn("local a = x", new_content)

    def test_math_log10(self):
        script = "local a = math.log(x, 10)"
        expected = "local a = math.log10(x)"
        self.assertEqual(self.transform(script), expected)

    def test_bit_bnot_bnot(self):
        script = "local a = bit.bnot(bit.bnot(x))"
        new_content = self.transform(script)
        self.assertIn("local a = x", new_content)

    def test_math_exp_log(self):
        script = "local a = math.exp(math.log(x))"
        new_content = self.transform(script)
        self.assertIn("local a = x", new_content)

    def test_nested_math_flatten(self):
        script = "local a = math.max(math.max(x, y), z)"
        new_content = self.transform(script)
        self.assertIn("local a = math.max(x, y, z)", new_content)

    def test_redundant_string_find_args(self):
        script = "local a = string.find(s, p, 1)"
        new_content = self.transform(script)
        self.assertIn("local a = string.find(s, p)", new_content)

        script = "local a = string.find(s, p, 1, false)"
        new_content = self.transform(script)
        self.assertIn("local a = string.find(s, p)", new_content)

    def test_sub_self(self):
        script = "local x = 10; local a = x - x"
        new_content = self.transform(script)
        self.assertIn("local a = 0", new_content)

    def test_string_sub_len(self):
        script = 'local s = "hello"; local a = string.sub(s, 1, #s)'
        new_content = self.transform(script)
        self.assertIn("local a = s", new_content)

    def test_string_byte_char(self):
        script = "local a = string.byte(string.char(x))"
        new_content = self.transform(script)
        self.assertIn("local a = x", new_content)

    def test_bit_bxor_neg1(self):
        script = "local a = bit.bxor(x, -1)"
        new_content = self.transform(script)
        self.assertIn("local a = bit.bnot(x)", new_content)

    def test_mult_zero(self):
        script = "local a = x * 0"
        # We need type inference for x to be number, which is hard in small scripts
        # Let's use a script that provides type hints
        script = "local x = 5; local a = x * 0"
        new_content = self.transform(script)
        self.assertIn("local a = 0", new_content)

    def test_div_zero_numerator(self):
        script = "local x = 5; local a = 0 / x"
        new_content = self.transform(script)
        self.assertIn("local a = 0", new_content)

    def test_mult_zero_unsafe(self):
        # table * 0 should NOT be optimized
        script = "local x = {}; local a = x * 0"
        new_content = self.transform(script)
        self.assertIn("local a = x * 0", new_content)

    def test_math_pow_square(self):
        script = "local x = 5; local a = math.pow(x, 2)"
        new_content = self.transform(script)
        self.assertIn("local a = x*x", new_content)

    def test_math_pow_cube(self):
        script = "local x = 5; local a = math.pow(x, 3)"
        new_content = self.transform(script)
        self.assertIn("local a = x*x*x", new_content)

    def test_math_pow_sqrt(self):
        script = "local a = math.pow(x, 0.5)"
        new_content = self.transform(script)
        self.assertIn("local a = math.sqrt(x)", new_content)

    def test_math_pow_neg1(self):
        script = "local a = math.pow(x, -1)"
        new_content = self.transform(script)
        self.assertIn("local a = 1/x", new_content)

    def test_string_sub_len_call(self):
        script = 'local s = "hello"; local a = string.sub(s, 1, string.len(s))'
        new_content = self.transform(script)
        self.assertIn("local a = s", new_content)

    def test_abs_mult_self(self):
        script = "local x = 5; local a = math.abs(x) * math.abs(x)"
        new_content = self.transform(script)
        self.assertIn("local a = x * x", new_content)

    def test_nested_flatten_bitwise(self):
        script = "local a = bit.band(bit.band(x, y), z)"
        new_content = self.transform(script)
        self.assertIn("local a = bit.band(x, y, z)", new_content)

    def test_bit_complements(self):
        script = """
local a = bit.band(x, bit.bnot(x))
local b = bit.bor(y, bit.bnot(y))
local c = bit.bxor(z, bit.bnot(z))
"""
        new_content = self.transform(script)
        self.assertIn("local a = 0", new_content)
        self.assertIn("local b = -1", new_content)
        self.assertIn("local c = -1", new_content)

if __name__ == "__main__":
    unittest.main()
