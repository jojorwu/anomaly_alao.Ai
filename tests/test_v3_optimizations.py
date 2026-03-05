import unittest
from pathlib import Path
from ast_analyzer import ASTAnalyzer
from ast_transformer import ASTTransformer

class TestV3Optimizations(unittest.TestCase):
    def setUp(self):
        self.analyzer = ASTAnalyzer()
        self.transformer = ASTTransformer()
        self.test_dir = Path("tests/temp_v3")
        self.test_dir.mkdir(exist_ok=True)

    def tearDown(self):
        for f in self.test_dir.glob("*"):
            f.unlink()
        if self.test_dir.exists():
            self.test_dir.rmdir()

    def test_redundant_tonumber(self):
        script = 'local x = 10\nlocal y = tonumber(x)'
        path = self.test_dir / "tonum.lua"
        path.write_text(script)

        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn('local y = x', new_content)

    def test_redundant_tostring(self):
        script = 'local x = "test"\nlocal y = tostring(x)'
        path = self.test_dir / "tostr.lua"
        path.write_text(script)

        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn('local y = x', new_content)

    def test_string_byte_1(self):
        script = 'local b = string.byte(s, 1)'
        path = self.test_dir / "byte1.lua"
        path.write_text(script)

        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn('string.byte(s)', new_content)

    def test_return_ternary(self):
        script = """function test(x)
            if x > 0 then
                return 1
            else
                return 0
            end
        end"""
        path = self.test_dir / "ret_ternary.lua"
        path.write_text(script)

        modified, new_content, _ = self.transformer.transform_file(path, backup=False, fix_yellow=True)
        self.assertTrue(modified)
        self.assertIn('return x > 0 and 1 or 0', new_content)

    def test_pairs_to_next(self):
        script = """function actor_on_update()
            for k,v in pairs(t) do
                print(v)
            end
        end"""
        path = self.test_dir / "pairs.lua"
        path.write_text(script)
        findings = self.analyzer.analyze_file(path)
        self.assertTrue(any(f.pattern_name == 'pairs_to_next' for f in findings))

if __name__ == "__main__":
    unittest.main()
