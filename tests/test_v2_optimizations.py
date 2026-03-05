import unittest
from pathlib import Path
from ast_analyzer import ASTAnalyzer
from ast_transformer import ASTTransformer

class TestV2Optimizations(unittest.TestCase):
    def setUp(self):
        self.analyzer = ASTAnalyzer()
        self.transformer = ASTTransformer()
        self.test_dir = Path("tests/temp_v2")
        self.test_dir.mkdir(exist_ok=True)

    def tearDown(self):
        for f in self.test_dir.glob("*"):
            f.unlink()
        if self.test_dir.exists():
            self.test_dir.rmdir()

    def test_string_match_existence(self):
        script = 'if string.match(s, "plain") then end'
        path = self.test_dir / "match.lua"
        path.write_text(script)

        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn('string.find(s, "plain", 1, true)', new_content)

    def test_unpack_to_indexing(self):
        script = 'local a, b = unpack(t)'
        path = self.test_dir / "unpack.lua"
        path.write_text(script)

        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn('local a, b = t[1], t[2]', new_content)

    def test_divide_by_constant(self):
        script = 'local x = y / 2'
        path = self.test_dir / "div.lua"
        path.write_text(script)

        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn('local x = y * 0.5', new_content)

    def test_if_nil_assign(self):
        script = 'if x == nil then x = 10 end'
        path = self.test_dir / "nil_assign.lua"
        path.write_text(script)

        # This is YELLOW severity, so we need fix_yellow=True
        modified, new_content, _ = self.transformer.transform_file(path, backup=False, fix_yellow=True)
        self.assertTrue(modified)
        self.assertIn('x = x or 10', new_content)

    def test_generalized_string_sub(self):
        script = 'local c = string.sub(s, 5, 5)'
        path = self.test_dir / "sub_gen.lua"
        path.write_text(script)

        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn('string.char(string.byte(s, 5))', new_content)

if __name__ == "__main__":
    unittest.main()
