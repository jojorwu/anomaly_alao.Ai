import unittest
from pathlib import Path
import os
import shutil
from ast_analyzer import ASTAnalyzer
from ast_transformer import ASTTransformer

class TestOptimizer(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("tests/temp_scripts")
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_analyzer_table_insert(self):
        script_path = self.test_dir / "test_insert.lua"
        script_path.write_text('table.insert(t, v)')
        analyzer = ASTAnalyzer()
        findings = analyzer.analyze_file(script_path)
        self.assertTrue(any(f.pattern_name == 'table_insert_append' for f in findings))

    def test_analyzer_math_pow(self):
        script_path = self.test_dir / "test_pow.lua"
        script_path.write_text('local x = math.pow(base, 2)')
        analyzer = ASTAnalyzer()
        findings = analyzer.analyze_file(script_path)
        self.assertTrue(any(f.pattern_name == 'math_pow_simple' for f in findings))

    def test_transformer_table_insert(self):
        script_path = self.test_dir / "test_insert_fix.lua"
        content = 'table.insert(my_table, "value")'
        script_path.write_text(content)
        transformer = ASTTransformer()
        modified, new_content, _ = transformer.transform_file(script_path, backup=False)
        self.assertTrue(modified)
        self.assertEqual(new_content.strip(), 'my_table[#my_table+1] = "value"')

    def test_transformer_string_concat(self):
        script_path = self.test_dir / "test_concat_fix.lua"
        content = 'local s = ""\nfor i=1,10 do\n    s = s .. "test"\nend'
        script_path.write_text(content)
        transformer = ASTTransformer()
        modified, new_content, _ = transformer.transform_file(script_path, backup=False, experimental=True)
        self.assertTrue(modified)
        self.assertIn("_s_parts = {}", new_content)
        self.assertIn("table.concat(_s_parts)", new_content)

if __name__ == "__main__":
    unittest.main()
