import unittest
from pathlib import Path
import os
import shutil
from ast_analyzer import ASTAnalyzer
from ast_transformer import ASTTransformer
from whole_program_analyzer import WholeProgramAnalyzer
from discovery import discover_mods, get_mod_info

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

    def test_transformer_long_string(self):
        script_path = self.test_dir / "test_long_string.lua"
        content = 'table.insert(t, [[long string with "quotes" and (parens)]])'
        script_path.write_text(content)
        transformer = ASTTransformer()
        modified, new_content, _ = transformer.transform_file(script_path, backup=False)
        self.assertTrue(modified)
        self.assertIn('t[#t+1] = [[long string with "quotes" and (parens)]]', new_content)

    def test_whole_program_analyzer(self):
        script1 = self.test_dir / "mod1.script"
        script1.write_text('function global_func() end')
        script2 = self.test_dir / "mod2.script"
        script2.write_text('global_func()')

        analyzer = WholeProgramAnalyzer()
        analysis = analyzer.analyze_files([script1, script2])

        self.assertIn('global_func', analysis.definitions)
        self.assertIn('global_func', analysis.usages)

    def test_analyzer_string_find_plain(self):
        script_path = self.test_dir / "test_find.lua"
        script_path.write_text('string.find(s, "plain")')
        analyzer = ASTAnalyzer()
        findings = analyzer.analyze_file(script_path)
        self.assertTrue(any(f.pattern_name == 'string_find_plain' for f in findings))

    def test_analyzer_pairs_hot_callback(self):
        script_path = self.test_dir / "test_pairs.lua"
        script_path.write_text('function actor_on_update()\n  for k,v in pairs(t) do end\nend')
        analyzer = ASTAnalyzer()
        findings = analyzer.analyze_file(script_path)
        self.assertTrue(any(f.pattern_name == 'pairs_on_array' for f in findings))

    def test_transformer_string_find_plain(self):
        script_path = self.test_dir / "test_find_fix.lua"
        content = 'string.find(str, "key")'
        script_path.write_text(content)
        transformer = ASTTransformer()
        modified, new_content, _ = transformer.transform_file(script_path, backup=False)
        self.assertTrue(modified)
        self.assertIn('string.find(str, "key", 1, true)', new_content)

    def test_analyzer_unused_parameter(self):
        script_path = self.test_dir / "test_param.lua"
        script_path.write_text('function my_func(used, unused) return used end')
        analyzer = ASTAnalyzer()
        findings = analyzer.analyze_file(script_path)
        self.assertTrue(any(f.pattern_name == 'unused_parameter' and f.details.get('var_name') == 'unused' for f in findings))
        self.assertFalse(any(f.pattern_name == 'unused_parameter' and f.details.get('var_name') == 'used' for f in findings))

    def test_analyzer_string_format_loop(self):
        script_path = self.test_dir / "test_format_loop.lua"
        script_path.write_text('for i=1,10 do string.format("%d", i) end')
        analyzer = ASTAnalyzer()
        findings = analyzer.analyze_file(script_path)
        self.assertTrue(any(f.pattern_name == 'string_format_in_loop' for f in findings))

    def test_analyzer_table_insert_front(self):
        script_path = self.test_dir / "test_insert_front.lua"
        script_path.write_text('table.insert(my_table, 1, "first")')
        analyzer = ASTAnalyzer()
        findings = analyzer.analyze_file(script_path)
        self.assertTrue(any(f.pattern_name == 'table_insert_front' for f in findings))

    def test_discovery_standard(self):
        mod_dir = self.test_dir / "my_mod"
        scripts_dir = mod_dir / "gamedata" / "scripts"
        scripts_dir.mkdir(parents=True)
        (scripts_dir / "test.script").write_text("-- test")

        mods = discover_mods(self.test_dir)
        self.assertIn("my_mod", mods)
        self.assertEqual(len(mods["my_mod"]), 1)

    def test_get_mod_info(self):
        mod_dir = self.test_dir / "info_mod"
        mod_dir.mkdir()
        (mod_dir / "meta.ini").write_text("name=Custom Mod\nversion=1.2.3")

        info = get_mod_info(mod_dir)
        self.assertEqual(info["name"], "Custom Mod")
        self.assertEqual(info["version"], "1.2.3")

if __name__ == "__main__":
    unittest.main()
