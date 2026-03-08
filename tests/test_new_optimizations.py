import unittest
from pathlib import Path
import os
from ast_analyzer import ASTAnalyzer
from ast_transformer import ASTTransformer

class TestNewOptimizations(unittest.TestCase):
    def setUp(self):
        self.analyzer = ASTAnalyzer()
        self.transformer = ASTTransformer()
        self.test_dir = Path("tests/temp_new")
        self.test_dir.mkdir(exist_ok=True)

    def tearDown(self):
        for f in self.test_dir.glob("*"):
            f.unlink()
        if self.test_dir.exists():
            self.test_dir.rmdir()

    def test_luajit_nyi(self):
        script = """
        function actor_on_update()
            local s = string.format("%s", "test")
            for i=1, 10 do
                pairs({})
            end
        end
        """
        path = self.test_dir / "nyi.lua"
        path.write_text(script)
        findings = self.analyzer.analyze_file(path)

        nyi_findings = [f for f in findings if f.pattern_name == 'luajit_nyi_warning']
        self.assertTrue(any("string.format" in f.message for f in nyi_findings))
        self.assertTrue(any("pairs" in f.message for f in nyi_findings))

    def test_math_sqrt(self):
        script = """
        local x = 10
        local y = x^0.5
        local z = math.pow(x, 0.5)
        """
        path = self.test_dir / "sqrt.lua"
        path.write_text(script)

        # Test detection of math.pow(x, 0.5)
        findings = self.analyzer.analyze_file(path)
        pow_findings = [f for f in findings if f.pattern_name == 'math_pow_simple']
        self.assertEqual(len(pow_findings), 1)

        # Test transformation
        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn("math.sqrt(x)", new_content)

    def test_redundant_boolean(self):
        script = """
        if x == true then
            print(1)
        elseif y == false then
            print(2)
        end
        """
        path = self.test_dir / "bool.lua"
        path.write_text(script)

        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn("if x then", new_content)
        self.assertIn("elseif not y then", new_content)

    def test_slow_loop_ops(self):
        script = """
        for i=1, 10 do
            table.insert(t, 1, v)
            table.remove(t, 2)
        end
        """
        path = self.test_dir / "slow.lua"
        path.write_text(script)
        findings = self.analyzer.analyze_file(path)

        slow_ops = [f for f in findings if f.pattern_name == 'slow_loop_operation']
        self.assertEqual(len(slow_ops), 2)

    def test_dead_assignment(self):
        script = """
        local function test()
            local x = 1
            x = 2
            return x
        end
        """
        path = self.test_dir / "dead.lua"
        path.write_text(script)
        findings = self.analyzer.analyze_file(path)

        dead = [f for f in findings if f.pattern_name == 'dead_assignment']
        self.assertEqual(len(dead), 1)
        self.assertEqual(dead[0].line_num, 3)

    def test_math_random_1(self):
        script = "local x = math.random(1, 100)"
        path = self.test_dir / "rand.lua"
        path.write_text(script)

        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn("math.random(100)", new_content)

    def test_math_abs_positive(self):
        script = "local x = math.abs(time_global())"
        path = self.test_dir / "abs.lua"
        path.write_text(script)

        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn("local x = time_global()", new_content)

    def test_table_insert_append_len(self):
        script = "table.insert(t, #t + 1, v)"
        path = self.test_dir / "ins.lua"
        path.write_text(script)

        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn("t[#t+1] = v", new_content)

    def test_string_sub_to_byte(self):
        script = 'if string.sub(s, 1, 1) == "a" then end'
        path = self.test_dir / "sub.lua"
        path.write_text(script)

        findings = self.analyzer.analyze_file(path)
        sub_findings = [f for f in findings if f.pattern_name == 'string_sub_to_byte']
        self.assertEqual(len(sub_findings), 1)

    def test_repeated_member_access(self):
        script = """
        for i=1, 10 do
            print(self.object:id())
            print(self.object:section())
            print(self.object:clsid())
        end
        """
        path = self.test_dir / "member.lua"
        path.write_text(script)
        findings = self.analyzer.analyze_file(path)

        repeated = [f for f in findings if f.pattern_name == 'repeated_member_access_in_loop']
        self.assertEqual(len(repeated), 1)

    def test_math_min_max_inline(self):
        script = "local x = math.min(a, b)"
        path = self.test_dir / "min.lua"
        path.write_text(script)

        modified, new_content, _ = self.transformer.transform_file(path, backup=False, fix_yellow=True)
        self.assertTrue(modified)
        self.assertIn("a < b and a or b", new_content)

    def test_table_concat_literal(self):
        script = "local x = table.concat({a, b, c}, ':')"
        path = self.test_dir / "concat.lua"
        path.write_text(script)

        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn('a .. ":" .. b .. ":" .. c', new_content)

    def test_ipairs_hot_loop_fix(self):
        script = """function actor_on_update()
            for k, v in ipairs(t) do
                print(v)
            end
        end"""
        path = self.test_dir / "ipairs.lua"
        path.write_text(script)

        findings = self.analyzer.analyze_file(path)
        # debug: print(findings)

        # severity is GREEN now for fixable ones
        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn("for k = 1, #t do", new_content)
        self.assertIn("local v = t[k]", new_content)

    def test_redundant_return_bool(self):
        script = """function test(x)
            if x > 0 then
                return true
            else
                return false
            end
        end"""
        path = self.test_dir / "retbool.lua"
        path.write_text(script)

        modified, new_content, _ = self.transformer.transform_file(path, backup=False, fix_yellow=True)
        self.assertTrue(modified)
        self.assertIn("return x > 0", new_content)

    def test_string_format_to_concat(self):
        script = 'local x = string.format("val: %s", a)'
        path = self.test_dir / "fmt.lua"
        path.write_text(script)

        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn('"val: " .. a', new_content)

    def test_always_false_comparison(self):
        script = 'if string.lower(s) == "UPPER" then end'
        path = self.test_dir / "lower.lua"
        path.write_text(script)

        findings = self.analyzer.analyze_file(path)
        self.assertTrue(any(f.pattern_name == 'always_false_comparison' for f in findings))

    def test_constant_folding(self):
        script = "local x = 20 * 2"
        path = self.test_dir / "fold.lua"
        path.write_text(script)

        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn("local x = 40", new_content)

    def test_expo_to_mult(self):
        script = "local x = a^2"
        path = self.test_dir / "expo.lua"
        path.write_text(script)

        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn("local x = a*a", new_content)

    def test_string_sub_simple(self):
        script = "local c = string.sub(s, 1, 1)"
        path = self.test_dir / "sub_simple.lua"
        path.write_text(script)

        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn("string.char(string.byte(s))", new_content)

    def test_inplace_vector_div(self):
        script = "function test() local v = vector() v = v / 2 end"
        path = self.test_dir / "vec_div.lua"
        path.write_text(script)
        modified, new_content, _ = self.transformer.transform_file(path, backup=False, fix_yellow=True)
        self.assertTrue(modified)
        self.assertIn("v:div(2)", new_content)

    def test_inplace_vector_neg(self):
        script = "function test() local v = vector() v = -v end"
        path = self.test_dir / "vec_neg.lua"
        path.write_text(script)
        modified, new_content, _ = self.transformer.transform_file(path, backup=False, fix_yellow=True)
        self.assertTrue(modified)
        self.assertIn("v:mul(-1)", new_content)

    def test_string_concat_tostring(self):
        # Use a non-constant number (from math.random) to avoid full folding into a single string literal
        script = 'local n = math.random() local s = "res" .. tostring(n)'
        path = self.test_dir / "concat_ts.lua"
        path.write_text(script)
        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn('"res" .. n', new_content)

    def test_string_byte_range(self):
        script = "local b = string.byte(s, i, i)"
        path = self.test_dir / "byte_range.lua"
        path.write_text(script)
        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn("string.byte(s, i)", new_content)

    def test_comparison_identities_abs_neg(self):
        script = "if math.abs(x) < -1 then print(1) end"
        path = self.test_dir / "abs_neg.lua"
        path.write_text(script)
        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn("if false then", new_content)

    def test_table_insert_safety(self):
        script = "table.insert(get_table(), v)"
        path = self.test_dir / "ins_safe.lua"
        path.write_text(script)
        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertFalse(modified)

    def test_precedence_algebraic(self):
        script = "local x = 0 - (a + b)"
        path = self.test_dir / "prec.lua"
        path.write_text(script)
        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn("local x = - (a + b)", new_content)

    def test_vector_local_safety(self):
        # Should NOT transform LocalAssign
        script = "local v = v + d"
        path = self.test_dir / "vec_local.lua"
        path.write_text(script)
        modified, new_content, _ = self.transformer.transform_file(path, backup=False, fix_yellow=True)
        self.assertFalse(modified)

    def test_hoisting_shadow_safety(self):
        # Should NOT hoist if local_name is redefined in loop
        script = """
        for i=1, 10 do
            local actor = "shadow"
            print(db.actor)
        end
        """
        path = self.test_dir / "hoist_shadow.lua"
        path.write_text(script)
        modified, new_content, _ = self.transformer.transform_file(path, backup=False, fix_yellow=True)
        self.assertFalse(modified)

    def test_table_clear_pattern(self):
        script = "for k,v in pairs(t) do t[k] = nil end"
        path = self.test_dir / "clear.lua"
        path.write_text(script)
        findings = self.analyzer.analyze_file(path)
        self.assertTrue(any(f.pattern_name == 'table_clear_pattern' for f in findings))

    def test_assignment_ternary(self):
        script = "if cond then x = 1 else x = 2 end"
        path = self.test_dir / "ternary.lua"
        path.write_text(script)
        modified, new_content, _ = self.transformer.transform_file(path, backup=False, fix_yellow=True)
        self.assertTrue(modified)
        self.assertIn("x = cond and 1 or 2", new_content)

    def test_redundant_string_format(self):
        script = 'local s = string.format("%s", x)'
        path = self.test_dir / "fmt_red.lua"
        path.write_text(script)
        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn("tostring(x)", new_content)

    def test_constant_folding_extended(self):
        script = 'local s = string.sub("hello", 1, 3) .. table.concat({"a", "b"}, "-")'
        path = self.test_dir / "fold_ext.lua"
        path.write_text(script)
        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        self.assertTrue(modified)
        self.assertIn('local s = "hela-b"', new_content)

if __name__ == "__main__":
    unittest.main()
