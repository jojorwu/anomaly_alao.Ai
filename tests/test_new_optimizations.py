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

    def test_inplace_vector_op(self):
        script = """
        function test()
            local v = vector()
            local d = vector()
            for i=1, 10 do
                v = v + d
            end
        end
        """
        path = self.test_dir / "vadd.lua"
        path.write_text(script)

        modified, new_content, _ = self.transformer.transform_file(path, backup=False, fix_yellow=True)
        self.assertTrue(modified)
        self.assertIn("v:add(d)", new_content)

    def test_table_emptiness(self):
        script = """
        function test()
            local t = {}
            if #t == 0 then
                return
            end
        end
        """
        path = self.test_dir / "empty.lua"
        path.write_text(script)

        modified, new_content, _ = self.transformer.transform_file(path, backup=False, fix_yellow=True)
        self.assertTrue(modified)
        self.assertIn("next(t) == nil", new_content)

    def test_loop_invariant_global_hoist(self):
        script = """
        function test()
            for i=1, 10 do
                local a = db.actor:position()
            end
        end
        """
        path = self.test_dir / "hoist.lua"
        path.write_text(script)

        modified, new_content, _ = self.transformer.transform_file(path, backup=False, fix_yellow=True)
        self.assertTrue(modified)
        self.assertIn("local actor = db.actor", new_content)
        self.assertIn("actor:position()", new_content)

    def test_math_max_comparison(self):
        script = "if math.max(a, b) == a then end"
        path = self.test_dir / "maxcomp.lua"
        path.write_text(script)

        findings = self.analyzer.analyze_file(path)
        self.assertTrue(any(f.pattern_name == 'math_identity' and 'a >= b' in f.message for f in findings))

    def test_logical_identity_expansion(self):
        script = "if x ~= nil and x ~= false then end"
        path = self.test_dir / "logic.lua"
        path.write_text(script)

        findings = self.analyzer.analyze_file(path)
        self.assertTrue(any(f.pattern_name == 'logical_identity' and f.details.get('replacement') == 'x' for f in findings))

if __name__ == "__main__":
    unittest.main()
