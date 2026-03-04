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

if __name__ == "__main__":
    unittest.main()
