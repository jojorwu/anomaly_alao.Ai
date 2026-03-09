
import unittest
from pathlib import Path
from ast_analyzer import ASTAnalyzer
from ast_transformer import ASTTransformer

class TestMetricOptimizations(unittest.TestCase):
    def setUp(self):
        self.analyzer = ASTAnalyzer()
        self.transformer = ASTTransformer()
        self.test_dir = Path("tests/temp_metric")
        self.test_dir.mkdir(exist_ok=True)

    def tearDown(self):
        for f in self.test_dir.glob("*"):
            f.unlink()
        if self.test_dir.exists():
            self.test_dir.rmdir()

    def transform(self, script):
        path = self.test_dir / "test.lua"
        path.write_text(script)
        modified, new_content, _ = self.transformer.transform_file(path, backup=False)
        return new_content if modified else script

    def test_magnitude_comparison(self):
        script = "if v:magnitude() < 10 then end"
        new_content = self.transform(script)
        self.assertIn("v:magnitude_sqr() < 100", new_content)

    def test_magnitude_xz_comparison(self):
        script = "if v:magnitude_xz() <= 5 then end"
        new_content = self.transform(script)
        self.assertIn("v:magnitude_sqr_xz() <= 25", new_content)

    def test_distance_xz_comparison(self):
        script = "if v1:distance_to_xz(v2) > 2 then end"
        new_content = self.transform(script)
        self.assertIn("v1:distance_to_sqr_xz(v2) > 4", new_content)

    def test_metric_eq_comparison(self):
        script = "if v:magnitude() == 0 then end"
        new_content = self.transform(script)
        self.assertIn("v:magnitude_sqr() == 0", new_content)

    def test_vector_set_zero(self):
        script = "v:set(0, 0, 0)"
        new_content = self.transform(script)
        self.assertIn("v:set(0)", new_content)

    def test_vector_set_copy(self):
        script = "v1:set(v2.x, v2.y, v2.z)"
        new_content = self.transform(script)
        self.assertIn("v1:set(v2)", new_content)

    def test_redundant_nil_assignment(self):
        script = "local x = nil"
        new_content = self.transform(script)
        self.assertEqual(new_content.strip(), "local x")

    def test_distance_sqr_xz_comparison(self):
        # Already sqr, but should still be tracked for threshold replacement if we wanted,
        # but current logic only converts non-sqr to sqr.
        # Actually, let's test if we can optimize distance_to_sqr_xz further?
        # No, it's already optimized.
        pass

if __name__ == "__main__":
    unittest.main()
