
import unittest
from pathlib import Path
import tempfile
import os
from ast_analyzer import ASTAnalyzer
from ast_transformer import ASTTransformer

class TestV14Optimizations(unittest.TestCase):
    def setUp(self):
        self.analyzer = ASTAnalyzer()
        self.transformer = ASTTransformer()

    def test_inplace_vector_op(self):
        code = """
function test()
    local v = vector():set(1,1,1)
    local d = vector():set(0,1,0)
    for i=1, 10 do
        v = v + d
    end
end
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lua', delete=False) as f:
            f.write(code)
            tmp_path = Path(f.name)

        try:
            # inplace_vector_op is YELLOW severity now
            modified, new_code, count = self.transformer.transform_file(tmp_path, fix_yellow=True)
            self.assertTrue(modified)
            self.assertIn("v:add(d)", new_code)
            self.assertNotIn("v = v + d", new_code)
        finally:
            os.remove(tmp_path)

    def test_table_emptiness_check(self):
        code = """
function test()
    local t = {}
    if #t == 0 then
        print("empty")
    end
end
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lua', delete=False) as f:
            f.write(code)
            tmp_path = Path(f.name)

        try:
            # table_emptiness_check is YELLOW severity
            modified, new_code, count = self.transformer.transform_file(tmp_path, fix_yellow=True)
            self.assertTrue(modified)
            self.assertIn("next(t) == nil", new_code)
        finally:
            os.remove(tmp_path)

    def test_loop_invariant_global(self):
        code = """
function test()
    for i=1, 100 do
        local a = db.actor:position()
        local b = db.actor:direction()
    end
end
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lua', delete=False) as f:
            f.write(code)
            tmp_path = Path(f.name)

        try:
            # loop_invariant_global is YELLOW severity
            modified, new_code, count = self.transformer.transform_file(tmp_path, fix_yellow=True)
            self.assertTrue(modified)
            self.assertIn("local actor = db.actor", new_code)
            self.assertIn("actor:position()", new_code)
            self.assertIn("actor:direction()", new_code)
        finally:
            os.remove(tmp_path)

    def test_loop_invariant_global_collision(self):
        code = """
function test()
    local actor = "not a global"
    for i=1, 100 do
        local a = db.actor:position()
    end
end
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lua', delete=False) as f:
            f.write(code)
            tmp_path = Path(f.name)

        try:
            # Should NOT hoist because 'actor' is already a local variable
            modified, new_code, count = self.transformer.transform_file(tmp_path, fix_yellow=True)
            self.assertFalse(modified)
            self.assertNotIn("local actor = db.actor", new_code)
        finally:
            os.remove(tmp_path)

    def test_inplace_vector_op_inferred(self):
        code = """
function test(v)
    -- v is not known to be a vector here (parameter)
    local d = vector()
    v = v + d -- should not trigger without knowing v is a vector
end

function test_inferred()
    local v = vector()
    local d = vector()
    v = v + d -- should trigger
end
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lua', delete=False) as f:
            f.write(code)
            tmp_path = Path(f.name)

        try:
            modified, new_code, count = self.transformer.transform_file(tmp_path, fix_yellow=True)
            self.assertTrue(modified)
            self.assertIn("v:add(d)", new_code)
            # Should only happen once in the test_inferred function
            self.assertEqual(new_code.count("v:add(d)"), 1)
        finally:
            os.remove(tmp_path)

    def test_constant_folding_priority(self):
        # Regression test for priority conflict
        script = "local x = -1 + 2"
        path = Path("fold_test.lua")
        path.write_text(script)

        try:
            modified, new_content, _ = self.transformer.transform_file(path, backup=False)
            self.assertTrue(modified)
            self.assertIn("local x = 1", new_content)
        finally:
            if path.exists(): path.unlink()

if __name__ == '__main__':
    unittest.main()
