import unittest
from pathlib import Path

# Get the src directory path
src_path = Path(__file__).resolve().parents[1] / "src"
import importlib.util

spec = importlib.util.spec_from_file_location(
    "thread_the_needle", src_path / "thread_the_needle" / "__init__.py"
)
ttn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ttn)


from thread_the_needle.thread_the_needle import make_thread_the_needle_walls


class TestImports(unittest.TestCase):

    def test_make_thread_the_needle(self):
        make = ttn.make
        env = make("thread_the_needle")
        self.assertEqual(env.__class__.__name__, "GridWorldEnv")

    def test_make_open_env(self):
        make = ttn.make

        env = make("open_env")
        self.assertEqual(env.__class__.__name__, "GridWorldEnv")

    def test_thread_the_needle_walls(self):
        make = ttn.make
        env = make("thread_the_needle")
        walls = env.transition_model.walls
        reference_walls = make_thread_the_needle_walls(20)

        self.assertIsInstance(walls, list)
        self.assertEqual(len(walls), len(reference_walls))

        # Check that every element in walls and reference_walls are equal
        for wall, ref_wall in zip(walls, reference_walls):
            self.assertIsInstance(wall, list)
            self.assertIsInstance(ref_wall, list)
            self.assertEqual(len(wall), len(ref_wall))
            for w, rw in zip(wall, ref_wall):
                self.assertEqual(w, rw)
