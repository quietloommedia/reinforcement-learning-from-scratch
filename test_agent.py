"""Meaningful correctness checks; run: python -m unittest -v"""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from agent import update, greedy_action, choose_action, exploration_rate
from records import save_agent, load_agent


class LearningTests(unittest.TestCase):
    def test_continuing_transition(self):
        q = np.zeros((16, 4)); q[8, 2] = .2; q[9, 1] = .8
        update(q, 8, 2, 0, 9, False, .1, .99)
        self.assertAlmostEqual(q[8, 2], .2592)

    def test_terminal_does_not_bootstrap(self):
        for reward, expected in ((1, .28), (0, .18)):
            q = np.zeros((16, 4)); q[14, 2] = .2; q[15] = 99
            update(q, 14, 2, reward, 15, True, .1, .99)
            self.assertAlmostEqual(q[14, 2], expected)

    def test_external_time_limit_still_bootstraps(self):
        # Truncated=True ends the rollout, but terminated=False is passed to update.
        q = np.zeros((16, 4)); q[8, 2] = .2; q[9, 1] = .8
        update(q, 8, 2, 0, 9, terminated=False, alpha=.1, gamma=.99)
        self.assertAlmostEqual(q[8, 2], .2592)

    def test_ties_and_exploration(self):
        q = np.zeros((16, 4)); rng = np.random.default_rng(7)
        self.assertEqual({greedy_action(q, 0, rng) for _ in range(100)}, {0, 1, 2, 3})
        q[0, 2] = 1
        self.assertEqual({choose_action(q, 0, 0, rng) for _ in range(100)}, {2})
        self.assertEqual({choose_action(q, 0, 1, rng) for _ in range(100)}, {0, 1, 2, 3})
        self.assertEqual(exploration_rate(0), 1)
        self.assertEqual(exploration_rate(3000), .05)

    def test_save_reload_and_reject_bad_shape(self):
        with tempfile.TemporaryDirectory() as folder:
            config = dict(environment="FrozenLake-v1", map_name="4x4", is_slippery=False)
            q = np.arange(64, dtype=float).reshape(16, 4)
            save_agent(folder, q, config)
            loaded, actual = load_agent(folder)
            np.testing.assert_array_equal(loaded, q)
            self.assertEqual(actual, config)
            np.save(Path(folder) / "q.npy", np.zeros((2, 4)))
            with self.assertRaises(ValueError): load_agent(folder)

if __name__ == "__main__":
    unittest.main()
