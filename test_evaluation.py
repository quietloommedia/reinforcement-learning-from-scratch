import unittest
import numpy as np
from evaluate import evaluate


class EvaluationTests(unittest.TestCase):
    def test_evaluation_counts_and_does_not_learn(self):
        q = np.zeros((16, 4))
        before = q.copy()
        rows = evaluate(q, episodes=25)
        np.testing.assert_array_equal(q, before)
        self.assertEqual(len(rows), 25)
        for row in rows:
            self.assertIn(
                row["outcome"], ("goal", "hole", "timeout")
            )
            self.assertTrue(1 <= row["steps"] <= 100)


if __name__ == "__main__":
    unittest.main()
