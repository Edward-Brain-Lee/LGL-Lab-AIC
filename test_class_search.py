"""Regression checks for row alignment, paired changes and proxy uncertainty."""
import tempfile
import unittest
from pathlib import Path

from class_search import load_predictions, measure


class ClassSearchTests(unittest.TestCase):
    def test_paired_decomposition_and_uncertainty(self):
        ref = dict(a='0000', b='0000', c='0001', d='0001')
        base = dict(a='0000', b='0001', c='0001', d='0000')
        new = dict(a='0001', b='0000', c='0001', d='0001')
        stats, rows, _ = measure(new, ref, base, {'0001'}, {'0000': 200, '0001': 10})
        self.assertEqual((stats['new_win'], stats['old_win']), (2, 1))
        self.assertEqual(stats['delta_pp'], 25)
        self.assertEqual(stats['reliable_delta_pp'], 0)
        self.assertEqual((stats['delta_lower_pp'], stats['delta_upper_pp']), (-25, 25))
        self.assertEqual(stats['tail20_macro'], 1)
        self.assertEqual(sum(r['net'] for r in rows), 1)

    def test_alignment_fails(self):
        with self.assertRaises(ValueError):
            measure(dict(a='0000'), dict(b='0000'), dict(b='0000'))

    def test_duplicate_and_invalid_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.csv'
            for text in ('a,0000\na,0001\n', 'a,0\n', ''):
                path.write_text(text, encoding='utf-8')
                with self.assertRaises(ValueError):
                    load_predictions(path)


if __name__ == '__main__':
    unittest.main()
