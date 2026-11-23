import json
import tempfile
import unittest
from pathlib import Path
from train import ROOT,load_data,metrics,run

class TrainingTests(unittest.TestCase):
    def test_source_schema(self):
        d,f=load_data(ROOT/"Real estate.csv");self.assertEqual(len(d),414);self.assertEqual(len(f),6)
    def test_metric_direction_and_values(self):
        m=metrics([1,2,3],[2,2,2]);self.assertAlmostEqual(m["mae"],2/3);self.assertEqual(m["r2"],0)
    def test_end_to_end_report(self):
        with tempfile.TemporaryDirectory() as t:
            result=run(output=Path(t))
            self.assertEqual(result["train_rows"]+result["holdout_rows"],414)
            self.assertLess(result["holdout"]["ridge"]["mae"],result["holdout"]["median_baseline"]["mae"])
            self.assertTrue((Path(t)/"holdout.png").is_file())
            self.assertEqual(json.loads((Path(t)/"metrics.json").read_text()),result)
if __name__=="__main__":unittest.main()
