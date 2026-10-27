"""Independent small fixtures for leakage, dictionary rules, and edge cases."""
import copy
import unittest
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from quality import validate, conflict_audit
from transforms import AdultCleaner, FeatureEngineer, build_preprocessor
from schema import FEATURES, NUMERIC


def fixture():
    rows = [[20,"Private",10000,"HS-grad",9,"Never-married","Sales","Own-child","White","Male",0,0,20,"United-States"],
            [30,"Private",20000,"Bachelors",13,"Never-married","Sales","Not-in-family","Black","Female",100,0,40,"Poland"],
            [40,"State-gov",30000,"HS-grad",9,"Married-civ-spouse","Adm-clerical","Husband","White","Male",200,100,60,"United-States"],
            [50,"?",40000,"HS-grad",9,"Divorced",None,"Unmarried","White","Female",300,200,80,"?"]]
    f = pd.DataFrame(rows,columns=FEATURES,index=[10,11,12,13])
    f["income"] = ["<=50K",">50K","<=50K",">50K"]
    return f


class PipelineTests(unittest.TestCase):
    def test_train_medians_modes_quantiles_match_independent_expected_values(self):
        train = fixture().drop(columns="income")
        train.loc[10,"age"] = np.nan
        cleaner = AdultCleaner().fit(train)
        self.assertEqual(cleaner.numeric_imputer_.statistics_[0],40)
        self.assertEqual(cleaner.categorical_imputer_.statistics_[0],"Private")
        self.assertAlmostEqual(cleaner.lower_[0],30.3)
        self.assertAlmostEqual(cleaner.upper_[0],49.7)
        cleaned = cleaner.transform(train)
        self.assertEqual(cleaned.loc[13,"workclass"],"Private")
        self.assertEqual(cleaned.loc[13,"occupation"],"Sales")
        self.assertFalse(cleaned.isna().any().any())

    def test_holdout_extremes_and_new_categories_do_not_change_training_state(self):
        cfg = yaml.safe_load((Path(__file__).resolve().parents[1]/"config.yaml").read_text())
        train = fixture()[FEATURES]
        pre = build_preprocessor(cfg,True).fit(train)
        before = copy.deepcopy(pre)
        holdout = train.iloc[:1].copy()
        holdout.loc[10,NUMERIC] = [90,99999999,16,99999,4356,99]
        holdout.loc[10,"workclass"] = "New-employer"
        result = pre.transform(holdout)
        self.assertTrue(np.isfinite(result).all())
        for attr in ["lower_","upper_","fit_row_ids_"]:
            np.testing.assert_array_equal(getattr(before.named_steps["clean"],attr),getattr(pre.named_steps["clean"],attr))
        np.testing.assert_array_equal(before.named_steps["encode"].named_transformers_["numeric"].mean_,
                                      pre.named_steps["encode"].named_transformers_["numeric"].mean_)
        np.testing.assert_array_equal(before.transform(train),pre.transform(train))

    def test_validation_reports_bad_values_without_mutating_or_dropping(self):
        f = fixture()
        f.loc[10,"age"] = 16
        f.loc[11,"hours-per-week"] = 100
        f.loc[12,"sex"] = "unknown"
        f["fnlwgt"] = f["fnlwgt"].astype(float)
        f.loc[13,"fnlwgt"] = 2.5
        original = f.copy(deep=True)
        bad, _ = validate(f)
        self.assertEqual(len(bad),4)
        self.assertEqual(set(bad.rule),{"manual_range","unknown_category","integer_type"})
        pd.testing.assert_frame_equal(f,original)

    def test_missing_markers_are_permitted_and_target_missing_is_rejected(self):
        f = fixture()
        bad,missing = validate(f)
        self.assertTrue(bad.empty)
        self.assertEqual(missing.loc[missing.column=="workclass","missing_total"].iloc[0],1)
        f.loc[10,"income"] = None
        bad,_ = validate(f)
        self.assertIn("missing_target",bad.rule.tolist())

    def test_derived_values_and_denominator_guard(self):
        f = fixture()[FEATURES]
        derived = FeatureEngineer().transform(f)
        self.assertEqual(derived.loc[12,"capital_net"],100)
        self.assertEqual(derived.loc[10,"hours_per_age"],5)
        self.assertEqual(derived.loc[11,"hours_per_age"],40/14)
        self.assertNotIn("capital_net",f)
        f.loc[10,"age"] = 16
        with self.assertRaises(ValueError): FeatureEngineer().transform(f)

    def test_duplicate_conflict_audit_includes_missing_features(self):
        f = fixture().iloc[[3]].copy()
        f = pd.concat([f,f,f],ignore_index=True)
        f.loc[2,"income"] = "<=50K"
        audit,rows = conflict_audit(f)
        self.assertEqual(audit["conflicting_feature_groups"],1)
        self.assertEqual(audit["rows_in_conflicting_groups"],3)
        self.assertEqual(audit["exact_duplicate_excess_rows"],1)
        self.assertEqual(len(rows),3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
