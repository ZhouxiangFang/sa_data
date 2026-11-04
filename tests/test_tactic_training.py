import argparse
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import helper


class FakeTokenizer:
    """Formats a prompt as itself and counts characters as tokens."""

    def apply_chat_template(self, conversations, tokenize, add_generation_prompt):
        return [conversation[0]["content"] for conversation in conversations]

    def __call__(self, texts, add_special_tokens, return_length, truncation):
        return {"length": [len(text) for text in texts]}


def gretel_rows():
    rows = []
    for tactic, count in (("quid pro quo", 6), ("Phishing", 4), ("blackmail", 2)):
        for i in range(count):
            rows.append({"prompt": f"{tactic} {i}", "response": "no",
                         "prompt_harm_label": "harmful", "response_harm_label": "safe",
                         "sub_abbrs": ["discrimination" if i % 2 else "malicious_use"],
                         "tactic": tactic})
    rows.append({"prompt": "no tactic", "response": "no", "prompt_harm_label": "harmful",
                 "response_harm_label": "safe", "sub_abbrs": ["discrimination"], "tactic": ""})
    rows.append({"prompt": "unsafe target", "response": "yes", "prompt_harm_label": "harmful",
                 "response_harm_label": "unsafe", "sub_abbrs": ["discrimination"],
                 "tactic": "Phishing"})
    return pd.DataFrame(rows)


class TacticTests(unittest.TestCase):
    def test_pools_apply_training_filters_and_snake_case_names(self):
        names, pools = helper.tactic_pools(gretel_rows(), "gretel")
        self.assertEqual(names, {"quid_pro_quo": "quid pro quo", "phishing": "Phishing",
                                 "blackmail": "blackmail"})
        self.assertEqual({abbr: len(pool) for abbr, pool in pools.items()},
                         {"quid_pro_quo": 6, "phishing": 4, "blackmail": 2})
        self.assertEqual(pools["phishing"]["tactic_abbrs"].iloc[0], ["phishing"])

    def test_baselines_pool_only_tactics_with_enough_rows(self):
        pool = helper.select_tactic(gretel_rows(), "gretel", "random", harmful_count=4)
        self.assertEqual(sorted(pool["tactic"].unique()), ["Phishing", "quid pro quo"])
        with self.assertRaises(ValueError):
            helper.select_tactic(gretel_rows(), "gretel", "unknown_tactic")

    def test_evenly_gives_each_tactic_the_same_share(self):
        df = gretel_rows()
        pool = helper.select_tactic(df, "gretel", "evenly", harmful_count=4)
        args = argparse.Namespace(num_train=4, harmful_rate=1.0, general_rate=0.0,
                                  alignment_dataset="gretel", abbr="evenly",
                                  group_by="tactic", benign_data_type="vanilla_benign",
                                  seed=0, max_length=100)
        train_df, available, _, _ = helper.build_training_data(
            args, FakeTokenizer(), harmful_pool=pool, benign_pool=pd.DataFrame(),
        )
        self.assertEqual(available, 10)
        self.assertEqual(train_df.attrs["harmful_subcategory_counts"],
                         {"phishing": 2, "quid_pro_quo": 2})
        self.assertEqual(train_df.attrs["baseline_groups"], ["phishing", "quid_pro_quo"])


if __name__ == "__main__":
    unittest.main()
