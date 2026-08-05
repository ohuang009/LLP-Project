from __future__ import annotations

import unittest

from Pipeline.Node_Pipeline import metadata_mentions


class AuthorMetadataTests(unittest.TestCase):
    def test_affiliation_markers_do_not_become_people(self):
        parsed = {
            "document": {
                "id": "doc",
                "filename": "paper.pdf",
                "title": "Paper",
                "authors_text": (
                    "Kehua Chen #, a,e, Hongcheng Wang #, a,b,c, "
                    "Borja Valverde-P \u0301erez d, Siyuan Zhai a, "
                    "Luca Vezzaro d, **, Aijie Wang a,c, *"
                ),
            }
        }
        names = {
            row["canonical_name"]
            for row in metadata_mentions(parsed)
            if row["label"] == "Person"
        }
        self.assertEqual(names, {
            "Kehua Chen", "Hongcheng Wang", "Borja Valverde-Perez",
            "Siyuan Zhai", "Luca Vezzaro", "Aijie Wang",
        })
        rows = metadata_mentions(parsed)
        self.assertTrue(all(row["source"]["sentence_id"] for row in rows))
        self.assertTrue(all(
            row["source"]["sentence_id"] in row["source"]["context_sentence_ids"]
            for row in rows
        ))


if __name__ == "__main__":
    unittest.main()
