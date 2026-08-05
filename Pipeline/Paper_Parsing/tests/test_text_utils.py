import unittest

from Pipeline.Paper_Parsing.text_utils import (
    has_terminal_punctuation, join_lines, looks_like_formula, looks_like_heading,
    normalize_text, split_sentences,
)
from Pipeline.Paper_Parsing.pipeline import (
    _apply_section_hierarchy, _back_matter_label, _clean_authors_text,
    _front_matter, _is_back_matter_heading, _is_introduction_heading,
    _is_narrative_text, _paragraph_groups,
)
from Pipeline.Paper_Parsing.pdf_parser import Line, _infer_ruled_table_bboxes, _words_to_lines
from Pipeline.Paper_Parsing.profiles import (
    ACS_EST, CONFERENCE_PREPRINT, FRONTIERS, GENERIC, IEEE_ARXIV,
    WATER_RESEARCH, detect_profile, is_profile_heading, visible_heading,
)
from types import SimpleNamespace


class TextUtilitiesTest(unittest.TestCase):
    def test_dehyphenates_line_break(self):
        self.assertEqual(join_lines(["environ-", "mental change"]), "environmental change")

    def test_preserves_real_compound_at_line_break(self):
        self.assertEqual(join_lines(["net-", "zero target"]), "net-zero target")

    def test_repairs_short_hyphenated_line_fragment(self):
        self.assertEqual(join_lines(["per-", "formance improved."]), "performance improved.")

    def test_repairs_unicode_hyphenated_line_fragment(self):
        self.assertEqual(join_lines(["environ\u2010", "mental change."]), "environmental change.")

    def test_normalizes_grammatical_seams(self):
        self.assertEqual(
            normalize_text("The model converged. where pressure i . was measured"),
            "The model converged. Where pressure i. was measured",
        )

    def test_normalizes_unicode_compound_hyphen(self):
        self.assertEqual(join_lines(["self ‐ evolution continues."]), "self-evolution continues.")

    def test_preserves_scientific_abbreviation(self):
        result = split_sentences("Sites, e.g. wetlands, were sampled. Results improved.")
        self.assertEqual(len(result), 2)

    def test_preserves_abbreviation_capitalization(self):
        result = split_sentences("Results are presented in Fig. 2. Performance improved.")
        self.assertEqual(result[0], "Results are presented in Fig. 2.")

    def test_incomplete_fragment_is_not_a_sentence(self):
        self.assertEqual(split_sentences("emissions across the supply chain"), [])

    def test_layout_change_does_not_cut_unfinished_sentence(self):
        lines = [
            Line("Environmental assess-", 1, 10, 20, 50, 210, 10),
            Line("ments quantify impacts across", 1, 22, 32, 320, 540, 10),
            Line("the full product lifecycle.", 1, 34, 44, 320, 515, 10),
            Line("A new paragraph begins here.", 1, 55, 65, 50, 270, 10),
        ]
        groups = [group for group in _paragraph_groups(lines, 10, ACS_EST) if group[1]]
        self.assertEqual(len(groups), 2)
        first = join_lines([line.text for line in groups[0][1]])
        self.assertEqual(
            first,
            "Environmental assessments quantify impacts across the full product lifecycle.",
        )

    def test_citation_after_period_is_terminal(self):
        self.assertTrue(has_terminal_punctuation("Prior work established this. 1-3"))

    def test_citation_fragment_remains_in_sentence_coverage(self):
        text = "Prior research documented pesticide exposure in bees. 47,48"
        self.assertEqual(split_sentences(text), [text])

    def test_numbered_prose_is_not_a_heading(self):
        self.assertFalse(looks_like_heading("1. Development of a hybrid model.", 10, 10))

    def test_styled_numbered_section_is_a_heading(self):
        self.assertTrue(looks_like_heading("2.1. Methodology", 10, 10, bold=True))

    def test_plain_numbered_section_is_a_heading(self):
        self.assertTrue(looks_like_heading("2. Methodology", 10, 10))

    def test_numbered_prose_and_chart_labels_are_not_headings(self):
        self.assertFalse(looks_like_heading("30 benchmark covers 30 algorithms spanning sorting and searching", 10, 10))
        self.assertFalse(looks_like_heading("3 eff", 8, 12))
        self.assertFalse(looks_like_heading("1 Hz or greater, whereas data sets are sampled", 10, 10))
        self.assertFalse(looks_like_heading("1. Finding a valid path from source to sink", 10, 10))

    def test_equation_fragment_is_not_narrative(self):
        self.assertFalse(_is_narrative_text("Y i Y i"))

    def test_cid_formula_encoding_is_detected(self):
        self.assertTrue(looks_like_formula("δ t (cid:0) t (cid:0) 1 = 0"))

    def test_short_scientific_sentence_is_narrative(self):
        self.assertTrue(_is_narrative_text("Nutrient loads declined across monitored sites."))

    def test_detects_acs_est_profile(self):
        lines = [SimpleNamespace(text="Environmental Science & Technology")]
        self.assertEqual(detect_profile(lines), ACS_EST)

    def test_detects_water_research_profile(self):
        lines = [SimpleNamespace(text="Water Research 303 (2026)")]
        self.assertEqual(detect_profile(lines), WATER_RESEARCH)

    def test_detects_frontiers_profile(self):
        lines = [SimpleNamespace(text="DOI 10.3389/frwa.2023.1199632")]
        self.assertEqual(detect_profile(lines), FRONTIERS)

    def test_detects_ieee_arxiv_profile_and_repairs_small_caps(self):
        lines = [SimpleNamespace(text="I. I NTRODUCTION")]
        self.assertEqual(detect_profile(lines), IEEE_ARXIV)
        self.assertEqual(visible_heading("II. R ELATED W ORK"), "II. RELATED WORK")

    def test_detects_all_caps_conference_preprint_profile(self):
        lines = [
            SimpleNamespace(text="ABSTRACT"),
            SimpleNamespace(text="INTRODUCTION"),
            SimpleNamespace(text="EXPERIMENTAL SETUP"),
        ]
        self.assertEqual(detect_profile(lines), CONFERENCE_PREPRINT)

    def test_recognizes_numbered_introduction_heading(self):
        self.assertTrue(_is_introduction_heading("1. INTRODUCTION"))
        self.assertTrue(_is_introduction_heading("I. I NTRODUCTION"))
        self.assertTrue(_is_introduction_heading("INTRODUCTION"))
        self.assertFalse(_is_introduction_heading("2. MATERIALS AND METHODS"))

    def test_numbered_references_heading_is_back_matter(self):
        self.assertEqual(_back_matter_label("8. References"), "references")

    def test_reference_table_header_before_midpoint_is_not_back_matter(self):
        line = Line("Reference", 14, 100, 112, 420, 490, 10, bold=True)
        self.assertFalse(_is_back_matter_heading(line, GENERIC, 34, 10))

    def test_late_bold_references_is_back_matter(self):
        line = Line("References", 29, 100, 112, 60, 150, 10, bold=True)
        self.assertTrue(_is_back_matter_heading(line, GENERIC, 34, 10))

    def test_column_gap_splits_words_before_line_interleaving(self):
        words = [
            {"text": "left", "x0": 40, "x1": 72, "top": 100, "bottom": 110, "size": 10, "fontname": "Regular"},
            {"text": "text", "x0": 75, "x1": 100, "top": 100, "bottom": 110, "size": 10, "fontname": "Regular"},
            {"text": "right", "x0": 112, "x1": 145, "top": 100, "bottom": 110, "size": 10, "fontname": "Regular"},
            {"text": "column", "x0": 148, "x1": 190, "top": 100, "bottom": 110, "size": 10, "fontname": "Regular"},
        ]
        self.assertEqual([line.text for line in _words_to_lines(words, 1)], ["left text", "right column"])

    def test_cleans_preprint_author_markers(self):
        value = "Yinon Goldshtein 1, �, Gal Perelman 2, �,*, Assaf Schuster 3, and Avi Ostfeld 4"
        self.assertEqual(
            _clean_authors_text(value),
            "Yinon Goldshtein, Gal Perelman, Assaf Schuster, and Avi Ostfeld",
        )
        self.assertEqual(_clean_authors_text("Zepeng Zhang, Olga Fink a"), "Zepeng Zhang, Olga Fink")

    def test_bold_preprint_front_matter_fallback_and_reference_stop(self):
        lines = [
            Line("Large Language Models for Water Distribution Systems Modeling and Decision-", 1, 70, 82, 90, 520, 12, bold=True),
            Line("Making", 1, 91, 103, 275, 340, 12, bold=True),
            Line("Yinon Goldshtein 1, Gal Perelman 2, Assaf Schuster 3, and Avi Ostfeld 4", 1, 130, 142, 90, 520, 12),
            Line("Abstract", 1, 420, 432, 90, 160, 12, bold=True),
            Line("This study introduces a framework for water distribution modeling.", 1, 450, 462, 90, 520, 12),
            Line("1. Introduction", 2, 70, 82, 90, 220, 12, bold=True),
            Line("The framework supports hydraulic simulation.", 2, 100, 112, 90, 520, 12),
            Line("8. References", 3, 70, 82, 90, 220, 12, bold=True),
            Line("Thomas, A. A cited work.", 3, 100, 112, 90, 520, 12),
        ]
        metadata, content = _front_matter(lines, 12, GENERIC)
        self.assertEqual(metadata["title"], "Large Language Models for Water Distribution Systems Modeling and Decision-Making")
        self.assertIn("Yinon Goldshtein", metadata["authors_text"])
        self.assertFalse(any("Thomas" in line.text for line in content))

    def test_wrapped_smaller_font_author_line_is_retained(self):
        lines = [
            Line("Optimal control towards sustainable wastewater treatment plants", 1, 100, 117, 80, 520, 17.2),
            Line("based on multi-agent reinforcement learning", 1, 120, 137, 80, 520, 17.2),
            Line("Kehua Chen, Hongcheng Wang, Borja Valverde-Perez, Siyuan Zhai, Luca", 1, 166, 179, 80, 520, 12),
            Line("Vezzaro, Aijie Wang", 1, 181, 194, 80, 270, 10),
            Line("Key Laboratory of Environmental Biotechnology, Research Center", 1, 205, 216, 80, 520, 10),
            Line("Abstract", 1, 368, 380, 80, 160, 12, bold=True),
            Line("This study used multi-agent reinforcement learning for optimal control.", 1, 395, 407, 80, 520, 12),
        ]
        metadata, _ = _front_matter(lines, 12, GENERIC)
        self.assertIn("Luca Vezzaro", metadata["authors_text"])
        self.assertIn("Aijie Wang", metadata["authors_text"])
        self.assertNotIn("Key Laboratory", metadata["authors_text"])
    def test_same_baseline_author_fragments_are_joined(self):
        lines = [
            Line("Leveraging large language models for automating water distribution", 1, 169, 183, 80, 520, 13.4),
            Line("network optimization", 1, 187, 200, 80, 260, 13.4),
            Line("Jian Wang", 1, 213, 224, 80, 150, 10.6),
            Line(", Guangtao Fu", 1, 213, 224, 170, 280, 10.6),
            Line(", Dragan Savic", 1, 213, 224, 300, 410, 10.6),
            Line("A B S T R A C T", 1, 273, 280, 80, 200, 7.0),
            Line("This study proposes a multi-agent framework for water networks.", 1, 293, 305, 80, 520, 7.2),
        ]
        metadata, _ = _front_matter(lines, 7.2, GENERIC)
        self.assertIn("Jian Wang", metadata["authors_text"])
        self.assertIn("Guangtao Fu", metadata["authors_text"])
        self.assertIn("Dragan Savic", metadata["authors_text"])

    def test_affiliation_lines_are_not_appended_to_authors(self):
        lines = [
            Line("Scalable and Robust Physics-Informed Graph", 1, 52, 76, 80, 520, 23.9),
            Line("Neural Networks for Water Distribution Systems", 1, 80, 104, 80, 520, 23.9),
            Line("Inaam Ashraf, Andre Artelt, Barbara Hammer", 1, 122, 133, 80, 520, 11),
            Line("Center for Cognitive Interaction Technology", 1, 135, 145, 80, 420, 10),
            Line("Bielefeld University", 1, 147, 157, 80, 260, 10),
            Line("Abstract", 1, 207, 216, 80, 160, 9, bold=True),
            Line("We propose a model for hydraulic state estimation.", 1, 217, 229, 80, 520, 9),
        ]
        metadata, _ = _front_matter(lines, 9, GENERIC)
        self.assertIn("Barbara Hammer", metadata["authors_text"])
        self.assertNotIn("Center", metadata["authors_text"])
        self.assertNotIn("University", metadata["authors_text"])

    def test_first_page_introduction_is_retained_after_abstract(self):
        lines = [
            Line("A robust parser for environmental research papers", 1, 40, 60, 60, 540, 18),
            Line("Jane Example and John Sample", 1, 70, 82, 60, 300, 10),
            Line("Abstract", 1, 110, 122, 310, 380, 10, bold=True),
            Line("This study improves extraction from environmental research papers.", 1, 130, 142, 310, 550, 10),
            Line("1. Introduction", 1, 220, 232, 310, 410, 10, bold=True),
            Line("Environmental evidence is often distributed across complex layouts.", 1, 245, 257, 310, 550, 10),
            Line("2. Methods", 2, 70, 82, 60, 150, 10, bold=True),
            Line("We evaluated several representative formats.", 2, 95, 107, 60, 400, 10),
        ]
        _, content = _front_matter(lines, 10, WATER_RESEARCH)
        self.assertTrue(any(line.text == "1. Introduction" for line in content))
        self.assertTrue(any("environmental evidence" in line.text.lower() for line in content))

    def test_unlabelled_water_research_introduction_is_synthesized(self):
        lines = [
            Line("Environmental catalyst discovery with language models", 1, 40, 60, 40, 540, 18),
            Line("Jane Example and John Sample", 1, 75, 87, 60, 300, 10),
            Line("A B S T R A C T", 1, 120, 132, 200, 310, 8),
            Line("The abstract describes the environmental catalyst framework.", 1, 145, 157, 200, 550, 8),
            Line("It reports accurate recommendations for wastewater treatment.", 1, 160, 172, 200, 550, 8),
            Line("Global water pollution motivates improved catalyst design.", 1, 205, 217, 40, 280, 8),
            Line("Existing approaches rely on fragmented empirical evidence.", 1, 205, 217, 305, 550, 8),
            Line("1. Materials and methods", 2, 70, 82, 40, 230, 8, bold=True),
        ]
        _, content = _front_matter(lines, 8, WATER_RESEARCH)
        self.assertTrue(any(line.text == "Introduction" for line in content))
        abstract_marker = next(i for i, line in enumerate(content) if line.text == "Abstract")
        intro_marker = next(i for i, line in enumerate(content) if line.text == "Introduction")
        self.assertLess(abstract_marker, intro_marker)

    def test_centered_abstract_body_is_retained(self):
        lines = [
            Line("Groundwater velocity forecasting with neural networks", 1, 40, 60, 80, 520, 18),
            Line("Martha Example and Sam Researcher", 1, 75, 87, 120, 440, 10),
            Line("Abstract", 1, 120, 132, 270, 340, 10, bold=True),
            Line("This centered abstract body begins left of its heading and must remain.", 1, 145, 157, 80, 520, 10),
            Line("1 Introduction", 2, 70, 82, 80, 210, 10, bold=True),
            Line("The study evaluates online treatment forecasting.", 2, 100, 112, 80, 500, 10),
        ]
        _, content = _front_matter(lines, 10, GENERIC)
        self.assertTrue(any("centered abstract body" in line.text for line in content))

    def test_frontiers_unlabelled_abstract_is_synthesized(self):
        lines = [
            Line("Machine learning for bacteriological forecasting", 1, 40, 60, 70, 530, 18),
            Line("Grigorios Kyritsakas, Joby Boxall and Vanessa Speight", 1, 75, 87, 70, 520, 10),
            Line("Department of Civil Engineering", 1, 92, 104, 70, 360, 9),
            Line("Forecasting microbial water quality supports operational decisions.", 1, 150, 162, 300, 550, 10),
            Line("INTRODUCTION", 1, 240, 252, 300, 410, 10, bold=True),
            Line("Drinking water safety requires reliable forecasts.", 1, 270, 282, 300, 550, 10),
        ]
        _, content = _front_matter(lines, 10, FRONTIERS)
        self.assertEqual(content[0].text, "Abstract")
        self.assertTrue(any("microbial water quality" in line.text for line in content))

    def test_wrapped_styled_heading_is_joined(self):
        lines = [
            Line("EXPERIMENTAL", 2, 70, 84, 60, 170, 12, bold=True),
            Line("SETUP", 2, 85, 99, 60, 120, 12, bold=True),
            Line("The experiments use several environmental datasets.", 2, 115, 127, 60, 500, 10),
        ]
        groups = _paragraph_groups(lines, 10, CONFERENCE_PREPRINT)
        self.assertEqual(groups[0][0], "EXPERIMENTAL SETUP")

    def test_bare_number_is_not_a_profile_heading(self):
        self.assertFalse(is_profile_heading("3.3", CONFERENCE_PREPRINT, size=12, body_size=10))

    def test_body_line_with_changed_style_does_not_extend_heading(self):
        lines = [
            Line("2.1. Data Collection", 2, 70, 82, 60, 220, 10, bold=True),
            Line("Acute toxicity data were collected from public sources.", 2, 84, 96, 60, 500, 10),
        ]
        groups = _paragraph_groups(lines, 10, ACS_EST)
        self.assertEqual(groups[0][0], "2.1. Data Collection")

    def test_acs_inline_subsection_body_is_split_and_retained(self):
        lines = [
            Line(
                "2.3.1. Class-Balanced Weight (Pos_Weight). A multiplicative weight was applied to the positive class.",
                3, 70, 82, 60, 520, 10, bold=True,
            ),
        ]
        groups = _paragraph_groups(lines, 10, ACS_EST)
        self.assertEqual(groups[0][0], "2.3.1. Class-Balanced Weight (Pos_Weight)")
        body_groups = [group for group in groups if group[1]]
        self.assertEqual(body_groups[0][1][0].text, "A multiplicative weight was applied to the positive class.")

    def test_author_selection_prefers_names_over_email_and_banner(self):
        lines = [
            Line("Scalable environmental graph learning", 1, 50, 70, 70, 530, 18),
            Line("Published as part of Environmental Science & Technology special issue", 1, 90, 102, 60, 540, 12, italic=True),
            Line("Tackling Complex Environmental Problems Holistically", 1, 104, 116, 60, 470, 12, italic=True),
            Line("Inaam Ashraf, Andre Artelt, Barbara Hammer", 1, 125, 137, 150, 460, 11),
            Line("{ mashraf, aartelt, bhammer } @example.edu", 1, 145, 157, 150, 460, 10),
            Line("Abstract", 1, 180, 192, 70, 140, 10),
        ]
        metadata, _ = _front_matter(lines, 10, GENERIC)
        self.assertEqual(metadata["authors_text"], "Inaam Ashraf, Andre Artelt, Barbara Hammer")

    def test_author_selection_joins_columns_and_wrapped_surnames(self):
        lines = [
            Line("Optimizing irrigation efficiency using deep reinforcement learning", 1, 50, 70, 70, 530, 18),
            Line("Xianzhong Ding", 1, 90, 102, 130, 240, 12),
            Line("Wan Du", 1, 90, 102, 390, 460, 12),
            Line("ABSTRACT", 1, 135, 147, 70, 150, 11),
        ]
        metadata, _ = _front_matter(lines, 10, CONFERENCE_PREPRINT)
        self.assertEqual(metadata["authors_text"], "Xianzhong Ding, Wan Du")

        wrapped = [
            Line("Online prediction for environmental water treatment", 1, 50, 70, 70, 530, 18),
            Line("Muhammad Janjua, Haseeb Shah, Martha", 1, 90, 102, 80, 390, 12),
            Line("1,2", 1, 104, 111, 100, 130, 8),
            Line("White", 1, 106, 118, 75, 120, 12),
            Line(", Erfan Miahi, Marlos Machado", 1, 106, 118, 125, 330, 12),
            Line("and Adam", 1, 106, 118, 335, 410, 12),
            Line("White", 1, 122, 134, 195, 245, 12),
            Line("Department of Computing Science", 1, 150, 162, 70, 350, 10),
            Line("Abstract", 1, 190, 202, 200, 270, 10),
        ]
        metadata, _ = _front_matter(wrapped, 10, GENERIC)
        self.assertIn("Martha White", metadata["authors_text"])
        self.assertIn("Adam White", metadata["authors_text"])

    def test_author_selection_separates_full_names_on_wrapped_lines(self):
        lines = [
            Line("Environmental multi-agent systems for water management", 1, 50, 70, 40, 550, 18),
            Line("Fangkun Lin, Zhipeng Luo, Binyu Ma, Xin Yu", 1, 90, 102, 40, 500, 11),
            Line("Boyan Xu", 1, 104, 116, 40, 120, 11),
            Line(", Huabin Zeng", 1, 104, 116, 125, 240, 11),
            Line("Abstract", 1, 150, 162, 200, 280, 10),
        ]
        metadata, _ = _front_matter(lines, 10, WATER_RESEARCH)
        self.assertIn("Xin Yu, Boyan Xu, Huabin Zeng", metadata["authors_text"])


    def test_infers_ruled_column_table(self):
        page = SimpleNamespace(
            width=600,
            height=800,
            edges=[
                {"orientation": "h", "x0": 40, "x1": 290, "top": 680},
                {"orientation": "h", "x0": 40, "x1": 290, "top": 741},
            ],
        )
        self.assertEqual(_infer_ruled_table_bboxes(page), [(40.0, 646.0, 290.0, 744.0)])

    def test_builds_section_hierarchy_and_inherits_pages(self):
        sections = [
            {"id": "s2", "title": "2. Methodology", "pages": [], "paragraphs": []},
            {"id": "s21", "title": "2.1. Methods", "pages": [3], "paragraphs": []},
            {"id": "s211", "title": "2.1.1. Sampling", "pages": [4], "paragraphs": []},
        ]
        _apply_section_hierarchy(sections)
        self.assertEqual(sections[1]["parent_section_id"], "s2")
        self.assertEqual(sections[2]["parent_section_id"], "s21")
        self.assertEqual(sections[0]["pages"], [3, 4])


if __name__ == "__main__":
    unittest.main()
