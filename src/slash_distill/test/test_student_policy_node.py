from slash_distill.nodes.student_policy_node import _normalize_scan_topics


def test_normalize_scan_topics_wraps_single_topic_string():
    assert _normalize_scan_topics("/scan_3d") == ["/scan_3d"]


def test_normalize_scan_topics_preserves_topic_list():
    assert _normalize_scan_topics(
        ["/perception/scan_layer_low", "/perception/scan_layer_body"]
    ) == ["/perception/scan_layer_low", "/perception/scan_layer_body"]
