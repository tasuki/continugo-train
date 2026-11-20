"""Numbered feature contracts. Schema 5 is recovered from Git commit 71f8daf.

Extraction/inference may select a supported schema; fresh training defaults to 7.
Never infer schema semantics from tensor dimensions alone.
"""

SCHEMA4 = 4
SCHEMA5 = 5
SCHEMA6 = 6
SCHEMA7 = 7
SCHEMA4_COARSE_PLANES = (
    "friendly_stones",
    "enemy_stones",
    "friendly_group_size",
    "enemy_group_size",
    "friendly_liberty_area_stones",
    "enemy_liberty_area_stones",
    "friendly_capturable_in_one",
    "enemy_capturable_in_one",
    "left_border",
    "right_border",
    "top_border",
    "bottom_border",
)
SCHEMA5_COARSE_PLANES = (
    "friendly_stones",
    "enemy_stones",
    "friendly_group_size",
    "enemy_group_size",
    "friendly_liberty_centers",
    "enemy_liberty_centers",
    "friendly_liberty_footprints",
    "enemy_liberty_footprints",
    "left_border",
    "right_border",
    "top_border",
    "bottom_border",
)
SCHEMA6_COARSE_PLANES = (
    *SCHEMA5_COARSE_PLANES[:4],
    "friendly_group_liberty_area",
    "enemy_group_liberty_area",
    *SCHEMA5_COARSE_PLANES[4:],
)
SCHEMA7_COARSE_PLANES = (
    *SCHEMA6_COARSE_PLANES[:2],
    "enemy_move_1_ago_coverage",
    "friendly_move_2_ago_coverage",
    "enemy_move_3_ago_coverage",
    *SCHEMA6_COARSE_PLANES[2:],
)
SCHEMA4_MOVE_FEATURES = (
    "legal",
    "connected_groups",
    "connected_stones",
    "captured_groups",
    "captured_stones",
    "resulting_group_size",
    "left_border",
    "right_border",
    "top_border",
    "bottom_border",
)
SCHEMA5_MOVE_FEATURES = (
    *SCHEMA4_MOVE_FEATURES[:6],
    "splits_friendly_region",
    *SCHEMA4_MOVE_FEATURES[6:],
)
SCHEMA6_MOVE_FEATURES = (
    *SCHEMA4_MOVE_FEATURES[1:5],
    "splits_friendly_region",
    *SCHEMA4_MOVE_FEATURES[6:],
)
SCHEMA7_MOVE_FEATURES = SCHEMA6_MOVE_FEATURES
SCHEMA7_GLOBAL_FEATURES = ("coarse_width_div100", "komi_self_div_board_area")
SUPPORTED_SCHEMAS = (SCHEMA4, SCHEMA5, SCHEMA6, SCHEMA7)


def coarse_planes(schema=SCHEMA7):
    try:
        return {
            SCHEMA4: SCHEMA4_COARSE_PLANES,
            SCHEMA5: SCHEMA5_COARSE_PLANES,
            SCHEMA6: SCHEMA6_COARSE_PLANES,
            SCHEMA7: SCHEMA7_COARSE_PLANES,
        }[schema]
    except KeyError:
        raise ValueError(f"unsupported feature schema {schema}") from None


def move_features(schema=SCHEMA7):
    try:
        return {
            SCHEMA4: SCHEMA4_MOVE_FEATURES,
            SCHEMA5: SCHEMA5_MOVE_FEATURES,
            SCHEMA6: SCHEMA6_MOVE_FEATURES,
            SCHEMA7: SCHEMA7_MOVE_FEATURES,
        }[schema]
    except KeyError:
        raise ValueError(f"unsupported feature schema {schema}") from None
