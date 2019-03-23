from game_state import GameState
import pytest 


def test_find_box_centers(boxes):
    test_boxes = [{'ymin': 2, 'ymax': 4, 'xmin': 1, 'xmax': 3}]
    box_centers = GameState.find_box_centers(test_boxes)[0]
    ref_box_centers = (3,2)
    assert box_centers == ref_box_centers





