# -*- coding: utf-8 -*-
"""
Demo: chay thu MCTS + logging voi mot "mang no-ron gia lap" (random),
KHONG can TensorFlow/TensorLayer, chi de kiem tra dinh dang log dung nhu mong muon.

Chay thu:
    python demo_log_playout.py

Sau khi chay that voi mang thuc (train.py), chi can thay ham gia policy_value_fn
bang self.policy_value_net.policy_value_fn_random that su la duoc, code MCTS/log
khong doi.
"""
import logging
import numpy as np
from logging_setup import setup_logging
from mcts_alphaZero import MCTS

# Bat DEBUG ca tren console de xem truc tiep chi tiet tung playout trong demo nay.
# Khi train that (nhieu tram nghin van), nen de console_level=INFO (xem train.py)
# va chi doc chi tiet playout trong FILE log khi can debug.
logger = setup_logging(log_dir="logs_demo", level=logging.DEBUG, console_level=logging.DEBUG)


class FakeBoard(object):
    """Ban co toi gian 3x3 chi de demo luong MCTS, khong phai game_board.Board that."""
    def __init__(self, size=3):
        self.width = size
        self.height = size
        self.states = {}
        self.availables = list(range(size * size))
        self.current_player = 1
        self.players = [1, 2]

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = 2 if self.current_player == 1 else 1

    def game_end(self):
        # demo don gian: het nuoc di la hoa, chua bao gio thang som
        if not self.availables:
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


def fake_policy_value_fn(state, action_fc, evaluation_fc):
    """
    Gia lap dau ra mang no-ron: xac suat ngau nhien (chuan hoa tong=1) cho moi nuoc con
    kha thi, va gia tri v ngau nhien trong [-1, 1]. Thay the ham nay bang
    policy_value_net.policy_value_fn_random that khi chay voi mang da huan luyen.
    """
    n = len(state.availables)
    raw = np.random.rand(n)
    probs = raw / raw.sum()
    action_probs = list(zip(state.availables, probs))
    leaf_value = float(np.random.uniform(-1, 1))
    return action_probs, leaf_value


if __name__ == "__main__":
    board = FakeBoard(size=3)
    mcts = MCTS(policy_value_fn=fake_policy_value_fn,
                action_fc=None, evaluation_fc=None,
                is_selfplay=True, c_puct=5, n_playout=5)

    logger.info("\n\n########## DEMO: chay %d playout tren ban co gia lap 3x3 ##########\n", mcts._n_playout)
    acts, visits = mcts.get_move_visits(board)
    logger.info("\nKET QUA CUOI CUNG - visit count tung nuoc di kha thi tai root: %s",
                dict(zip(acts, visits)))
