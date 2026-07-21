# -*- coding: utf-8 -*-
"""
GreedyPlayer cho Gomoku.

Nguyen tac: KHONG tim kiem cay (khac han MCTS/AlphaZero).
Voi moi o trong, cham diem theo the co (pattern) ma nuoc di do tao ra:
    - tan cong: neu MINH dat quan vao o do thi tao duoc the co manh co nao
    - phong thu: neu DOI THU dat quan vao o do thi ho se tao duoc the co manh co nao
                 (minh dat truoc de chan)
Chon o co tong diem (tan cong + he_so * phong thu) cao nhat.
Day la player "1 nuoc nhin xa", dung lam baseline yeu de so sanh voi MCTS/AlphaZero.

Tuong thich voi interface cua project (giong MCTSPlayer trong mcts_pure.py /
mcts_alphaZero.py) de co the dung truc tiep trong game_board.Game.start_play():
    - set_player_ind(p)
    - reset_player()
    - get_action(board, is_selfplay=False, print_probs_value=False) -> (move, None)
    - __str__()
"""

from __future__ import print_function
import random

# Thang diem cac the co: (so quan lien tiep, so dau ho) -> diem
SCORE_TABLE = {
    (5, 0): 100000, (5, 1): 100000, (5, 2): 100000,  # >=5 la thang roi
    (4, 2): 10000,   # 4 quan, 2 dau ho -> "song tu" (open four), chac chan thang nuoc sau
    (4, 1): 1000,    # 4 quan, 1 dau ho -> "miên tứ" (bi chan 1 dau), van con doa
    (4, 0): 0,       # 4 quan bi chan 2 dau -> vo dung
    (3, 2): 1000,    # 3 quan, 2 dau ho -> "song tam" (open three), rat nguy hiem
    (3, 1): 100,     # 3 quan, 1 dau ho -> "miên tam", it nguy hiem hon
    (3, 0): 0,
    (2, 2): 100,     # 2 quan, 2 dau ho
    (2, 1): 10,
    (2, 0): 0,
    (1, 2): 1,
    (1, 1): 1,
    (1, 0): 0,
}

DIRECTIONS = [(1, 0), (0, 1), (1, 1), (1, -1)]  # ngang, doc, cheo chinh, cheo phu

# He so uu tien phong thu so voi tan cong (1.0 = ngang nhau).
# De < 1.0 chut de uu tien tan cong khi 2 phuong an ngang diem nhau,
# giong xu huong choi thuc te (tan cong truoc, phong thu sau).
BLOCK_WEIGHT = 0.9


def _count_line(board, move, player, dx, dy):
    """
    Dem so quan lien tiep cua `player` qua diem `move` theo huong (dx,dy),
    coi nhu move da duoc gan cho player (mo phong, khong sua board that).
    Tra ve (count, open_ends): so quan lien tiep va so dau ho (0,1,2).
    """
    width, height = board.width, board.height
    h0, w0 = move // width, move % width
    count = 1

    # huong duong
    h, w = h0 + dx, w0 + dy
    while 0 <= h < height and 0 <= w < width and board.states.get(h * width + w) == player:
        count += 1
        h += dx
        w += dy
    open_pos = (0 <= h < height and 0 <= w < width and (h * width + w) not in board.states)

    # huong nguoc
    h, w = h0 - dx, w0 - dy
    while 0 <= h < height and 0 <= w < width and board.states.get(h * width + w) == player:
        count += 1
        h -= dx
        w -= dy
    open_neg = (0 <= h < height and 0 <= w < width and (h * width + w) not in board.states)

    open_ends = int(open_pos) + int(open_neg)
    return count, open_ends


def _pattern_score(board, move, player):
    """
    Tong diem the co ma `player` tao duoc neu dat quan vao `move`,
    tinh tren ca 4 huong (ngang / doc / 2 duong cheo).
    """
    total = 0
    for dx, dy in DIRECTIONS:
        count, open_ends = _count_line(board, move, player, dx, dy)
        count = min(count, 5)  # >=5 deu la thang, khong can dem qua 5
        total += SCORE_TABLE.get((count, open_ends), 0)
    return total


class GreedyPlayer(object):
    """
    AI choi Gomoku theo kieu greedy: khong tim kiem cay,
    chi cham diem tung nuoc di dua tren the co tao ra (1-ply).
    """

    def __init__(self, block_weight=BLOCK_WEIGHT):
        self.block_weight = block_weight
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        # greedy khong co trang thai/cay can reset, giu de tuong thich interface
        pass

    def get_action(self, board, is_selfplay=False, print_probs_value=False):
        sensible_moves = board.availables
        if not sensible_moves:
            print("WARNING: the board is full")
            return -1, None

        me = self.player
        opponent = board.players[0] if me == board.players[1] else board.players[1]

        # Neu bang trong (nuoc di dau tien), chon o gan trung tam
        if not board.states:
            center = (board.height // 2) * board.width + (board.width // 2)
            return center, None

        best_score = None
        best_moves = []
        for move in sensible_moves:
            attack = _pattern_score(board, move, me)
            defend = _pattern_score(board, move, opponent)
            score = attack + self.block_weight * defend

            if best_score is None or score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

        move = random.choice(best_moves)

        if print_probs_value:
            print("GreedyPlayer chon nuoc {} voi diem {:.1f}".format(
                board.move_to_location(move), best_score))

        return move, None

    def __str__(self):
        return "Greedy {}".format(self.player)
