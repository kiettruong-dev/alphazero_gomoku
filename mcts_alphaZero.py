# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 22:05:17 2018

@author: initial
"""


import numpy as np
import logging
from logging_setup import get_logger

logger = get_logger()

def softmax(x):
    probs = np.exp(x - np.max(x))
    # https://mp.weixin.qq.com/s/2xYgaeLlmmUfxiHCbCa8dQ
    # avoid float overflow and underflow
    probs /= np.sum(probs)
    return probs

class TreeNode(object):
    '''
    A node in the MCTS tree.
    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    '''

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p # its the prior probability that action's taken to get this node

    def expand(self, action_priors,add_noise):
        '''
        Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        '''
        # when train by self-play, add dirichlet noises in each node

        # should note it's different from paper that only add noises in root node
        # i guess alphago zero discard the whole tree after each move and rebuild a new tree, so it's no conflict
        # while here i contained the Node under the chosen action, it's a little different.
        # there's no idea which is better
        # in addition, the parameters should be tried
        # for 11x11 board,
        # dirichlet parameter :0.3 is ok, should be smaller with a bigger board,such as 20x20 with 0.03
        # weights between priors and noise: 0.75 and 0.25 in paper and i don't change it here,
        # but i think maybe 0.8/0.2 or even 0.9/0.1 is better because i add noise in every node
        # rich people can try some other parameters
        if add_noise:
            action_priors = list(action_priors)
            length = len(action_priors)
            dirichlet_noise = np.random.dirichlet(0.3 * np.ones(length))
            for i in range(length):
                if action_priors[i][0] not in self._children:
                    self._children[action_priors[i][0]] = TreeNode(self,0.75*action_priors[i][1]+0.25*dirichlet_noise[i])
        else:
            for action, prob in action_priors:
                if action not in self._children:
                    self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        '''
        Select action among children that gives maximum action value Q plus bonus u(P).
        Return: A tuple of (action, next_node)
        '''
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        '''
        Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        '''
        self._n_visits += 1
        # update visit count
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits
        # Update Q, a running average of values for all visits.
        # there is just: (v-Q)/(n+1)+Q = (v-Q+(n+1)*Q)/(n+1)=(v+n*Q)/(n+1)

    def update_recursive(self, leaf_value):
        '''
        Like a call to update(), but applied recursively for all ancestors.
        '''
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
            # every step for revursive update,
            # we should change the perspective by the way of taking the negative
        self.update(leaf_value)

    def get_value(self, c_puct):
        '''
        Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        '''
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        '''
        check if leaf node (i.e. no nodes below this have been expanded).
        '''
        return self._children == {}

    def is_root(self):
        '''
        check if it's root node
        '''
        return self._parent is None


class MCTS(object):
    '''
    An implementation of Monte Carlo Tree Search.
    '''
    def __init__(self, policy_value_fn,action_fc,evaluation_fc, is_selfplay,c_puct=5, n_playout=400):
        '''
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        '''
        self._root = TreeNode(None, 1.0)
        # root node do not have parent ,and sure with prior probability 1

        self._policy_value_fn = policy_value_fn
        self._action_fc = action_fc
        self._evaluation_fc = evaluation_fc

        self._c_puct = c_puct
        # it's 5 in paper and don't change here,but maybe a better number exists in gomoku domain
        self._n_playout = n_playout # times of tree search
        self._is_selfplay = is_selfplay
        self._playout_id = 0  # dem so playout da chay, dung de danh so log

    def _playout(self, state):
        '''
        Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.

        TOI UU: KHONG con nhan "state da duoc copy san" nua. Ham nay gio
        thao tac TRUC TIEP tren board that (self._root tuong ung dung
        trang thai hien tai cua `state`), di xuong bang do_move() doc
        theo duong di duoc chon, roi UNDO lai (undo_move()) sau khi
        evaluate + backup xong, de tra board ve dung trang thai ban dau
        truoc khi ham nay tra ve. Nho vay tranh duoc copy.deepcopy() toan
        bo object Board (rat cham o Python) tren MOI playout.
        '''
        self._playout_id += 1
        pid = self._playout_id
        debug_on = logger.isEnabledFor(logging.INFO)
        if debug_on:
            logger.debug("----- Playout #%d bat dau -----", pid)

        node = self._root
        depth = 0

        while(1):
            if node.is_leaf():
                if debug_on:
                    if depth == 0:
                        logger.debug("  [do sau 0] Root chua co con -> dung ngay, se expand root o buoc danh gia ben duoi")
                    else:
                        logger.debug("  [do sau %d] Cham nut la (chua tung mo rong) -> dung selection, chuyen sang Expand/Evaluate", depth)
                break

            # log toan bo cac nhanh con truoc khi chon (P, Q, u, value)
            if debug_on:
                rows = []
                for act, child in node._children.items():
                    u = (self._c_puct * child._P *
                         np.sqrt(node._n_visits) / (1 + child._n_visits))
                    value = child._Q + u
                    rows.append((act, child._P, child._Q, u, value, child._n_visits))
                rows.sort(key=lambda r: r[4], reverse=True)
                logger.debug("  [do sau %d] Node hien tai N=%d, xet %d nhanh con (sap theo value giam dan, toi da 8 dong):",
                             depth, node._n_visits, len(rows))
                for act, P, Q, u, value, n in rows[:8]:
                    logger.debug("      action=%-5s  P=%.4f  Q=%+.4f  u=%.4f  value(PUCT)=%+.4f  N=%d",
                                 act, P, Q, u, value, n)

            # Greedily select next move.
            action, node = node.select(self._c_puct)
            if debug_on:
                logger.debug("  --> CHON action=%s (value cao nhat trong PUCT)", action)
            state.do_move(action)
            depth += 1

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        action_probs, leaf_value = self._policy_value_fn(state,self._action_fc,self._evaluation_fc)
        # Check for end of game.
        end, winner = state.game_end()
        if not end:
            if debug_on:
                action_probs = list(action_probs)
                top5 = sorted(action_probs, key=lambda ap: -ap[1])[:5]
                logger.debug("  Danh gia bang mang no-ron: v=%.4f, top-5 P cua node la: %s",
                             leaf_value, ["a{}:{:.4f}".format(a, p) for a, p in top5])
            node.expand(action_probs,add_noise=self._is_selfplay) #tắt nhiễu
            if debug_on:
                logger.debug("  EXPAND: tao %d nut con moi tai node la (add_noise=%s)",
                             len(node._children), self._is_selfplay)
        else:
            # for end state，return the "true" leaf_value
            if winner == -1:  # tie
                leaf_value = 0.0
                if debug_on:
                    logger.debug("  Node la la KET THUC VAN: hoa, leaf_value=0.0")
            else:
                leaf_value = (
                    1.0 if winner == state.get_current_player() else -1.0
                )
                if debug_on:
                    logger.debug("  Node la la KET THUC VAN: winner=%s, leaf_value(goc nhin hien tai)=%+.1f",
                                 winner, leaf_value)

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)
        if debug_on:
            logger.debug("  BACKUP: lan truyen gia tri %+.4f nguoc len %d nut cha qua duong di",
                         -leaf_value, depth)
        # no rollout here

        # TOI UU: hoan tac (undo) dung "depth" nuoc da di trong vong while
        # o tren, de tra `state` (board that, dung chung cho ca n_playout
        # lan goi _playout) ve DUNG trang thai ban dau truoc khi ham nay
        # duoc goi, san sang cho playout tiep theo - khong can deepcopy.
        for _ in range(depth):
            state.undo_move()

    def get_move_visits(self, state):
        '''
        Run all playouts sequentially and return the available actions and
        their corresponding visiting times.
        state: the current game state

        TOI UU: khong con copy.deepcopy(state) moi playout. _playout()
        gio tu do_move()/undo_move() tren chinh `state` truyen vao, va
        LUON tra board ve dung trang thai ban dau sau moi lan goi, nen
        goi lien tiep n_playout lan tren CUNG 1 object la an toan.
        '''
        self._playout_id = 0  # reset so playout ve 0 cho nuoc di moi
        for n in range(self._n_playout):
            # print('playout:',n)
            self._playout(state)

        # calc the visit counts at the root node
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)

        if logger.isEnabledFor(logging.DEBUG):
            summary = sorted(zip(acts, visits), key=lambda x: -x[1])[:10]
            total = sum(visits)
            logger.debug("===== Xong %d playout. Phan bo visit count tai root (top 10) =====", self._n_playout)
            for act, v in summary:
                logger.debug("    action=%-5s  visits=%-5d  ty le=%.3f", act, v, v/total)

        return acts, visits

    def update_with_move(self, last_move):
        '''
        Step forward in the tree, keeping everything we already know
        about the subtree.
        '''
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"

class MCTSPlayer(object):
    '''
    AI player based on MCTS
    '''
    def __init__(self, policy_value_function,action_fc,evaluation_fc,c_puct=5, n_playout=400, is_selfplay=0):
        '''
        init some parameters
        '''
        self._is_selfplay = is_selfplay
        self.policy_value_function = policy_value_function
        self.action_fc = action_fc
        self.evaluation_fc = evaluation_fc
        self.first_n_moves = 12
        # For the first n moves of each game, the temperature is set to τ = 1,
        # For the remainder of the game, an infinitesimal temperature is used, τ→ 0.
        # in paper n=30, here i choose 12 for 11x11, entirely by feel
        self.mcts = MCTS(policy_value_fn = policy_value_function,
                         action_fc = action_fc,
                         evaluation_fc = evaluation_fc,
                         is_selfplay = self._is_selfplay,
                         c_puct = c_puct,
                         n_playout = n_playout)

    def set_player_ind(self, p):
        '''
        set player index
        '''
        self.player = p

    def reset_player(self):
        '''
        reset player
        '''
        self.mcts.update_with_move(-1)

    def get_action(self,board,is_selfplay,print_probs_value):
        '''
        get an action by mcts
        do not discard all the tree and retain the useful part
        '''
        sensible_moves = board.availables
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(board.width * board.height)
        if len(sensible_moves) > 0:
            if is_selfplay:
                move_number = board.width * board.height - len(board.availables)
                acts, visits = self.mcts.get_move_visits(board)
                if move_number <= self.first_n_moves:
                    # For the first n moves of each game, the temperature is set to τ = 1
                    temp = 1
                    probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
                    move = np.random.choice(acts, p=probs)
                else:
                    # For the remainder of the game, an infinitesimal temperature is used, τ→ 0
                    temp = 1e-3
                    probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
                    move = np.random.choice(acts, p=probs)

                logger.info("Nuoc thu %d (tu choi): chon action=%s, temp=%.4f, N(action)=%d",
                            move_number + 1, move, temp, visits[list(acts).index(move)])

                self.mcts.update_with_move(move)
                # update the tree with self move
            else:
                self.mcts.update_with_move(board.last_move)
                # update the tree with opponent's move and then do mcts from the new node

                acts, visits = self.mcts.get_move_visits(board)
                temp = 1e-3
                # always choose the most visited move
                probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
                move = np.random.choice(acts, p=probs)

                self.mcts.update_with_move(move)
                # update the tree with self move

            p = softmax(1.0 / 1.0 * np.log(np.array(visits) + 1e-10))
            move_probs[list(acts)] = p
            # return the prob with temp=1

            if print_probs_value and move_probs is not None:
                act_probs, value = self.policy_value_function(board,self.action_fc,self.evaluation_fc)
                print('-' * 10)
                print('value',value)
                # print the probability of each move
                probs = np.array(move_probs).reshape((board.width, board.height)).round(3)[::-1, :]
                for p in probs:
                    for x in p:
                        print("{0:6}".format(x), end='')
                    print('\r')

            return move,move_probs

        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "Alpha {}".format(self.player)


