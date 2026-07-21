# -*- coding: utf-8 -*-
"""
policy_value_net_pytorch.py

Ban thay the cho policy_value_net_tensorlayer.py, dung PyTorch thay vi
TensorFlow 1.x + TensorLayer (vi TF1/TensorLayer 1.10 khong con tuong thich
voi driver/CUDA/cuDNN cua cac GPU doi moi, vd RTX 30/40/50, hay GPU tren
Colab hien nay).

MUC TIEU khi viet lai file nay:
    - Giu NGUYEN kien truc mang (so lop, thu tu lop, kich thuoc filter,
      resnet block, ZeroPad2d(2), 2 nhanh policy/value) giong 100% ban goc.
    - Giu NGUYEN cong thuc loss (value MSE + policy cross-entropy voi
      MCTS visit-prob + L2 penalty 1e-4 tren trong so khong tinh bias),
      cong thuc entropy, va optimizer Adam.
    - Giu NGUYEN "luong hoat dong" (workflow): cung cac ham public
      policy_value / policy_value_fn / policy_value_fn_random / train_step /
      save_model / restore_model / save_numpy / load_numpy, cung tham so
      constructor, de train.py, train_mpi.py, mcts_alphaZero.py,
      human_play.py, human_play_MPI.py hau nhu khong phai sua logic,
      chi doi dong import.

LUU Y QUAN TRONG:
    - Model cu luu bang tf.train.Saver (.index/.data-00000-of-00001) KHONG
      the doc truc tiep bang PyTorch. Ban se can train lai tu dau, hoac
      neu can giu model cu, phai dung mot may/moi truong con TF1 de doc
      checkpoint cu, xuat ra numpy, roi nap vao day bang load_numpy_dict()
      (co san o phia duoi) - nhung mac dinh workflow chinh la train moi.
    - Model moi luu bang torch.save(...) voi duoi ".pt".
      save_model('model/best_policy.model') se tao ra file
      'model/best_policy.model.pt' (tu dong them duoi), va restore_model
      cung tu dong nhan ca hai kieu duong dan (co hoac khong co ".pt").
"""

from __future__ import print_function
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Dinh nghia kien truc mang (giu nguyen so voi ban TensorLayer)
# ---------------------------------------------------------------------------
class _ResidualBlock(nn.Module):
    """
    Mot block resnet don gian, giong ham residual_block() ban goc:
        conv3x3 -> BN -> relu -> conv3x3 -> BN -> (+identity) -> relu
    """
    def __init__(self, channels):
        super(_ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        out = F.relu(out)
        return out


class _Net(nn.Module):
    """
    Kien truc giong ham network() ban goc:
      input (N, planes, H, W)
        -> ZeroPad2d(2)                      # giu nguyen y tuong pad them 2 vien
        -> Conv2d 1x1 -> 64 filters           # conv2d_1
        -> nb_block x ResidualBlock(64)       # resnet
        -> nhanh policy: Conv1x1(2) -> BN -> relu -> flatten -> Linear -> log_softmax
        -> nhanh value : Conv1x1(1) -> BN -> relu -> flatten -> Linear(256) -> relu
                          -> Linear(1) -> tanh
    """
    def __init__(self, board_width, board_height, planes_num, nb_block):
        super(_Net, self).__init__()
        self.board_width = board_width
        self.board_height = board_height

        pad = 2
        padded_h = board_height + 2 * pad
        padded_w = board_width + 2 * pad

        self.zero_pad = nn.ZeroPad2d(pad)
        self.conv1 = nn.Conv2d(planes_num, 64, kernel_size=1, stride=1)

        self.res_blocks = nn.ModuleList([_ResidualBlock(64) for _ in range(nb_block)])

        # --- policy (action) head ---
        self.act_conv = nn.Conv2d(64, 2, kernel_size=1, stride=1)
        self.act_bn = nn.BatchNorm2d(2)
        self.act_fc = nn.Linear(2 * padded_h * padded_w, board_width * board_height)

        # --- value head ---
        self.val_conv = nn.Conv2d(64, 1, kernel_size=1, stride=1)
        self.val_bn = nn.BatchNorm2d(1)
        self.val_fc1 = nn.Linear(1 * padded_h * padded_w, 256)
        self.val_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # x: (N, planes_num, H, W)  -- da la NCHW, khong can transpose nhu TF
        x = self.zero_pad(x)
        x = self.conv1(x)
        for block in self.res_blocks:
            x = block(x)

        # policy head
        act = F.relu(self.act_bn(self.act_conv(x)))
        act = act.reshape(act.size(0), -1)
        act = self.act_fc(act)
        act_log_probs = F.log_softmax(act, dim=1)

        # value head
        val = F.relu(self.val_bn(self.val_conv(x)))
        val = val.reshape(val.size(0), -1)
        val = F.relu(self.val_fc1(val))
        val = torch.tanh(self.val_fc2(val))

        return act_log_probs, val


# ---------------------------------------------------------------------------
# 2. Lop PolicyValueNet, giu nguyen API cua ban TensorLayer
# ---------------------------------------------------------------------------
class PolicyValueNet(object):
    def __init__(self, board_width, board_height, block, init_model=None,
                 transfer_model=None, cuda=False):
        print()
        print('building network (PyTorch) ...')
        print()
        print('torch version:', torch.__version__)
        print('cuda available:', torch.cuda.is_available())
        if torch.cuda.is_available():
            print('gpu:', torch.cuda.get_device_name(0))

        self.planes_num = 9  # feature planes, giong ban goc
        self.nb_block = block
        self.board_width = board_width
        self.board_height = board_height

        # cuda=True nghia la "dung GPU neu co", giong tinh than tham so cu
        # (ban cu dung cuda=False de TAT gpu bang CUDA_VISIBLE_DEVICES=-1)
        if cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print('using device:', self.device)

        # mang chinh
        self.net = _Net(board_width, board_height, self.planes_num, self.nb_block).to(self.device)

        # mang "oppo" (doi thu), dung cho train_mpi.py khi self-play voi
        # phien ban truoc cua chinh minh (giong network_oppo trong ban goc)
        self.net_oppo = _Net(board_width, board_height, self.planes_num, self.nb_block).to(self.device)
        self.net_oppo.load_state_dict(self.net.state_dict())

        # cac "token" thay the cho action_fc/evaluation_fc (truoc la tensor
        # trong TF graph). Gio chi la nhan de biet dung mang nao & mode nao,
        # cac ham policy_value/policy_value_fn nhan token nay va tu quyet
        # dinh goi self.net hay self.net_oppo, train() hay eval().
        self.action_fc_train = ('main', 'train')
        self.action_fc_test = ('main', 'test')
        self.evaluation_fc2_train = ('main', 'train')
        self.evaluation_fc2_test = ('main', 'test')

        self.action_fc_test_oppo = ('oppo', 'test')
        self.evaluation_fc2_test_oppo = ('oppo', 'test')

        # de tuong thich voi save_numpy/load_numpy/train_mpi.py, giu 2 danh
        # sach "params" tro toi state_dict cua 2 mang
        self.network_all_params = self.net
        self.network_oppo_all_params = self.net_oppo

        l2_penalty_beta = 1e-4
        self.l2_penalty_beta = l2_penalty_beta

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        # learning_rate duoc set truc tiep vao optimizer o train_step()

        if init_model is not None:
            self.restore_model(init_model)
            print('model loaded!')
        elif transfer_model is not None:
            self.restore_model(transfer_model, transfer_only=True)
            print('transfer model loaded !')
        else:
            print('can not find saved model, learn from scratch !')

    # ------------------------------------------------------------------
    # cac ham noi bo
    # ------------------------------------------------------------------
    def _select_net(self, token):
        """token la 1 trong cac thuoc tinh action_fc_*/evaluation_fc2_* o tren"""
        which, mode = token
        net = self.net if which == 'main' else self.net_oppo
        return net, mode

    def _forward(self, net, mode, state_batch_np):
        state = torch.as_tensor(np.array(state_batch_np), dtype=torch.float32, device=self.device)
        if mode == 'train':
            net.train()
            log_probs, value = net(state)
        else:
            net.eval()
            with torch.no_grad():
                log_probs, value = net(state)
        return log_probs, value

    # ------------------------------------------------------------------
    # cac ham public - giu nguyen ten & chu ky (signature) nhu ban goc
    # ------------------------------------------------------------------
    def policy_value(self, state_batch, actin_fc, evaluation_fc):
        '''
        input: a batch of states, actin_fc, evaluation_fc (token)
        output: a batch of action probabilities and state values
        '''
        net, mode = self._select_net(actin_fc)
        log_act_probs, value = self._forward(net, mode, state_batch)
        act_probs = np.exp(log_act_probs.detach().cpu().numpy())
        value = value.detach().cpu().numpy()
        return act_probs, value

    def policy_value_fn(self, board, actin_fc, evaluation_fc):
        '''
        input: board, actin_fc, evaluation_fc
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        '''
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
            -1, self.planes_num, self.board_width, self.board_height))
        act_probs, value = self.policy_value(current_state, actin_fc, evaluation_fc)
        act_probs = zip(legal_positions, act_probs[0][legal_positions])
        return act_probs, value

    def policy_value_fn_random(self, board, actin_fc, evaluation_fc):
        '''
        input: board, actin_fc, evaluation_fc
        output: a list of (action, probability) tuples for each available
        action and the score of the board state

        Giong ban goc: ap dung mot phep quay/lat ngau nhien (dihedral)
        truoc khi dua vao mang, roi quay/lat nguoc lai ket qua - dung
        cong thuc (di(p), v) = f_theta(di(sL)) nhu paper AlphaZero.
        '''
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
            -1, self.planes_num, self.board_width, self.board_height))

        rotate_angle = np.random.randint(1, 5)
        flip = np.random.randint(0, 2)
        equi_state = np.array([np.rot90(s, rotate_angle) for s in current_state[0]])
        if flip:
            equi_state = np.array([np.fliplr(s) for s in equi_state])

        act_probs, value = self.policy_value(np.array([equi_state]), actin_fc, evaluation_fc)

        equi_mcts_prob = np.flipud(act_probs[0].reshape(self.board_height, self.board_width))
        if flip:
            equi_mcts_prob = np.fliplr(equi_mcts_prob)
        equi_mcts_prob = np.rot90(equi_mcts_prob, 4 - rotate_angle)
        act_probs = np.flipud(equi_mcts_prob).flatten()

        act_probs = zip(legal_positions, act_probs[legal_positions])
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        '''
        perform a training step, giu nguyen cong thuc loss:
            loss = value_loss (MSE) + policy_loss (cross-entropy voi mcts_probs)
                   + l2_penalty (1e-4 * ||w||^2 / 2, tru bias)
        '''
        self.net.train()
        for g in self.optimizer.param_groups:
            g['lr'] = lr

        state = torch.as_tensor(np.array(state_batch), dtype=torch.float32, device=self.device)
        mcts_probs_t = torch.as_tensor(np.array(mcts_probs), dtype=torch.float32, device=self.device)
        winner_batch_np = np.reshape(np.array(winner_batch, dtype=np.float32), (-1, 1))
        labels = torch.as_tensor(winner_batch_np, dtype=torch.float32, device=self.device)

        self.optimizer.zero_grad()

        log_act_probs, value = self.net(state)

        # 3-1 value loss (MSE)
        value_loss = F.mse_loss(value, labels)
        # 3-2 policy loss
        policy_loss = -torch.mean(torch.sum(mcts_probs_t * log_act_probs, dim=1))
        # 3-3 L2 penalty, tru bias - giong tf.nn.l2_loss (sum(v^2)/2) * beta
        l2_penalty = 0.0
        for name, p in self.net.named_parameters():
            if 'bias' not in name.lower():
                l2_penalty = l2_penalty + 0.5 * torch.sum(p ** 2)
        l2_penalty = self.l2_penalty_beta * l2_penalty

        loss = value_loss + policy_loss + l2_penalty
        loss.backward()
        self.optimizer.step()

        # entropy chi de theo doi, khong dung de train
        with torch.no_grad():
            self.net.eval()
            log_act_probs_eval, _ = self.net(state)
            entropy = -torch.mean(torch.sum(torch.exp(log_act_probs_eval) * log_act_probs_eval, dim=1))
            self.net.train()

        return float(loss.item()), float(entropy.item())

    def save_model(self, model_path):
        '''
        save model. model_path se tu dong duoc them duoi ".pt" neu chua co.
        Chi luu mang chinh (khong luu mang oppo), giong ban goc.
        '''
        path = model_path if model_path.endswith('.pt') else model_path + '.pt'
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        torch.save(self.net.state_dict(), path)

    def restore_model(self, model_path, transfer_only=False):
        '''
        restore model tu file .pt (hoac duong dan khong co duoi, se tu them).
        '''
        path = model_path
        if not os.path.exists(path):
            if os.path.exists(path + '.pt'):
                path = path + '.pt'
        state_dict = torch.load(path, map_location=self.device)
        if transfer_only:
            # chi nap cac layer conv/resnet/bn (giong saver_restore ban goc
            # loc theo ten 'conv2d'/'resnet'/'bn'), bo qua vai layer khong khop
            own_state = self.net.state_dict()
            filtered = {k: v for k, v in state_dict.items()
                        if k in own_state and own_state[k].shape == v.shape}
            own_state.update(filtered)
            self.net.load_state_dict(own_state)
        else:
            self.net.load_state_dict(state_dict)
        self.net_oppo.load_state_dict(self.net.state_dict())

    # ------------------------------------------------------------------
    # tuong thich voi train_mpi.py: save_numpy / load_numpy dung de
    # "chup nhanh" trong so mang chinh, roi nap vao mang oppo lam doi thu
    # tu-choi (self evaluate) - giong het y tuong ban goc.
    # ------------------------------------------------------------------
    def save_numpy(self, params):
        '''
        params: self.network_all_params (chinh la self.net)
        luu state_dict ra numpy .npy (dang dict cac mang numpy)
        '''
        print('saving model as numpy form ...')
        state_dict = {k: v.detach().cpu().numpy() for k, v in params.state_dict().items()}
        os.makedirs('tmp', exist_ok=True)
        np.save('tmp/model.npy', state_dict, allow_pickle=True)

    def load_numpy(self, params, path='tmp/model.npy'):
        '''
        params: self.network_oppo_all_params (chinh la self.net_oppo)
        nap trong so tu numpy .npy vao mang oppo (dung de self-evaluate)
        '''
        print('loading model from numpy form ...')
        state_dict = np.load(path, allow_pickle=True).item()
        state_dict = {k: torch.as_tensor(v, device=self.device) for k, v in state_dict.items()}
        params.load_state_dict(state_dict)
        print('load model from numpy!')

    def sync_oppo_with_main(self):
        '''
        tien ich them: dong bo truc tiep mang oppo = mang chinh (khong qua
        file numpy), dung khi khong can luu file trung gian.
        '''
        self.net_oppo.load_state_dict(self.net.state_dict())

    def print_params(self, params=None):
        # only for debug
        target = self.net if params is None else params
        return {k: v.shape for k, v in target.state_dict().items()}
