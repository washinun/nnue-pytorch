import chess
import ranger
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import sys
import math
from muon import SingleDeviceMuonWithAuxAdam
import torch.optim.lr_scheduler
torch.set_float32_matmul_precision('high')

# 3 layer fully connected network
L1 = 512
L2 = 8
L3 = 64

class NNUE(pl.LightningModule):
  """
  This model attempts to directly represent the nodchip Stockfish trainer methodology.

  lambda_ = 0.0 - purely based on game results
  lambda_ = 1.0 - purely based on search scores

  It is not ideal for training a Pytorch quantized model directly.
  """
  def __init__(
      self, feature_set, lambda_=[1.0], lr=[1.0],
      label_smoothing_eps=0.0, score_scaling=361,
      momentum=0.0, ply_begin_threshold=100.0, ply_end_threshold=120.0):
    super(NNUE, self).__init__()
    self.input = nn.Linear(feature_set.num_features, L1)
    self.feature_set = feature_set
    self.l1 = nn.Linear(2 * L1, L2)
    self.l2 = nn.Linear(L2, L3)
    self.output = nn.Linear(L3, 1)
    self.lambda_ = lambda_
    self.lr = lr
    self.label_smoothing_eps = label_smoothing_eps
    self.score_scaling = score_scaling
    self.parameter_index = 0
    self.momentum = momentum
    self.ply_begin_threshold = ply_begin_threshold
    self.ply_end_threshold = ply_end_threshold
    
    # For tracking validation loss
    self.current_val_loss = None

    self._zero_virtual_feature_weights()


  '''
  We zero all virtual feature weights because during serialization to .nnue
  we compute weights for each real feature as being the sum of the weights for
  the real feature in question and the virtual features it can be factored to.
  This means that if we didn't initialize the virtual feature weights to zero
  we would end up with the real features having effectively unexpected values
  at initialization - following the bell curve based on how many factors there are.
  '''
  def _zero_virtual_feature_weights(self):
    weights = self.input.weight
    with torch.no_grad():
      for a, b in self.feature_set.get_virtual_feature_ranges():
        weights[:, a:b] = 0.0
    self.input.weight = nn.Parameter(weights)

  '''
  This method attempts to convert the model from using the self.feature_set
  to new_feature_set.
  '''
  def set_feature_set(self, new_feature_set):
    if self.feature_set.name == new_feature_set.name:
      return

    # TODO: Implement this for more complicated conversions.
    #       Currently we support only a single feature block.
    if len(self.feature_set.features) > 1:
      raise Exception('Cannot change feature set from {} to {}.'.format(self.feature_set.name, new_feature_set.name))

    # Currently we only support conversion for feature sets with
    # one feature block each so we'll dig the feature blocks directly
    # and forget about the set.
    old_feature_block = self.feature_set.features[0]
    new_feature_block = new_feature_set.features[0]

    # next(iter(new_feature_block.factors)) is the way to get the
    # first item in a OrderedDict. (the ordered dict being str : int
    # mapping of the factor name to its size).
    # It is our new_feature_factor_name.
    # For example old_feature_block.name == "HalfKP"
    # and new_feature_factor_name == "HalfKP^"
    # We assume here that the "^" denotes factorized feature block
    # and we would like feature block implementers to follow this convention.
    # So if our current feature_set matches the first factor in the new_feature_set
    # we only have to add the virtual feature on top of the already existing real ones.
    if old_feature_block.name == next(iter(new_feature_block.factors)):
      # We can just extend with zeros since it's unfactorized -> factorized
      weights = self.input.weight
      padding = weights.new_zeros((weights.shape[0], new_feature_block.num_virtual_features))
      weights = torch.cat([weights, padding], dim=1)
      self.input.weight = nn.Parameter(weights)
      self.feature_set = new_feature_set
    else:
      raise Exception('Cannot change feature set from {} to {}.'.format(self.feature_set.name, new_feature_set.name))

  def forward(self, us, them, w_in, b_in):
    w = self.input(w_in)
    b = self.input(b_in)
    l0_ = (us * torch.cat([w, b], dim=1)) + (them * torch.cat([b, w], dim=1))
    # clamp here is used as a clipped relu to (0.0, 1.0)
    l0_ = torch.clamp(l0_, 0.0, 1.0)
    l1_ = torch.clamp(self.l1(l0_), 0.0, 1.0)
    l2_ = torch.clamp(self.l2(l1_), 0.0, 1.0)
    x = self.output(l2_)
    return x

  def step_(self, batch, batch_idx, loss_type):
    us, them, white, black, outcome, score, ply = batch

    # 600 is the kPonanzaConstant scaling factor needed to convert the training net output to a score.
    # This needs to match the value used in the serializer
    nnue2score = 600
    scaling = self.score_scaling

    q = self(us, them, white, black) * nnue2score / scaling
    t = outcome * (1.0 - self.label_smoothing_eps * 2.0) + self.label_smoothing_eps
    p = (score / scaling).sigmoid()

    epsilon = 1e-12
    teacher_entropy = -(p * (p + epsilon).log() + (1.0 - p) * (1.0 - p + epsilon).log())
    outcome_entropy = -(t * (t + epsilon).log() + (1.0 - t) * (1.0 - t + epsilon).log())
    teacher_loss = -(p * F.logsigmoid(q) + (1.0 - p) * F.logsigmoid(-q))
    outcome_loss = -(t * F.logsigmoid(q) + (1.0 - t) * F.logsigmoid(-q))
    if self.lambda_[self.parameter_index] >= 0.0:
      lambda_ = self.lambda_[self.parameter_index]
    else:
      lambda_ = (self.ply_end_threshold - ply) / (self.ply_end_threshold - self.ply_begin_threshold)
      lambda_ = torch.clamp(lambda_ , 0.0, 1.0)
    result  = lambda_ * teacher_loss    + (1.0 - lambda_) * outcome_loss
    entropy = lambda_ * teacher_entropy + (1.0 - lambda_) * outcome_entropy
    loss = result.mean() - entropy.mean()
    self.log(loss_type, loss)
    return loss

    # MSE Loss function for debugging
    # Scale score by 600.0 to match the expected NNUE scaling factor
    # output = self(us, them, white, black) * 600.0
    # loss = F.mse_loss(output, score)

  def training_step(self, batch, batch_idx):
    return self.step_(batch, batch_idx, 'train_loss')

  def validation_step(self, batch, batch_idx):
    return self.step_(batch, batch_idx, 'val_loss')

  def test_step(self, batch, batch_idx):
    self.step_(batch, batch_idx, 'test_loss')

  def _setup_muon_parameters(self):
    """Muon用のパラメータ分類"""
    hidden_weights = []
    hidden_gains_biases = []
    nonhidden_params = []
    
    # 入力層 (特徴量埋め込み層) - 通常はAdamWで最適化
    nonhidden_params.extend(self.input.parameters())
    
    # 隠れ層 - Muonで最適化
    for layer in [self.l1, self.l2]:
      for param in layer.parameters():
        if param.ndim >= 2:  # 重み行列
          hidden_weights.append(param)
        else:  # バイアス
          hidden_gains_biases.append(param)
    
    # 出力層 - AdamWで最適化
    nonhidden_params.extend(self.output.parameters())
    
    return hidden_weights, hidden_gains_biases, nonhidden_params


  def configure_optimizers(self):
    # Muonオプティマイザーの設定
    hidden_weights, hidden_gains_biases, nonhidden_params = self._setup_muon_parameters()
    
    param_groups = []
    
    # Muon部分（隠れ層の重み行列）
    if len(hidden_weights) > 0:
      param_groups.append({
        'params': hidden_weights,
        'use_muon': True,
        'lr': self.lr[0] * 0.1,  # Muon用の基本学習率
        'momentum': 0.95,         # Muonのmomentum
        'weight_decay': 0.01,
      })
    
    # AdamW部分（バイアス、入力層、出力層）
    adamw_params = hidden_gains_biases + nonhidden_params
    if len(adamw_params) > 0:
      param_groups.append({
        'params': adamw_params,
        'use_muon': False,
        'lr': self.lr[0] * 1e-3,  # AdamW用の基本学習率
        'betas': (0.9, 0.95),     # AdamW用のbetas
        'eps': 1e-10,             # AdamW用のeps
        'weight_decay': 0.01,
      })
    
    # シングルデバイス版のMuonを使用
    optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=800, eta_min=1e-5)
    
    return {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1,
        }
    }

  def on_train_epoch_end(self):
    # クリッピング処理
    for child in self.children():
      if not isinstance(child, nn.Linear):
        continue

      if child == self.input:
        continue

      # FC layers are stored as int8 weights, and int32 biases
      kWeightScaleBits = 6
      kActivationScale = 127.0
      if child != self.output:
        kBiasScale = (1 << kWeightScaleBits) * kActivationScale # = 8128
      else:
        kBiasScale = 9600.0 # kPonanzaConstant * FV_SCALE = 600 * 16 = 9600
      kWeightScale = kBiasScale / kActivationScale # = 64.0 for normal layers
      kMaxWeight = 127.0 / kWeightScale # roughly 2.0
      child.weight.data.clamp_(-kMaxWeight, kMaxWeight)

    # Log learning rates to TensorBoard
    optimizer = self.trainer.optimizers[0]
    for i, param_group in enumerate(optimizer.param_groups):
      group_name = f"muon_lr" if param_group.get('use_muon', False) else f"adamw_lr"
      self.log(f'lr/{group_name}', param_group['lr'], on_epoch=True, prog_bar=True)

  def get_layers(self, filt):
    """
    Returns a list of layers.
    filt: Return true to include the given layer.
    """
    for i in self.children():
      if filt(i):
        if isinstance(i, nn.Linear):
          for p in i.parameters():
            if p.requires_grad:
              yield p
