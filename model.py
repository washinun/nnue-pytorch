import chess
import ranger21
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import sys
import math

# 3 layer fully connected network
L1 = 768
L2 = 8
L3 = 64

def get_parameters(layers):
    return [p for layer in layers for p in layer.parameters()]

class NNUE(pl.LightningModule):
  """
  This model attempts to directly represent the nodchip Stockfish trainer methodology.

  lambda_ = 0.0 - purely based on game results
  lambda_ = 1.0 - purely based on search scores

  It is not ideal for training a Pytorch quantized model directly.
  """
  def __init__(
      self,
      feature_set,
      lambda_=[1.0],
      max_epoch=800,
      gamma=0.992,
      lr=8.75e-4,
      epoch_size=100_000_000, 
      batch_size=16384, 
      label_smoothing_eps=0.0,
      score_scaling=511,
      ply_begin_threshold=100.0,
      ply_end_threshold=120.0
  ):
    super(NNUE, self).__init__()
    self.input = nn.Linear(feature_set.num_features, L1)
    self.feature_set = feature_set
    self.l1 = nn.Linear(2 * L1, L2)
    self.l2 = nn.Linear(L2, L3)
    self.output = nn.Linear(L3, 1)
    self.lambda_ = lambda_
    self.gamma = gamma
    self.lr = lr
    self.max_epoch = max_epoch
    self.epoch_size = epoch_size
    self.batch_size = batch_size
    self.label_smoothing_eps = label_smoothing_eps
    self.score_scaling = score_scaling
    self.parameter_index = 0
    self.ply_begin_threshold = ply_begin_threshold
    self.ply_end_threshold = ply_end_threshold

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

  # learning rate warm-up
  def optimizer_step(
      self,
      epoch,
      batch_idx,
      optimizer,
      optimizer_idx,
      optimizer_closure,
      on_tpu,
      using_native_amp,
      using_lbfgs,
  ):

    for pg in optimizer.param_groups:
      self.log("lr", pg["lr"])

    # update params
    optimizer.step(closure=optimizer_closure)

    # clip parameters
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

  def configure_optimizers(self):
    LR = self.lr
    train_params = [
        {"params": get_parameters([self.input]), "lr": LR, "gc_dim": 0},
        {"params": [self.l1.weight], "lr": LR},
        {"params": [self.l1.bias], "lr": LR},
        {"params": [self.l2.weight], "lr": LR},
        {"params": [self.l2.bias], "lr": LR},
        {"params": [self.output.weight], "lr": LR},
        {"params": [self.output.bias], "lr": LR},
    ]

    optimizer = ranger21.Ranger21(
        train_params,
        lr=1.0,
        betas=(0.9, 0.999),
        eps=1.0e-7,
        using_gc=False,
        using_normgc=False,
        weight_decay=0.0,
        num_batches_per_epoch=int(self.epoch_size / self.batch_size),
        num_epochs=self.max_epoch,
        warmdown_active=False,
        use_warmup=False,
        use_adaptive_gradient_clipping=False,
        softplus=False,
        pnm_momentum_factor=0.0,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=self.gamma
    )
    return [optimizer], [scheduler]

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
