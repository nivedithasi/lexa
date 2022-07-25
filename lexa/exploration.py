
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec
from tensorflow_probability import distributions as tfd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten

import models
import networks
import tools
import time

class Random(tools.Module):

  def __init__(self, config):


    self._config = config
    self._float = prec.global_policy().compute_dtype

  def actor(self, feat, *args):
    shape = feat.shape[:-1] + [self._config.num_actions]
    if self._config.actor_dist == 'onehot':
      return tools.OneHotDist(tf.zeros(shape))
    else:
      ones = tf.ones(shape, self._float)
      return tfd.Uniform(-ones, ones)

  def train(self, start, feat, embed, kl):
    return None, {}

class Plan2Explore(tools.Module):

  def __init__(self, config, world_model, reward=None):

    self._config = config
    self._reward = reward
    self._behavior = models.ImagBehavior(config, world_model)
    self.actor = self._behavior.actor
    size = {
        'embed': 32 * config.cnn_depth,
        'stoch': config.dyn_stoch,
        'deter': config.dyn_deter,
        'feat': config.dyn_stoch + config.dyn_deter,
    }[self._config.disag_target]
    kw = dict(
        shape=size, layers=config.disag_layers, units=config.disag_units,
        act=config.act)
    self._networks = [
        networks.DenseHead(**kw) for _ in range(config.disag_models)]
    self.dvd = self.mlp_dvd_model()
    self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    
    
    """
    TODO:
    Add a DVD like network head here which will take (1) a robot trajectory
    and (2) an embedded human video trajectory, and predict their similarity
    """
    self._opt = tools.Optimizer(
        'ensemble', config.model_lr, config.opt_eps, config.grad_clip,
        config.weight_decay, opt=config.opt)
    self._dvd_opt = tools.Optimizer(
        'dvd', config.model_lr, config.opt_eps, config.grad_clip,
        config.weight_decay, opt=config.opt)

  def train(self, start, feat, embed, kl, dvd_pos_latent=None, 
                                           dvd_neg_latent=None,
                                           dvd_anchor_latent=None):
    metrics = {}
    target = {
        'embed': embed,
        'stoch': start['stoch'],
        'deter': start['deter'],
        'feat': feat,
    }[self._config.disag_target]
    """
    TODO:
    Will need to add two new things here
    First, will need to write/call a function to train the DVD model (only on the paired human videos) and their labels
    Second, will need to modify (or make a new version) of the _intrinsic_reward function which is not just the 
    model disagreement, but also the learned models similarity score between robot behavior and some target video(s)
    """
    metrics.update(self._train_ensemble(feat, target))
    if dvd_pos_latent is not None:
        metrics.update(self._train_dvd(dvd_pos_latent[self._config.disag_target], 
                                       dvd_neg_latent[self._config.disag_target],
                                       dvd_anchor_latent[self._config.disag_target]))
    metrics.update(self._behavior.train(start, self._intrinsic_reward)[-1])
    return None, metrics

  def mlp_dvd_model(self):
    model = Sequential()
    model.add(Conv1D(32, 10, activation='relu', kernel_initializer='he_normal', input_shape=(self._config.dvd_trajlen, 100)))
    # model.add(Conv1D(32, 5, activation='relu', kernel_initializer='he_normal'))
    model.add(Conv1D(64, 5, activation='relu', kernel_initializer='he_normal'))
    # model.add(Conv1D(64, 5, activation='relu', kernel_initializer='he_normal'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile the model
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

        
  def _intrinsic_reward(self, feat, state, action):
    def measure_dvd_similarity(pred):
      pred_rs = tf.transpose(pred, perm=[1, 0, 2])
      tgt = tf.repeat(self.target_videos, self._config.batch_length, axis=0)
      inp = tf.concat([tgt, pred_rs], -1)
      score = tf.transpose(self.dvd(inp))
      return score
    
    t0 = time.time()
    preds = [head(feat, tf.float32).mean() for head in self._networks]
    if self._config.dvd_score_weight > 0:
      scores = [measure_dvd_similarity(head(feat).mean()) for head in self._networks]
      avg_score = tf.math.reduce_mean(scores, 0)
    t1 = time.time()
    print("DVD SIM TIME", t1-t0)

    disag = tf.reduce_mean(tf.math.reduce_std(preds, 0), -1)
    if self._config.disag_log:
      disag = tf.math.log(disag)
    reward = self._config.expl_intr_scale * disag

    if self._config.dvd_score_weight > 0:
      reward = reward + self._config.dvd_score_weight * tf.cast(avg_score, tf.float32)
    if self._config.expl_extr_scale:
      reward += tf.cast(self._config.expl_extr_scale * self._reward(
          feat, state, action), tf.float32)
    return reward

  def _train_dvd(self, dvd_pos_latent, dvd_neg_latent, dvd_anchor_latent):
    dvd_pos_latent = tf.reshape(dvd_pos_latent, [self._config.batch_size, self._config.dvd_trajlen, 50])
    dvd_neg_latent = tf.reshape(dvd_neg_latent, [self._config.batch_size, self._config.dvd_trajlen, 50])
    dvd_anchor_latent = tf.reshape(dvd_anchor_latent, [self._config.batch_size, self._config.dvd_trajlen, 50])
    
    self.target_videos = dvd_anchor_latent
    
    pos_example = tf.concat([dvd_anchor_latent, dvd_pos_latent], -1)
    neg_example = tf.concat([dvd_anchor_latent, dvd_neg_latent], -1)
    inp = tf.concat([pos_example, neg_example], 0)
    
    inp = tf.stop_gradient(inp)
    with tf.GradientTape() as tape:
      preds = self.dvd(inp)
      labels_neg = tf.zeros_like(preds[:(preds.shape[0] // 2)])
      labels_pos = tf.ones_like(preds[:(preds.shape[0] // 2)])
      labels = tf.concat([labels_pos, labels_neg], 0)
      loss = self.bce(preds, labels)
    metrics = self._dvd_opt(tape, loss, self.dvd)
    return metrics
  
  
  def _train_ensemble(self, inputs, targets):
    if self._config.disag_offset:
      targets = targets[:, self._config.disag_offset:]
      inputs = inputs[:, :-self._config.disag_offset]
    targets = tf.stop_gradient(targets)
    inputs = tf.stop_gradient(inputs)
    """ Niveditha: Should we just add the value of human score here?"""
    with tf.GradientTape() as tape:
      preds = [head(inputs) for head in self._networks]
      likes = [tf.reduce_mean(pred.log_prob(targets)) for pred in preds]
      loss = -tf.cast(tf.reduce_sum(likes), tf.float32)
    metrics = self._opt(tape, loss, self._networks)
    return metrics

  def act(self, feat, *args):
    return self.actor(feat)
