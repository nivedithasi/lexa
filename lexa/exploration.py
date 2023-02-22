
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
    # print(kw)
    # dvdkw = dict(
    #     shape=200, layers=4, units=config.disag_units,
    #     act=config.act)
    # self.dvd = networks.DenseHead(**dvdkw)
    if self._config.dvd_classifier:
      if self._config.use_sth_sth:
        self.dvd = networks.get_dvd_model_cls("dvd", [512, 512, 256, 128, 64, 32], 174)
      elif self._config.use_robot_videos:
        self.dvd = networks.get_dvd_model_cls("dvd", [512, 512, 256, 128, 64, 32], 1)
      else:
        self.dvd = networks.get_dvd_model_cls("dvd", [512, 512, 256, 128, 64, 32], 203)
    else:
      if not self._config.dvd_dist:
          self.dvd = networks.get_dvd_model("dvd", [512, 512, 256, 128, 64, 32], 1)
      else:
          self.dvd = networks.get_dvd_model_dist("dvd", [512, 512, 256, 128], 100)
    # assert(False)
    # self.dvd = self.mlp_dvd_model()
    self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    if self._config.use_robot_videos:
      self.classification_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    else:
      self.classification_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    self.acc = tf.keras.metrics.Accuracy()
    
    
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

  def train(self, start, feat, embed, kl, dvd_data=None,  worldmodel=None):
            # dvd_pos_latent=None, 
            #                                dvd_neg_latent=None,
            #                                dvd_anchor_latent=None, worldmodel=None):
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
    if dvd_data is not None:
        metrics.update(self._train_dvd(dvd_data, worldmodel))
    metrics.update(self._behavior.train(start, self._intrinsic_reward)[-1])
    return None, metrics
        
  def _intrinsic_reward(self, feat, state, action, use_dist=False):
#     print("This is input to _intrinsic_reward")
#     print(feat)
#     print()
#     print(state)
#     print()
#     print(action)
#     print()
    
    def measure_dvd_similarity(pred):
      if self._config.dvd_classifier:
        pred_rs = tf.transpose(pred, perm=[1, 0, 2])
        pred_rs = tf.stack([pred_rs[:, 0], pred_rs[:, -1]], 1)
        inp = tf.reshape(pred_rs, [self._config.batch_length * self._config.batch_size, self._config.dvd_trajlen * 50])
        if self._config.use_robot_videos:
          score = self.dvd(inp)[:, 0]
        elif self._config.use_sth_sth:
          # for sth_sth
          # 109, 94, 100, 45, 44, 20, 37, 87, 12, 
          score = tf.reduce_mean(self.dvd(inp)[:, 44:46], 1)
        else:
          # for Ego4D
          score = tf.reduce_mean(self.dvd(inp)[:, 12:17], 1)
      else:
        pred_rs = tf.transpose(pred, perm=[1, 0, 2])
        pred_rs = tf.stack([pred_rs[:, 0], pred_rs[:, -1]], 1)
        tgt = tf.repeat(self.target_videos, self._config.batch_length, axis=0)
        inp = tf.concat([tgt, pred_rs], -1)
        inp = tf.reshape(inp, [self._config.batch_length * self._config.batch_size, self._config.dvd_trajlen * 2 * 50])
        score = tf.transpose(self.dvd(inp))
      return score

    if not use_dist:
        t0 = time.time()
        preds = [head(feat, tf.float32).mean() for head in self._networks]
        if self._config.dvd_score_weight > 0:
            scores = [measure_dvd_similarity(tf.cast(p, tf.float16)) for p in preds]
            avg_score = tf.math.reduce_mean(scores, 0)
            # return tf.cast(tf.repeat(avg_score, self._config.imag_horizon, axis=0), tf.float32)
        t1 = time.time()

        disag = tf.reduce_mean(tf.math.reduce_std(preds, 0), -1)
        if self._config.disag_log:
            disag = tf.math.log(disag)
        reward = self._config.expl_intr_scale * disag

    else:
        print("Using distance for dvd reward...")
        
    if self._config.dvd_score_weight > 0:
        reward = (1 - self._config.dvd_score_weight) * reward + self._config.dvd_score_weight * tf.cast(avg_score, tf.float32)

    if self._config.expl_extr_scale:
        reward += tf.cast(self._config.expl_extr_scale * self._reward(feat, state, action), tf.float32)
    
    return reward

  def _train_dvd(self, dvd_data, worldmodel):
#     dvd_pos_latent = tf.reshape(dvd_pos_latent, [self._config.batch_size, self._config.dvd_trajlen, 50])
#     dvd_neg_latent = tf.reshape(dvd_neg_latent, [self._config.batch_size, self._config.dvd_trajlen, 50])
#     dvd_anchor_latent = tf.reshape(dvd_anchor_latent, [self._config.batch_size, self._config.dvd_trajlen, 50])
    
#     self.target_videos = dvd_anchor_latent
    
#     pos_example = tf.concat([dvd_anchor_latent, dvd_pos_latent], -1)
#     neg_example = tf.concat([dvd_anchor_latent, dvd_neg_latent], -1)
#     inp = tf.concat([pos_example, neg_example], 0)
#     inp = tf.reshape(inp, [self._config.batch_size*2, self._config.dvd_trajlen * 2 * 50])
    
    # inp = tf.stop_gradient(inp)
    if self._config.dvd_classifier:
      ims = dvd_data[0]
      label = dvd_data[1]
      with tf.GradientTape() as tape:
        dvd_reshaped= tf.reshape(ims, [self._config.batch_size*1*self._config.dvd_trajlen, 64, 64, 3])
        _, dvd_latent = worldmodel.get_init_feat({"image": dvd_reshaped})
        dvd_latent_reshaped = tf.reshape(dvd_latent[self._config.disag_target], [self._config.batch_size, 1, self._config.dvd_trajlen, 50])

        # pos_example = tf.concat([dvd_latent_reshaped[:, 1], dvd_latent_reshaped[:, 0]], -1)
        # neg_example = tf.concat([dvd_latent_reshaped[:, 1], dvd_latent_reshaped[:, 2]], -1)
        # self.target_videos = dvd_latent_reshaped[:, 3]
        # inp = tf.concat([pos_example, neg_example], 0)
        # inp = tf.reshape(inp, [self._config.batch_size*2, self._config.dvd_trajlen * 2 * 50])
        inp = tf.reshape(dvd_latent_reshaped, [self._config.batch_size, self._config.dvd_trajlen * 50])

        if not self._config.dvd_e2e:
          inp = tf.stop_gradient(inp)

        print("this is input:", inp)
        preds = self.dvd(inp)
        labels = label
        loss = self.classification_loss(labels, preds)
    else:
      with tf.GradientTape() as tape:
        dvd_reshaped= tf.reshape(dvd_data, [self._config.batch_size*4*self._config.dvd_trajlen, 64, 64, 3])
        _, dvd_latent = worldmodel.get_init_feat({"image": dvd_reshaped})
        dvd_latent_reshaped = tf.reshape(dvd_latent[self._config.disag_target], [self._config.batch_size, 4, self._config.dvd_trajlen, 50])

        pos_example = tf.concat([dvd_latent_reshaped[:, 1], dvd_latent_reshaped[:, 0]], -1)
        neg_example = tf.concat([dvd_latent_reshaped[:, 1], dvd_latent_reshaped[:, 2]], -1)
        self.target_videos = dvd_latent_reshaped[:, 3]
        inp = tf.concat([pos_example, neg_example], 0)
        inp = tf.reshape(inp, [self._config.batch_size*2, self._config.dvd_trajlen * 2 * 50])

        if not self._config.dvd_e2e:
          inp = tf.stop_gradient(inp)

        print("this is input:", inp)
        if not self._config.dvd_dist:
          preds = self.dvd(inp)
          labels_neg = tf.zeros_like(preds[:(preds.shape[0] // 2)])
          labels_pos = tf.ones_like(preds[:(preds.shape[0] // 2)])
          labels = tf.concat([labels_pos, labels_neg], 0)
          loss = self.bce(labels, preds)
        else:
          preds = self.dvd(inp)
          print(preds)
          # labels_neg = tf.zeros_like(preds[:(preds.shape[0] // 2)])
          # labels_pos = tf.ones_like(preds[:(preds.shape[0] // 2)])
          # labels = tf.concat([labels_pos, labels_neg], 0)
          # loss = self.bce(labels, preds)

    if self._config.dvd_e2e:
      metrics = self._dvd_opt(tape, loss, [self.dvd, worldmodel])
    else:
      metrics = self._dvd_opt(tape, loss, [self.dvd])
      
    if self._config.dvd_classifier:
      self.acc.update_state(labels, preds)
      metrics["dvd_acc"] =  self.acc.result()
      # metrics["dvd_acc"] = tf.reduce_mean(tf.cast(tf.cast((preds > 0.5), tf.float16) == labels, tf.float32))
    else:
      metrics["dvd_pos_pred"] = tf.reduce_mean(preds[:(preds.shape[0] // 2)])
      metrics["dvd_neg_pred"] = tf.reduce_mean(preds[(preds.shape[0] // 2):])
      metrics["dvd_acc"] = tf.reduce_mean(tf.cast(tf.cast((preds > 0.5), tf.float16) == labels, tf.float32))
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
