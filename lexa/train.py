import os
import pdb
import functools
import tools
import tensorflow as tf
import numpy as np
import pickle
import pathlib
import off_policy
from dataloader import VideoFolder
from args import load_args
from collections import defaultdict
import torchvision
from transforms_video import ComposeMix, RandomCropVideo, RandomRotationVideo, Scale
from dreamer import Dreamer, setup_dreamer, create_envs, count_steps, make_dataset, make_dvd_dataset, parse_dreamer_args

# if (!working) {
#     working = true;
# }

class GCDreamer(Dreamer):
  def __init__(self, config, logger, dataset, dvd_dataset):
    if config.offpolicy_opt:
      self._off_policy_handler = off_policy.GCOffPolicyOpt(config)
    super().__init__(config, logger, dataset, dvd_dataset)
    self._should_expl_ep = tools.EveryNCalls(config.expl_every_ep)
    self.skill_to_use = tf.zeros([0], dtype=tf.float16)

  def get_one_time_skill(self):
    skill = self.skill_to_use
    self.skill_to_use = tf.zeros([0], dtype=tf.float16)
    return skill

  def _policy(self, obs, state, training, reset):
    # Choose the goal for the next episode
    # TODO this would be much more elegant if it was implemented in the training loop (can simulate a single episode)
    if state is None:
      if training and self._config.training_goals == 'batch':
        # TODO this modifies the underlying dict, is that fine?
        obs['image_goal'], obs['goal'] = self.sample_replay_goal(obs)
      obs['skill'] = self.get_one_time_skill()
      # The world model adds the observation to the state in this case

    if reset.any() and state is not None:
      # Replace the goal in the agent state at new episode
      # The actor always takes goal from the state, not the observation
      if training and self._config.training_goals == 'batch':
        state[0]['image_goal'], state[0]['goal'] = self.sample_replay_goal(obs)
      else:
        state[0]['image_goal'] = tf.cast(obs['image_goal'], self._float) / 255.0 - 0.5
      state[0]['skill'] = self.get_one_time_skill()

      # Toggle exploration
      self._should_expl_ep()

    # TODO double check everything
    #return super()._policy(obs, state, training, reset, should_expl=self._should_expl_ep.value)
    #if not training:
    return super()._policy(obs, state, training, reset, should_expl=self._should_expl_ep.value)

  def sample_replay_goal(self, obs):
    """ Sample goals from replay buffer """
    assert self._config.gc_input != 'state'
    random_batch = next(self._dataset)
    random_batch = self._wm.preprocess(random_batch)
    
    images = random_batch['image']
    states = random_batch['state']
    if self._config.labelled_env_multiplexing:
      assert obs['env_idx'].shape[0] == 1
      env_ids = random_batch['env_idx'][:, 0]
      if tf.reduce_any(env_ids == obs['env_idx']):
        ids = np.nonzero(env_ids == obs['env_idx'])[0]
        images = tf.gather(images, ids)
        states = tf.gather(states, ids)
    
    random_goals = tf.reshape(images, (-1,) + tuple(images.shape[2:]))
    random_goal_states = tf.reshape(states, (-1,) + tuple(states.shape[2:]))
    # random_goals = tf.random.shuffle(random_goals)
    return random_goals[:obs['image_goal'].shape[0]], random_goal_states[:obs['image_goal'].shape[0]]

def process_eps_data(eps_data):
  keys = eps_data[0].keys()
  new_data = {}
  for key in keys:
    new_data[key] = np.array([eps_data[i][key] for i in range(len(eps_data))]).squeeze()
  return new_data

def sample_dvd(episodes, length=None, balance=False, seed=0):
    pass

def main(logdir, config):
  logdir, logger = setup_dreamer(config, logdir)
  eval_envs, eval_eps, train_envs, train_eps, acts = create_envs(config, logger)

  prefill = max(0, config.prefill - count_steps(config.traindir))
  print(f'Prefill dataset ({prefill} steps).')
#   args = load_args()
  upscale_size_train = int(1.4 * 64)
  upscale_size_eval = int(1.0 * 64)
#   transform_train_pre = ComposeMix([
#             [RandomRotationVideo(15), "vid"],
#             [Scale(upscale_size_train), "img"],
#             [RandomCropVideo(64), "img"],
#              ])
  transform_eval_pre = ComposeMix([
          [Scale(upscale_size_eval), "img"],
          [torchvision.transforms.ToPILImage(), "img"],
          [torchvision.transforms.CenterCrop(64), "img"],
           ])
  transform_post = ComposeMix([[torchvision.transforms.ToTensor(), "img"],])
  dvd_data = VideoFolder(root='/iris/u/asc8/workspace/humans/Humans/20bn-something-something-v2-all-videos/',
                           json_file_input='/iris/u/surajn/workspace/language_offline_rl/sthsth/something-something-v2-train.json',
                           json_file_labels='/iris/u/surajn/workspace/language_offline_rl/sthsth/something-something-v2-labels.json',
                             clip_size= config.batch_length, #args.traj_length,
                             nclips=1,
                             step_size=1,
                             num_tasks=2, #args.num_tasks,
                             is_val=False,
                             transform_pre=transform_eval_pre,
                             transform_post=transform_post,
                             ) # Niveditha: add args back
    
  dvd_dataset = make_dvd_dataset(dvd_data, config)
  random_agent = lambda o, d, s: ([acts.sample() for _ in d], s)
  tools.simulate(random_agent, train_envs, prefill)
  if count_steps(config.evaldir) == 0:
    tools.simulate(random_agent, eval_envs, episodes=1)
  logger.step = config.action_repeat * count_steps(config.traindir)

  print('Simulate agent.')
  train_dataset = make_dataset(train_eps, config)
  eval_dataset = iter(make_dataset(eval_eps, config))
  # print("Next dvd iter: ", next(iter(dvd_data)))
  """
  TODO: 
  Create a new dataset here which returns clips from the something-something dataset
  Something like this class: https://github.com/anniesch/dvd/blob/main/data_loader_av.py#L21
  but it will need to use a tensorflow data loader instead of pytorch.
  Also will probably make sense for it to return two clips of the same task per batch, 
  for training the DVD model
  """
  agent = GCDreamer(config, logger, train_dataset, dvd_dataset)
  if (logdir / 'variables.pkl').exists():
    agent.load(logdir / 'variables.pkl')
    agent._should_pretrain._once = False

  pathlib.Path(logdir / "distance_func_logs_trained_model").mkdir(parents=True, exist_ok = True)

  state = None
  assert len(eval_envs) == 1
  while agent._step.numpy().item() < config.steps:
    logger.write()
    print('Start gc evaluation.')
    executions = []
    goals = []
    #rews_across_goals = []
    num_goals = min(100, len(eval_envs[0].get_goals()))
    all_eps_data = []
    num_eval_eps = 1
    for ep_idx in range(num_eval_eps):
      ep_data_across_goals = []
      for idx in range(num_goals):
        eval_envs[0].set_goal_idx(idx)
        eval_policy = functools.partial(agent, training=False)
        sim_out = tools.simulate(eval_policy, eval_envs, episodes=1)
        obs, eps_data = sim_out[4], sim_out[6]

        ep_data_across_goals.append(process_eps_data(eps_data))
        video = eval_envs[0]._convert([t['image'] for t in eval_envs[0]._episode])
        executions.append(video[None])
        goals.append(obs[0]['image_goal'][None])

      all_eps_data.append(ep_data_across_goals)

    if ep_idx == 0:
      executions = np.concatenate(executions, 0)
      goals = np.stack(goals, 0)
      goals = np.repeat(goals, executions.shape[1], 1)
      gc_video = np.concatenate([goals, executions], -3)
      agent._logger.video(f'eval_gc_policy', gc_video)
      logger.write()

    with pathlib.Path(logdir / ("distance_func_logs_trained_model/step_"+str(logger.step)+".pkl") ).open('wb') as f:
      pickle.dump(all_eps_data, f)

    if config.sync_s3:
      os.system('aws s3 sync '+str(logdir)+ ' s3://goalexp2021/research_code/goalexp_data/'+str(logdir))

    if not config.training:
        continue
    print('Start training.')
    state = tools.simulate(agent, train_envs, config.eval_every, state=state)
    agent.save(logdir / 'variables.pkl')
  for env in train_envs + eval_envs:
    try:
      env.close()
    except Exception:
      pass


if __name__ == '__main__':
  args, remaining = parse_dreamer_args()
  main(args.logdir, remaining)
