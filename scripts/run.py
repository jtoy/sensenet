import sys
import sensenet

if len(sys.argv) >= 2:
  model_file = sys.argv[1]
  print("loading model", model_file)
env = sensenet.make(name_to_env(model_file),{'render':True})
#model = sensenet.get_model name_to_pymodel(model_file)
observation = env.reset()
steps = 0
total_steps = 0
episode_count = 0
def select_action(state,n_actions,model):
  state = torch.from_numpy(state).float().unsqueeze(0)
  probs = model(Variable(state))
  action = probs.multinomial()
  return action.data[0][0]
while (1):
  if total_steps%100=0:
    print("total steps ...",total_steps)
    print("episode ",episode_count," step ", steps)
  if done:
    env.reset()
    episode_count += 1
    steps = 0
  action = select_action(observation,observation,model)
  observation,reward,done,info = env.step(action)
  steps +=1
  total_steps += 1

