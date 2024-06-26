
import torch
import numpy as np
import wandb
from diffusers import DDIMPipeline
def simulate(agents, env, num_interaction_episodes, writer, train=True):
    for epi in range(num_interaction_episodes):
        act = []
        det = []
        pipeline = DDIMPipeline.from_pretrained("ddim")
        pipeline = pipeline.to(agents[0].device)
        observation, _ = env.reset()
        for x in range(env.num_agents):

            posterior, deterministic = agents[x].rssm.recurrent_model_input_init(1)
            action = np.zeros(env.action_space.shape[0], dtype=np.float32)
            action[1] = 1
            act.append(action)
            det.append((posterior, deterministic))
        observation, _ = env.reset()
        for i in range(200):
            _, _, _, _, _, = env.step(None)

        for i in range(4):
            observation, _, _, _, _, = env.step(act)
        emb = []
        for x in range(env.num_agents):

            embedded_observation = agents[x].encoder(
            torch.from_numpy(observation[x]).float().to(agents[0].device) )
            emb.append(embedded_observation)
        score = np.zeros(env.num_agents)
        score_lst = []
        done = np.zeros(env.num_agents)
        while not done.all():
            
            for i in range(env.num_agents):
                if not done[i] and not det[i] is None:

                    deterministic = agents[i].rssm.recurrent_model(
                    det[i][0], act[i]
                    )
                    _, posterior = agents[i].rssm.representation_model(
                    emb[i], deterministic
                    )
                    det[i] = (posterior, deterministic)
                    a = agents[i].actor(posterior, deterministic)[0].detach()
                    buffer_action = a.cpu().numpy()
                    env_action = buffer_action
                    act[i] = env_action
                else:
                    act[i] = None
                    det[i] = None
            env.render()
            next_observation, reward, done, info, _ = env.step(act)
            emb = []

            if train:
                for j in range(env.num_agents):
                    if next_observation[j] is not None:
                        agents[j].buffer.add(
                        observation[j], act[j], reward[j], next_observation[j], done[j]
                          )
                        emb.append(agents[j].encoder(
                        torch.from_numpy(next_observation[j]).float().to(agents[0].device)))
                    else:
                        emb.append(None)
                    

            score += reward
            
            observation = next_observation
            if done.all():
                if train and epi > 5 and epi % 2 == 0:
                    print(
                            "training scores", score, epi
                        )
                    for a in agents:
                        a.save_state_dict()
                        writer["episodic_return_" + str(a.agent_id+1)] = score[agents.index(a)]

                    print(">>>Saving Parameters<<<")
                    for j in range(env.num_agents):
                        agents[j].train(writer, pipeline)                    
                        wandb.log(data=writer)

                else:
                    score_lst = np.append(score_lst, score)
                    break
