import torch
import torch.nn as nn
import os
import numpy as np
import wandb
import random
import torch.nn.functional as F
from dreamer.modules.model import RSSM, RewardModel, ContinueModel

from dreamer.modules.encoder import Encoder, ImagEncoder
from dreamer.modules.decoder import Decoder
from dreamer.modules.actor import Actor
from dreamer.modules.critic import Critic
from dreamer.utils.utils import (
    compute_lambda_values,
    create_normal_dist,
    DynamicInfos,
    find_dir,
    symlog,
    GaussianFilterLayer,
    Disc
)
from dreamer.utils.buffer import ReplayBuffer

'''
Contributions:

soft_update: The soft update uses a target and incrementally updates the weights towards the 
target weights slowly. The updated weights are treated as a convex hull of the target and the source.

hard_update: The hard update will replace all of the weights with the target weights

_model_update: Added KL divergence in the model update 

agent_update: Updates the actor and the critic model, we backpropagate using policy gradient method REINFORCE

latent_imagination: Takes the current state (both a determinisitc state and a stochastic state)
then uses it to imagine trajectories of horizon length found in the config file 
without interaction in the world using the RSSM world model. Uses diffusion generated images, and takes in true images
to train a discriminator to differentiate between the true and generated images.

latent_imagination: Takes the current state (both a determinisitc state and a stochastic state)
then uses it to imagine trajectories of horizon length found in the config file 
without interaction in the world using the RSSM world model. Uses only observations 
from the true environment (ie no diffusion).

'''

class DreamerV3:
    def __init__(self,
        agent_id,
        observation_shape,
        action_size,
        writer,
        device,
        config,
        LSTM, 
        baseline = False
    ):
        self.agent_id = agent_id
        self.device = device
        self.action_size = action_size
        self.baseline = baseline
        
        self.encoder = Encoder(observation_shape, config).to(self.device)
        self.target_encoder = Encoder(observation_shape, config).to(self.device)
        self.hard_update(self.target_encoder, self.encoder)
        self.decoder = Decoder(observation_shape, config).to(self.device)
        self.rssm = RSSM(action_size, config, LSTM).to(self.device)
        self.reward_predictor = RewardModel(config).to(self.device)
        self.hard_update(self.rssm.transition_model_target, self.rssm.transition_model)
        if config.parameters.dreamer.use_continue_flag:
            self.continue_predictor = ContinueModel(config).to(self.device)
        self.actor = Actor(action_size, config).to(self.device)
        self.critic = Critic(config).to(self.device)
        self.targ_critic = Critic(config).to(self.device)
        self.targ_critic.load_state_dict(self.critic.state_dict())
        self.buffer = ReplayBuffer(observation_shape, action_size, self.device, config)
        self.imag_encoder = ImagEncoder( config).to(self.device)
        self.config = config.parameters.dreamer
        self.disc = Disc(self.config.embedded_state_size).to(self.device)
        self.num_updates = 0
        
        # optimizer
        self.model_params = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.rssm.parameters())
            + list(self.reward_predictor.parameters())
        )
        self.imag_encoder_params = self.imag_encoder.parameters()
        if self.config.use_continue_flag:
            self.model_params += list(self.continue_predictor.parameters())

        # epsilon are 1e-8 and 1e-5
        self.model_optimizer = torch.optim.Adam(
            self.model_params, lr=self.config.model_learning_rate, eps=self.config.model_epsilon,
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.actor_learning_rate,eps=self.config.actor_epsilon
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config.critic_learning_rate, eps=self.config.critic_epsilon
        )
        self.imag_encoder_optim = torch.optim.Adam(self.imag_encoder_params, lr=self.config.encoder_learning_rate)

        self.dynamic_learning_infos = DynamicInfos(self.device)
        self.behavior_learning_infos = DynamicInfos(self.device)
        self.writer = writer
        self.num_total_episode = 0
        self.gf = GaussianFilterLayer().to(self.device)
        self.disc_optim = torch.optim.Adam(self.disc.parameters(), lr=self.config.discriminator_learning_rate)

        print("Agent " + str(self.agent_id)+ " Initiated!")
        
        
    def hard_update(self, target, original):
        target.load_state_dict(original.state_dict())


    def soft_update(self, target, source, tau=0.02):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


    def train(self, metrics, pipeline):

        for i in range(self.config.train_iterations):
            data = self.buffer.sample(
                    self.config.batch_size, self.config.batch_length
                )
            _, deterministic = self.dynamic_learning(data, metrics)
            deterministic = self.rssm.recurrent_model.input_init(50)
            #deterministic = self.rssm.recurrent_model.input_init(100)
            if self.baseline:
                self.latent_imagination_baseline(data.observation[:, 0, :], deterministic, metrics)
            else:
                if i % 4 == 0:
                    x = pipeline(batch_size = 50, num_inference_steps=30, output_type="np", return_dict = False)[0]
                    x = torch.from_numpy(x).to(self.device).float().squeeze(0).permute((0, 3,1,2))

                self.latent_imagination(data.observation[:, 0, :], x, deterministic, metrics)


    def dynamic_learning(self, data, metrics):
        prior, deterministic = self.rssm.recurrent_model_input_init(len(data.action))

        data.embedded_observation = self.encoder(data.observation, seq=1)
        for t in range(1, self.config.batch_length):
            deterministic = self.rssm.recurrent_model(
                prior, data.action[:, t - 1]
            )

            prior_dist, prior = self.rssm.transition_model(deterministic)
            

            posterior_dist, posterior = self.rssm.representation_model(
                data.embedded_observation[:,t], deterministic
            )

            

            self.dynamic_learning_infos.append(
                priors=prior,
                posteriors=posterior,
                prior_dists_sg_mean = prior_dist.mean.detach(),
                prior_dists_sg_std = prior_dist.scale.detach(),

                posterior_dists_sg_std = posterior_dist.scale.detach(),
                posterior_dists_sg_mean = posterior_dist.mean.detach(),

                prior_dist_means=prior_dist.mean,
                prior_dist_stds=prior_dist.scale,
                posterior=posterior,
                posterior_dist_means=posterior_dist.mean,
                posterior_dist_stds=posterior_dist.scale,
                deterministics=deterministic,
            )

            prior = posterior

        infos = self.dynamic_learning_infos.get_stacked()
        self._model_update(data, infos, metrics)
        return infos.posteriors.detach(), infos.deterministics.detach()


    def _model_update(self, data, posterior_info, metrics):
        reconstructed_observation = self.decoder(
            posterior_info.posteriors, posterior_info.deterministics, seq=1
        ) 
        reconstruction_observation_loss = nn.MSELoss(reduction="sum")(reconstructed_observation, 
            data.observation[:, 1:])
        
        if self.config.use_continue_flag:
            continue_dist = self.continue_predictor(
                posterior_info.posteriors, posterior_info.deterministics
            )
            continue_loss = -continue_dist.log_prob(
                 1 - data.done[:, 1:].squeeze(-1)
            ).mean()

        reward_dist = self.reward_predictor(
            posterior_info.posteriors, posterior_info.deterministics
        )
        reward_loss = nn.MSELoss()(reward_dist.mean, symlog(data.reward[:, 1:]))

        prior_dist = create_normal_dist(
            posterior_info.prior_dist_means,
            posterior_info.prior_dist_stds,
            event_shape=1,
        )
        posterior_dist = create_normal_dist(
            posterior_info.posterior_dist_means,
            posterior_info.posterior_dist_stds,
            event_shape=1,
        )
        prior_dist_sg = create_normal_dist(
            posterior_info.prior_dists_sg_mean.detach(),
            posterior_info.prior_dists_sg_std.detach(),
            event_shape=1,
        )
        posterior_dist_sg = create_normal_dist(
            posterior_info.posterior_dists_sg_mean.detach(),
            posterior_info.posterior_dists_sg_std.detach(),
            event_shape=1,
        )
        kl1 =  torch.distributions.kl.kl_divergence(posterior_dist, prior_dist_sg)
        kl2 = torch.distributions.kl.kl_divergence(posterior_dist_sg, prior_dist)
        kl_divergence_loss = torch.max(torch.ones_like(kl1, device=kl1.device), kl1).mean()+torch.max( torch.ones_like(kl2, device=kl2.device), kl2).mean()
        model_loss = (
            self.config.kl_divergence_scale * kl_divergence_loss
            + reconstruction_observation_loss
            + reward_loss
        )
        if self.config.use_continue_flag:
            model_loss += continue_loss.mean()

        self.model_optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(
            self.model_params,
            self.config.clip_grad*10,
            norm_type=self.config.grad_norm_type,
        )
        self.model_optimizer.step()
        id = self.agent_id + 1
        metrics['dynamics_loss_' + str(id)] = kl1.mean().item()
        metrics['representation_loss_' + str(id)] = kl2.mean().item()
        metrics['reconstruction_loss_'+str(id)] = reconstruction_observation_loss.mean().item()
        metrics['reward_loss_'+str(id)] = reward_loss.mean().item()
        metrics['continue_loss_'+str(id)] = continue_loss.mean().item()

        self.soft_update(self.rssm.transition_model_target, self.rssm.transition_model)
        self.soft_update(self.target_encoder, self.encoder)


    def latent_imagination_baseline(self, true_obs, det, metrics):
        z = self.encoder(true_obs).detach()
        deterministic = det.reshape(-1, self.config.deterministic_size)

        _, state = self.rssm.representation_model(z.clone(), deterministic.clone().detach())
        gt = torch.zeros((z.shape[0])).to(self.device)

        state = state.reshape(-1, self.config.stochastic_size)

        # continue_predictor reinit
        for t in range(self.config.horizon_length):
            action, log_prob = self.actor(state, deterministic)
            deterministic = self.rssm.recurrent_model(state, action)
            _, state = self.rssm.transition_model(deterministic)

            self.behavior_learning_infos.append(
                priors=state, deterministics=deterministic,
                actions=action, 
                log_probs = log_prob
            )

        self._agent_update(self.behavior_learning_infos.get_stacked(), metrics)


    def latent_imagination(self, true_obs, gen_obs, det, metrics):
        z = self.encoder(true_obs).detach()
        z_ = self.imag_encoder(self.encoder(gen_obs).detach())
        deterministic = det.reshape(-1, self.config.deterministic_size)

        _, state = self.rssm.representation_model(z_.clone(), deterministic.clone().detach())
        gt = torch.zeros((z_.shape[0])).to(self.device)


        self.disc_optim.zero_grad()
        real = self.disc(z)
        fake = self.disc(z_)

        real_loss = F.binary_cross_entropy_with_logits(real, torch.zeros_like(real))
        fake_loss = F.binary_cross_entropy_with_logits(fake, torch.ones_like(fake))
        loss = real_loss + fake_loss


        nn.utils.clip_grad_norm_(
            self.disc.parameters(),
            self.config.clip_grad * 10,
            norm_type=self.config.grad_norm_type,
        )

        loss.backward(retain_graph=True)


        for i in range(2):
            self.disc_optim.step()

            self.imag_encoder_optim.zero_grad()
            fake = self.disc(z_)

            gen_loss = F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake))

            gen_loss.backward(retain_graph=True)

            nn.utils.clip_grad_norm_(
                self.imag_encoder.parameters(),
                self.config.clip_grad * 10,
                norm_type=self.config.grad_norm_type,
            )

            self.imag_encoder_optim.step()


        metrics["discrimimnator_loss"] = loss.item()
        metrics["imag_encoder_loss"] = gen_loss.item()

        state = state.reshape(-1, self.config.stochastic_size)

        # continue_predictor reinit
        for t in range(self.config.horizon_length):
            action, log_prob = self.actor(state, deterministic)
            deterministic = self.rssm.recurrent_model(state, action)
            _, state = self.rssm.transition_model(deterministic)

            self.behavior_learning_infos.append(
                priors=state, deterministics=deterministic,
                actions=action, 
                log_probs = log_prob
            )

        self._agent_update(self.behavior_learning_infos.get_stacked(), metrics)
        
        
    def save_state_dict(self):
        self.rssm.recurrent_model.input_init(1)
        id = self.agent_id + 1
        torch.save(self.rssm.state_dict(), os.path.join(find_dir('pretrained_parameters'),'RSSM'+str(id)))
        torch.save(self.encoder.state_dict(), os.path.join(find_dir('pretrained_parameters'),'ENCODER'+str(id)))   
        torch.save(self.actor.state_dict(), os.path.join(find_dir('pretrained_parameters'),'ACTOR'+str(id)))        
        torch.save(self.critic.state_dict(), os.path.join(find_dir('pretrained_parameters'),'CRITIC'+str(id)))        
        torch.save(self.reward_predictor.state_dict(), os.path.join(find_dir('pretrained_parameters'),'REWARD'+str(id)))        
        torch.save(self.continue_predictor.state_dict(), os.path.join(find_dir('pretrained_parameters'),'CONTINUE'+str(id)))   


    def load_state_dict(self):
        id = self.agent_id + 1
        self.rssm.load_state_dict(torch.load('pretrained_parameters/RSSM'+ str(id)) )
        self.encoder.load_state_dict(torch.load('pretrained_parameters/ENCODER'+ str(id)))
        self.actor.load_state_dict(torch.load('pretrained_parameters/ACTOR'+ str(id)))
        self.critic.load_state_dict(torch.load('pretrained_parameters/CRITIC'+ str(id)))
        self.reward_predictor.load_state_dict(torch.load('pretrained_parameters/REWARD'+ str(id)))
        self.continue_predictor.load_state_dict(torch.load('pretrained_parameters/CONTINUE'+ str(id)))
        
        
    def _agent_update(self, behavior_learning_infos, metrics):
        predicted_rewards = self.reward_predictor(
            behavior_learning_infos.priors, behavior_learning_infos.deterministics, eval = True
        ).rsample()
        values = self.critic(
            behavior_learning_infos.priors, behavior_learning_infos.deterministics, eval=True
        ).mean
        if self.config.use_continue_flag:
            continues = self.continue_predictor(
                behavior_learning_infos.priors, behavior_learning_infos.deterministics
            ).sample().unsqueeze(-1)
        else:
            continues = self.config.discount * torch.ones_like(values)

        lambda_values = compute_lambda_values(
            predicted_rewards,
            values,
            continues,
            self.config.horizon_length,
            self.device,
            behavior_learning_infos.log_probs,
            self.config.lambda_,
            self.config.alpha
        )
        id = self.agent_id + 1
        if self.num_updates % 2 == 0:
            actor_loss = -torch.mean(lambda_values) #-torch.log(1-actions**2+0.001).view(-1).mean()
            metrics["actor_loss_" + str(id)] = actor_loss.mean().item()

            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                self.config.clip_grad,
                norm_type=self.config.grad_norm_type,
            )
            self.actor_optimizer.step()
        self.num_updates += 1
        with torch.no_grad():
            values = self.targ_critic(
            behavior_learning_infos.priors, behavior_learning_infos.deterministics, eval=False).mean
            predicted_rewards = self.reward_predictor(
            behavior_learning_infos.priors, behavior_learning_infos.deterministics, eval = False
            ).rsample()

            lambda_values_no_grad = compute_lambda_values(
            predicted_rewards.detach(),
            values,
            continues,
            self.config.horizon_length,
            self.device,
            behavior_learning_infos.log_probs,
            self.config.lambda_,
            self.config.alpha
            )
        value_dist = self.critic(
            behavior_learning_infos.priors.detach()[:, :-1],
            behavior_learning_infos.deterministics.detach()[:, :-1],
    
            )
        value_loss = nn.MSELoss()(value_dist.mean.squeeze(-1), symlog(lambda_values_no_grad.squeeze(1)))#+torch.max(torch.ones_like(value_dist.mean), torch.abs((lambda_values.detach())-value_dist.mean)/value_dist.stddev.pow(2)).mean()
        metrics["critic_loss_" + str(id)] = value_loss.mean().item()

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            self.config.clip_grad,
            norm_type=self.config.grad_norm_type,
        )
        self.critic_optimizer.step()
        self.soft_update(self.targ_critic, self.critic)

