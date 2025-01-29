"""Code for DistillTrainer."""
import torch
import torch.nn as nn
from torch.nn.functional import gelu, softmax, kl_div, log_softmax, cross_entropy, margin_ranking_loss, logsigmoid, one_hot
from transformers import Seq2SeqTrainer
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from .others import get_logger
import torch.nn.functional as F
import numpy as np
from torch.distributions.normal import Normal

logger = get_logger(__name__)
import pdb
import json
import transformers

def serialize_inputs(inputs):
    serialized_inputs = {}
    for key, value in inputs.items():
        if isinstance(value, transformers.tokenization_utils_base.BatchEncoding):
            # Convert BatchEncoding to dictionary and then each tensor to list
            value_dict = value.to_dict()
            serialized_inputs[key] = {k: v.tolist() for k, v in value_dict.items()}
        elif hasattr(value, 'tolist'):
            # Convert tensors to lists
            serialized_inputs[key] = value.tolist()
        else:
            serialized_inputs[key] = value
    return serialized_inputs

class DistillTrainer(Seq2SeqTrainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        # serialized_inputs = serialize_inputs(inputs)
        # with open('inputs_dump.txt', 'w') as f:
        #     json.dump(serialized_inputs, f, indent=4)
        # print('finished save')

        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        #pdb.set_trace()
        if "base_target_dist" in inputs:
            base_target_dist = inputs.pop("base_target_dist")
            base_metric = inputs.pop("metric_ce")
        else:
            base_target_dist = None
            base_metric = None
            
        if "aligned_target_dist_0" in inputs:
            aligned_target_dist_0 = inputs.pop("aligned_target_dist_0")
            aligned_metric_0 = inputs.pop("metric_ce_aligned_0")
        else:
            aligned_target_dist_0 = None
            aligned_metric_0 = None

        if "aligned_target_dist_1" in inputs:
            aligned_target_dist_1 = inputs.pop("aligned_target_dist_1")
            aligned_metric_1 = inputs.pop("metric_ce_aligned_1")
        else:
            aligned_target_dist_1 = None
            aligned_metric_1 = None

        if "aligned_target_dist_2" in inputs:
            aligned_target_dist_2 = inputs.pop("aligned_target_dist_2")
            aligned_metric_2 = inputs.pop("metric_ce_aligned_2")
        else:
            aligned_target_dist_2 = None
            aligned_metric_2 = None

        if "aligned_target_dist_3" in inputs:
            aligned_target_dist_3 = inputs.pop("aligned_target_dist_3")
            aligned_metric_3 = inputs.pop("metric_ce_aligned_3")
        else:
            aligned_target_dist_3 = None
            aligned_metric_3 = None
        # print('aligned_target_dist_3',aligned_target_dist_3)
        # print('aligned_target_dist_2',aligned_target_dist_3)
        # print('aligned_target_dist_1',aligned_target_dist_3)
        # print('aligned_target_dist_0',aligned_target_dist_3)
        # print('aligned_metric_3',aligned_metric_3)
        # print('aligned_metric_2',aligned_metric_2)
        # print('aligned_metric_1',aligned_metric_1)
        # print('aligned_metric_0',aligned_metric_0)
        # exit()

        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if self.args.do_distill:
            batch_size, seq_len, vocab_size = outputs["logits"].size(0), outputs["logits"].size(1), outputs["logits"].size(2)
            align_reward_0 = (1 / torch.exp(torch.tensor(aligned_metric_0, dtype=torch.bfloat16))).to(loss.device) if aligned_target_dist_0 is not None else None
            align_reward_1 = (1 / torch.exp(torch.tensor(aligned_metric_1, dtype=torch.bfloat16))).to(loss.device) if aligned_target_dist_1 is not None else None
            align_reward_2 = (1 / torch.exp(torch.tensor(aligned_metric_2, dtype=torch.bfloat16))).to(loss.device) if aligned_target_dist_2 is not None else None
            align_reward_3 = (1 / torch.exp(torch.tensor(aligned_metric_3, dtype=torch.bfloat16))).to(loss.device) if aligned_target_dist_3 is not None else None
            base_reward = (1 / torch.exp(torch.tensor(base_metric, dtype=torch.bfloat16))).to(loss.device) if base_target_dist is not None else None
            if self.args.distill_greater_as_gt is True:
                if base_target_dist is None: ####not entered
                    align_reward_0_expanded = align_reward_0.unsqueeze(-1).unsqueeze(-1).expand_as(aligned_target_dist_0) if aligned_target_dist_0 is not None else None
                    align_reward_1_expanded = align_reward_1.unsqueeze(-1).unsqueeze(-1).expand_as(aligned_target_dist_1) if aligned_target_dist_1 is not None else None
                    align_reward_2_expanded = align_reward_2.unsqueeze(-1).unsqueeze(-1).expand_as(aligned_target_dist_2) if aligned_target_dist_2 is not None else None
                    align_reward_3_expanded = align_reward_3.unsqueeze(-1).unsqueeze(-1).expand_as(aligned_target_dist_3) if aligned_target_dist_3 is not None else None
                    if aligned_target_dist_0 is not None and aligned_target_dist_1 is not None:
                        #target_dist = torch.where(align_reward_0_expanded > align_reward_1_expanded, aligned_target_dist_0, aligned_target_dist_1)
                        first_comparison = torch.where(align_reward_0_expanded > align_reward_1_expanded, aligned_target_dist_0, aligned_target_dist_1)
                        target_dist = torch.where(align_reward_2_expanded > torch.max(align_reward_0_expanded, align_reward_1_expanded), aligned_target_dist_2, first_comparison)
                    elif aligned_target_dist_0 is not None:
                        target_dist = aligned_target_dist_0
                    elif aligned_target_dist_1 is not None:
                        target_dist = aligned_target_dist_1
                    elif aligned_target_dist_2 is not None:
                        target_dist = aligned_target_dist_2
                    elif aligned_target_dist_3 is not None:
                        target_dist = aligned_target_dist_3
                    else:
                        raise ValueError
                    if self.args.distill_loss_type == "ce":
                        loss_lm = cross_entropy(input=outputs["logits"].view(-1, vocab_size),
                                                target=target_dist.view(-1, vocab_size),
                                                reduction="none").view(batch_size, -1)
                    elif self.args.distill_loss_type == "kl":
                        loss_lm = kl_div(input=log_softmax(outputs["logits"], dim=-1),
                                         target=target_dist,
                                         log_target=False,
                                         reduction="none").sum(dim=-1)
                    loss_lm = (loss_lm * inputs["attention_mask"]).sum() / inputs["attention_mask"].sum()
                    if self.args.distill_greater_as_gt_type == "hard":
                        loss = self.args.lm_loss_weight * loss + (1.0 - self.args.lm_loss_weight) * loss_lm
                    elif self.args.distill_greater_as_gt_type == "hard_and_decay":
                        decay_lm_loss_weight = self.args.lm_loss_weight + (1.0 - self.args.lm_loss_weight) * (self.state.global_step / self.state.max_steps)
                        loss = decay_lm_loss_weight * loss + (1.0 - decay_lm_loss_weight) * loss_lm
                    elif self.args.distill_greater_as_gt_type == "soft":
                        max_reward = torch.max(torch.stack([align_reward_0, align_reward_1], dim=-1), dim=-1)[0]
                        assert batch_size == 1
                        loss = (1.0 - max_reward[0]) * loss + max_reward[0] * loss_lm
                    else:
                        raise NotImplementedError
                else: #entered
                    base_reward_expanded = base_reward.unsqueeze(-1).unsqueeze(-1).expand_as(base_target_dist) if base_target_dist is not None else None
                    align_reward_0_expanded = align_reward_0.unsqueeze(-1).unsqueeze(-1).expand_as(aligned_target_dist_0) if aligned_target_dist_0 is not None else None
                    align_reward_1_expanded = align_reward_1.unsqueeze(-1).unsqueeze(-1).expand_as(aligned_target_dist_1) if aligned_target_dist_1 is not None else None
                    align_reward_2_expanded = align_reward_2.unsqueeze(-1).unsqueeze(-1).expand_as(aligned_target_dist_2) if aligned_target_dist_2 is not None else None
                    align_reward_3_expanded = align_reward_3.unsqueeze(-1).unsqueeze(-1).expand_as(aligned_target_dist_3) if aligned_target_dist_3 is not None else None
                    target_dist_list = []
                    reward_list = []
                    if base_target_dist is not None:
                        target_dist_list.append(base_target_dist)
                        reward_list.append(base_reward_expanded)
                    if aligned_target_dist_0 is not None:
                        target_dist_list.append(aligned_target_dist_0)
                        reward_list.append(align_reward_0_expanded)
                    if aligned_target_dist_1 is not None:
                        target_dist_list.append(aligned_target_dist_1)
                        reward_list.append(align_reward_1_expanded)
                    if aligned_target_dist_2 is not None:
                        target_dist_list.append(aligned_target_dist_2)
                        reward_list.append(align_reward_2_expanded)
                    if aligned_target_dist_3 is not None:
                        target_dist_list.append(aligned_target_dist_3)
                        reward_list.append(align_reward_3_expanded)

                    stacked_dists = torch.stack(target_dist_list, dim=-1)
                    stacked_rewards = torch.stack(reward_list, dim=-1)

                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    moe = MOE(num_experts=5, input_features=5).to(device).to(torch.bfloat16)

                    top_dists, top_rewards, moe_loss = moe(stacked_dists, stacked_rewards)
                    print('moe_loss',moe_loss)
                    #pdb.set_trace()
                    max_reward_indices = torch.argmax(top_rewards, dim=-1, keepdim=True)
                    target_dist = torch.gather(top_dists, -1, max_reward_indices).squeeze(-1)
                    #######################

                    # #########org##########
                    # stacked_rewards = torch.stack(reward_list, dim=-1)
                    # max_reward_indices = torch.argmax(
                    #     stacked_rewards, dim=-1, keepdim=True
                    # )
                    # target_dist = torch.gather(
                    #     stacked_dists, -1, max_reward_indices
                    # ).squeeze(-1)


                    if self.args.distill_loss_type == "ce":
                        loss_lm = cross_entropy(input=outputs["logits"].view(-1, vocab_size),
                                                target=target_dist.view(-1, vocab_size),
                                                reduction="none").view(batch_size, -1)
                    elif self.args.distill_loss_type == "kl":
                        loss_lm = kl_div(input=log_softmax(outputs["logits"], dim=-1),
                                         target=target_dist,
                                         log_target=False,
                                         reduction="none").sum(dim=-1)
                    loss_lm = (loss_lm * inputs["attention_mask"]).sum() / inputs["attention_mask"].sum()
                    if self.args.distill_greater_as_gt_type == "hard":
                        loss = self.args.lm_loss_weight * loss + (1.0 - self.args.lm_loss_weight) * loss_lm    #lm_loss_weight = 0.9
                    elif self.args.distill_greater_as_gt_type == "hard_and_decay":
                        decay_lm_loss_weight = self.args.lm_loss_weight + (1.0 - self.args.lm_loss_weight) * (self.state.global_step / self.state.max_steps)
                        loss = decay_lm_loss_weight * loss + (1.0 - decay_lm_loss_weight) * loss_lm
                    elif self.args.distill_greater_as_gt_type == "soft":
                        max_reward = torch.max(torch.stack([base_reward, align_reward_0, align_reward_1], dim=-1), dim=-1)[0]
                        assert batch_size == 1
                        loss = (1.0 - max_reward[0]) * loss + max_reward[0] * loss_lm
                    else:
                        raise NotImplementedError
                    
                    ###############add noise loss
                    loss = loss + 0.5 * moe_loss


            elif self.args.distill_weighted_as_gt is True:
                if base_target_dist is not None and aligned_target_dist_0 is not None and aligned_target_dist_1 is not None:
                    weights = torch.stack([base_reward, align_reward_0, align_reward_1], dim=1)
                    normalized_weights = torch.softmax(weights, dim=1)
                    weighted_label = normalized_weights[:, 0].unsqueeze(1).unsqueeze(2) * base_target_dist + normalized_weights[:, 1].unsqueeze(1).unsqueeze(2) * aligned_target_dist_0 + normalized_weights[:, 2].unsqueeze(1).unsqueeze(2) * aligned_target_dist_1
                elif aligned_target_dist_0 is not None and aligned_target_dist_1 is not None:
                    weights = torch.stack([align_reward_0, align_reward_1], dim=1)
                    normalized_weights = torch.softmax(weights, dim=1)
                    weighted_label = normalized_weights[:, 0].unsqueeze(1).unsqueeze(2) * aligned_target_dist_0 + normalized_weights[:, 1].unsqueeze(1).unsqueeze(2) * aligned_target_dist_1
                elif base_target_dist is not None and aligned_target_dist_0 is not None:
                    weights = torch.stack([base_reward, align_reward_0], dim=1)
                    normalized_weights = torch.softmax(weights, dim=1)
                    weighted_label = normalized_weights[:, 0].unsqueeze(1).unsqueeze(2) * base_target_dist + normalized_weights[:, 1].unsqueeze(1).unsqueeze(2) * aligned_target_dist_0
                elif base_target_dist is not None and aligned_target_dist_1 is not None:
                    weights = torch.stack([base_reward, align_reward_1], dim=1)
                    normalized_weights = torch.softmax(weights, dim=1)
                    weighted_label = normalized_weights[:, 0].unsqueeze(1).unsqueeze(2) * base_target_dist + normalized_weights[:, 1].unsqueeze(1).unsqueeze(2) * aligned_target_dist_1
                else:
                    raise ValueError
                if self.args.distill_loss_type == "ce":
                    loss_lm = cross_entropy(input=outputs["logits"].view(-1, vocab_size),
                                            target=weighted_label.view(-1, vocab_size),
                                            reduction="none").view(batch_size, -1)
                elif self.args.distill_loss_type == "kl":
                    loss_lm = kl_div(input=log_softmax(outputs["logits"], dim=-1),
                                     target=weighted_label,
                                     log_target=False,
                                     reduction="none").sum(dim=-1)
                else:
                    raise NotImplementedError
                loss_lm = (loss_lm * inputs["attention_mask"]).sum() / inputs["attention_mask"].sum()
                if self.args.distill_weighted_as_gt_type == "hard":
                    loss = self.args.lm_loss_weight * loss + (1.0 - self.args.lm_loss_weight) * loss_lm
                elif self.args.distill_weighted_as_gt_type == "hard_and_decay":
                    decay_lm_loss_weight = self.args.lm_loss_weight + (1.0 - self.args.lm_loss_weight) * (self.state.global_step / self.state.max_steps)
                    loss = decay_lm_loss_weight * loss + (1.0 - decay_lm_loss_weight) * loss_lm
                elif self.args.distill_weighted_as_gt_type == "soft":
                    mean_reward = weights.mean(dim=1)
                    assert batch_size == 1
                    loss = (1.0 - mean_reward[0]) * loss + mean_reward[0] * loss_lm
                else:
                    raise NotImplementedError
            else:
                loss = self.args.lm_loss_weight * loss

        return (loss, outputs) if return_outputs else loss



class MOE(torch.nn.Module):
    def __init__(self, num_experts, input_features, log_interval=10):
        super().__init__()
        self.gating_network_mlp = GatingNetwork_MLP(num_experts, input_features)
        self.log_interval = log_interval
        self.iteration_counter = 0
    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def forward(self, target_dists, rewards):
        ################## 1) Flatten & gate  ########################
        rewards_squashed = rewards.view(-1, rewards.shape[-1])    #rewards ([1, 2048, 32000, 4]), rewards_sum.shape ([65536000, 4])
        current_batch_size = rewards.shape[0]
        #logits = self.gating_network_logit(rewards_squashed)  # ([65536000, 4])
        logits = self.gating_network_mlp(rewards_squashed)  # use MLP as gate
        gating_probs = nn.functional.softmax(logits, dim=1).type_as(rewards)

        ################## 2) Dynamic selection based on threshold  ########################
        threshold = 0.15
        mask = gating_probs > threshold  # Boolean mask

        # Compute the number of selected experts
        num_selected = mask.sum(dim=1)

        # Ensure at least one expert is selected per sample
        no_expert_selected = num_selected == 0
        if no_expert_selected.any():
            max_probs, max_indices = gating_probs.max(dim=1)
            mask[no_expert_selected, max_indices[no_expert_selected]] = True
            num_selected = mask.sum(dim=1)  # Recompute num_selected after updating mask

        # # Print num_selected during training

        # num_selected_cpu = num_selected.detach().cpu()
        # avg_num_selected = num_selected_cpu.float().mean()
        # min_num_selected = num_selected_cpu.min()
        # max_num_selected = num_selected_cpu.max()
        # print(f"Iteration {self.iteration_counter}: "
        #         f"Avg num_selected={avg_num_selected.item()}, "
        #         f"Min={min_num_selected.item()}, "
        #         f"Max={max_num_selected.item()}")

        ################## 3) Normalize only selected probabilities ########################
        selected_probs = gating_probs * mask.float()
        prob_sums = selected_probs.sum(dim=1, keepdim=True)
        normalized_probs = selected_probs / (prob_sums + 1e-8)  # Add epsilon to prevent division by zero

        # Reshape to original dimensions
        reshaped_probs = normalized_probs.view(current_batch_size, 2048, 32000, -1)
        reshaped_mask = mask.float().view(current_batch_size, 2048, 32000, -1)

        ################## 4) Weighted sum of the experts' outputs  ########################
        weighted_outputs_target = target_dists * reshaped_probs * reshaped_mask
        weighted_outputs_rewards = rewards * reshaped_probs * reshaped_mask

        # Sum over experts dimension
        top_n_dists = weighted_outputs_target.sum(dim=-1, keepdim=True)
        top_n_combined_rewards = weighted_outputs_rewards.sum(dim=-1, keepdim=True)

        # Check for NaNs
        if torch.isnan(top_n_dists).any() or torch.isnan(top_n_combined_rewards).any():
            print("NaN detected in MoE outputs")


        ################## 5) Compute feedback loss (Eq. 13) ########################
        importance = normalized_probs.sum(0)
        moe_loss = self.cv_squared(importance)

        return top_n_dists, top_n_combined_rewards, moe_loss

class GatingNetwork_MLP(nn.Module):
    def __init__(self, num_experts, input_features):
        super(GatingNetwork_MLP, self).__init__()
        self.gate1 = nn.Linear(input_features, input_features * 2)
        self.gate2 = nn.Linear(input_features * 2, input_features)
        self.gate3 = nn.Linear(input_features, num_experts)

        # # Initialize weights using Xavier uniform initialization
        # nn.init.xavier_uniform_(self.gate1.weight)
        # nn.init.xavier_uniform_(self.gate2.weight)
        # nn.init.xavier_uniform_(self.gate3.weight)

    def forward(self, x):
        x = gelu(self.gate1(x))  # Using GELU here
        x = gelu(self.gate2(x))  # Using GELU again for the second layer
        x = self.gate3(x)  # No activation before the output
        return x

