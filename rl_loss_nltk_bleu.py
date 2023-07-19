# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from torch import Tensor

from dataclasses import dataclass, field
from nltk.translate.bleu_score import sentence_bleu

@dataclass
class LabelSmoothedDualImitationCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )


@register_criterion("rl_loss_nltk_bleu", dataclass=LabelSmoothedDualImitationCriterionConfig)
class LabelSmoothedDualImitationCriterion(FairseqCriterion):
    def __init__(self, task, label_smoothing):
        super().__init__(task)
        self.label_smoothing = label_smoothing

    # Compute the reward - we need the model to decode the indices into strings
    def compute_reward(self, model, sample_source, sample_targets, sample_prob):
        rewards = []
        # Loop through the sample
        for i in range(len(sample_targets)):
            # Append each word to the list (so we have an array of words - i.e. a sentence)
            sample_sentence = [model.decoder.dictionary[j] for j in sample_source[i]]
            target_sentence = [model.decoder.dictionary[j] for j in sample_targets[i]]

            # We do not need the sentence as a string for NLTK
            # Make string of sentence, torchmetrics expects an array with 1 string of the whole sentence
            # sample_sentence = [' '.join(word for word in sample_sentence)]
            # target_sentence = [' '.join(word for word in target_sentence)]
            score = sentence_bleu(sample_sentence, target_sentence)

            # Append the reward for the whole sentence 
            for k in range(len(sample_targets[i])):
                rewards.append(score)
        
        rewards = torch.Tensor(rewards).cuda(sample_prob.get_device())
        return rewards
        

    def _compute_loss(
        self, model, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0
    ):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len

        policy_logprob: if there is some policy
            depends on the likelihood score as rewards.
        """

        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]

        # Compute softmax
        logits = F.softmax(outputs, dim=-1)

        # Compute sentence lengths
        target_lengths = torch.sum(masks, dim = -1).long().tolist()
        targets = targets.data.tolist()

        # Multinomial sampling from softmax and turning them back into probabilities (since it returns indices)
        sample_index = torch.multinomial(logits,1)
        sample_prob = torch.gather(logits, -1, sample_index)
        # Need the indices for the decoding the index to strings
        sample_index = sample_index.data.view(-1).tolist()

        # Turn whole sample into array with sentences (respectively) - i.e. [[sentence1], [sentence2], ...]
        sample_targets = utils.list_sample(targets, target_lengths)
        sample_source = utils.list_sample(sample_index, target_lengths)

        # Compute the reward (change function accordingly)
        rewards = self.compute_reward(model, sample_source, sample_targets, sample_prob)
        # Calculate loss - L = \sum( -log(P(Y|X)  R(y_hat, y) ) )
        loss_sample = torch.sum((-1 * torch.log(sample_prob).view(-1) * rewards),dim = 0)
        loss = loss_sample.div(len(targets))

        return {"name": name, "loss": loss, "factor": factor}

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        tgt_tokens, prev_output_tokens = sample["target"], sample["prev_target"]

        outputs = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens)
        losses = [] 

        for obj in outputs:
            if outputs[obj].get("loss", None) is None:
                if obj is 'word_ins':
                    _losses = self._compute_loss(
                        model,
                        outputs[obj].get("out"),
                        outputs[obj].get("tgt"),
                        outputs[obj].get("mask", None),
                    )

            losses += [_losses]

        loss = sum(l["loss"] for l in losses)

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }

        for l in losses:
            logging_output[l["name"]] = (
                utils.item(l["loss"].data / l["factor"])
                if reduce
                else l[["loss"]].data / l["factor"]
            )

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )
        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))

        metrics.log_scalar(
            "loss", loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )

        for key in logging_outputs[0]:
            if key[-5:] == "-loss":
                val = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(
                    key[:-5],
                    val / sample_size / math.log(2) if sample_size > 0 else 0.0,
                    sample_size,
                    round=3,
                )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
