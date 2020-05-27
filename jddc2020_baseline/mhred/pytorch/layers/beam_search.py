import torch
from utils import EOS_ID


class Beam(object):
    def __init__(self, batch_size, hidden_size, vocab_size, beam_size, max_unroll, batch_position):
        """Beam class for beam search"""
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.beam_size = beam_size
        self.max_unroll = max_unroll

        # batch_position [batch_size]
        #   [0, beam_size, beam_size * 2, .., beam_size * (batch_size-1)]
        #   Points where batch starts in [batch_size x beam_size] tensors
        #   Ex. position_idx[5]: when 5-th batch starts
        self.batch_position = batch_position

        self.log_probs = list()  # [(batch*k, vocab_size)] * sequence_length
        self.scores = list()  # [(batch*k)] * sequence_length
        self.back_pointers = list()  # [(batch*k)] * sequence_length
        self.token_ids = list()  # [(batch*k)] * sequence_length
        # self.hidden = list()  # [(num_layers, batch*k, hidden_size)] * sequence_length

        self.metadata = {
            'inputs': None,
            'output': None,
            'scores': None,
            'length': None,
            'sequence': None,
        }

    def update(self, score, back_pointer, token_id):  # , h):
        """Append intermediate top-k candidates to beam at each step"""

        # self.log_probs.append(log_prob)
        self.scores.append(score)
        self.back_pointers.append(back_pointer)
        self.token_ids.append(token_id)
        # self.hidden.append(h)

    def backtrack(self):
        """Backtracks over batch to generate optimal k-sequences

        Returns:
            prediction ([batch, k, max_unroll])
                A list of Tensors containing predicted sequence
            final_score [batch, k]
                A list containing the final scores for all top-k sequences
            length [batch, k]
                A list specifying the length of each sequence in the top-k candidates
        """
        prediction = list()

        # import ipdb
        # ipdb.set_trace()
        # Initialize for length of top-k sequences
        length = [[self.max_unroll] * self.beam_size for _ in range(self.batch_size)]

        # Last step output of the beam are not sorted => sort here!
        # Size not changed [batch size, beam_size]
        top_k_score, top_k_idx = self.scores[-1].topk(self.beam_size, dim=1)

        # Initialize sequence scores
        top_k_score = top_k_score.clone()

        n_eos_in_batch = [0] * self.batch_size

        # Initialize Back-pointer from the last step
        # Add self.position_idx for indexing variable with batch x beam as the first dimension
        # [batch x beam]
        back_pointer = (top_k_idx + self.batch_position.unsqueeze(1)).view(-1)

        for t in reversed(range(self.max_unroll)):
            # Reorder variables with the Back-pointer
            # [batch x beam]
            token_id = self.token_ids[t].index_select(0, back_pointer)

            # Reorder the Back-pointer
            # [batch x beam]
            back_pointer = self.back_pointers[t].index_select(0, back_pointer)

            # Indices of ended sequences
            # [< batch x beam]
            eos_indices = self.token_ids[t].data.eq(EOS_ID).nonzero()

            # For each batch, every time we see an EOS in the backtracking process,
            # If not all sequences are ended
            #    lowest scored survived sequence <- detected ended sequence
            # if all sequences are ended
            #    lowest scored ended sequence <- detected ended sequence
            if eos_indices.dim() > 0:
                # Loop over all EOS at current step
                for i in range(eos_indices.size(0) - 1, -1, -1):
                    # absolute index of detected ended sequence
                    eos_idx = eos_indices[i, 0].item()

                    # At which batch EOS is located
                    batch_idx = eos_idx // self.beam_size
                    batch_start_idx = batch_idx * self.beam_size

                    # if n_eos_in_batch[batch_idx] > self.beam_size:

                    # Index of sequence with lowest score
                    _n_eos_in_batch = n_eos_in_batch[batch_idx] % self.beam_size
                    beam_idx_to_be_replaced = self.beam_size - _n_eos_in_batch - 1
                    idx_to_be_replaced = batch_start_idx + beam_idx_to_be_replaced

                    # Replace old information with new sequence information
                    back_pointer[idx_to_be_replaced] = self.back_pointers[t][eos_idx].item()
                    token_id[idx_to_be_replaced] = self.token_ids[t][eos_idx].item()
                    top_k_score[batch_idx,
                                beam_idx_to_be_replaced] = self.scores[t].view(-1)[eos_idx].item()
                    length[batch_idx][beam_idx_to_be_replaced] = t + 1

                    n_eos_in_batch[batch_idx] += 1

            # max_unroll * [batch x beam]
            prediction.append(token_id)

        # Sort and re-order again as the added ended sequences may change the order
        # [batch, beam]
        top_k_score, top_k_idx = top_k_score.topk(self.beam_size, dim=1)
        final_score = top_k_score.data

        for batch_idx in range(self.batch_size):
            length[batch_idx] = [length[batch_idx][beam_idx.item()]
                                 for beam_idx in top_k_idx[batch_idx]]

        # [batch x beam]
        top_k_idx = (top_k_idx + self.batch_position.unsqueeze(1)).view(-1)

        # Reverse the sequences and re-order at the same time
        # It is reversed because the backtracking happens in the reverse order
        # [batch, beam]

        prediction = [step.index_select(0, top_k_idx).view(
            self.batch_size, self.beam_size) for step in reversed(prediction)]

        # [batch, beam, max_unroll]
        prediction = torch.stack(prediction, 2)

        return prediction, final_score, length
