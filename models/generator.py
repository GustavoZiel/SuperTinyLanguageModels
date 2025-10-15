"""Generator Base Wrapper"""

import torch

from trainers.utils import set_seed
from utils.logger import get_logger

logger = get_logger(__name__)


class StandardGenerator(torch.nn.Module):
    """Standard Generator Wrapper for GPT models"""

    def __init__(self, model, generate_cfg):
        """Initialize the model and the configuration"""
        super().__init__()
        self.model = model
        self.model = self.model.to(torch.device("cuda"))
        self.generate_config = generate_cfg

        set_seed(self.generate_config["seed"])

    def default_generate(self, input_text):
        """Generate text using the default generation method"""
        return self.generate(
            input_text,
            self.generate_config["max_new_tokens"],
            self.generate_config["temperature"],
            self.generate_config["top_k"],
        )

    def _format_messages(self, messages, steps_to_log):
        return "".join(message + "\n" for message in messages[:steps_to_log])

    @torch.no_grad()
    def evaluate(self, input_text, correct_answer, temperature=1.0, top_k=None):
        """Evaluate the log-likelihood of the correct answer given the prompt"""
        original_mode = self.model.training
        self.model.eval()

        try:
            idx = self.model.embedding_model.tokenize_input(
                input_string=input_text, add_eot=False, truncate=True
            )
            idx = torch.tensor(idx).unsqueeze(0).to(torch.device("cuda"))

            idx_correct = self.model.embedding_model.tokenize_input(
                input_string=correct_answer, add_eot=False, truncate=True
            )
            idx_correct = (
                torch.tensor(idx_correct).unsqueeze(0).to(torch.device("cuda"))
            )

            probs_correct = []
            for i in range(idx_correct.shape[1]):
                logits, model_input = self.model.inference(idx)
                # logits = logits / temperature
                # if top_k is not None:
                #     v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                #     # check for dim
                #     if len(v.size()) == 3:
                #         logits[logits < v[:, :, [-1]]] = -float("Inf")
                #     else:
                #         logits[logits < v[:, [-1]]] = -float("Inf")
                probs = torch.nn.functional.softmax(logits, dim=-1)
                probs_correct.append(probs[0, idx_correct[0, i]].item())
                idx = torch.cat((idx, idx_correct[:, i].unsqueeze(0)), dim=1)
                # print(
                #     f"Step {i + 1}, Correct token index: {idx_correct[0, i].item()}, Probability: {probs_correct[-1]:.4f}"
                # )

            probs = torch.tensor(probs_correct)
            probs = torch.clamp(probs, min=1e-12)  # avoid log(0)

            # Negative log-likelihood (sum and mean)
            # log_likelihood = torch.sum(torch.log(probs))
            # nll_sum = -log_likelihood
            nll_mean = -torch.mean(torch.log(probs))

            # Perplexity = exp(mean NLL)
            ppl = torch.exp(nll_mean)

            # print(f"Probabilities of correct tokens: {probs_correct}")
            # print(f"Negative log-likelihood (sum): {nll_sum.item():.4f}")
            # print(f"Negative log-likelihood (mean per token): {nll_mean.item():.4f}")
            # print(f"Perplexity: {ppl.item():.4f}")

            return [round(float(x), 4) for x in probs.tolist()], ppl.item()

        finally:
            self.model.train(original_mode)

    @torch.no_grad()
    def generate(self, input_text, max_new_tokens, temperature=1.0, top_k=None):
        """Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        # Save current training state and switch to eval mode
        original_mode = self.model.training
        # logger.info(f"Original model training mode: {'train' if original_mode else 'eval'}")
        self.model.eval()

        try:
            idx = self.model.embedding_model.tokenize_input(
                input_string=input_text, add_eot=False, truncate=True
            )
            # push to device
            idx = torch.tensor(idx).unsqueeze(0).to(torch.device("cuda"))

            messages = []
            for i_token in range(max_new_tokens):
                message = f"Step {i_token + 1}:\n"

                # forward the model to get the logits for the index in the sequence
                logits, model_input = self.model.inference(idx)

                # pluck the logits at the final step and scale by desired temperature
                logits = logits / temperature

                # logits have shape (b,t,v)
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    # check for dim
                    if len(v.size()) == 3:
                        logits[logits < v[:, :, [-1]]] = -float("Inf")
                    else:
                        logits[logits < v[:, [-1]]] = -float("Inf")

                # apply softmax to convert logits to (normalized) probabilities
                probs = torch.nn.functional.softmax(logits, dim=-1)

                # sample from the distribution
                # check if byte-level and if so, flatten
                if len(probs.size()) == 4:
                    B, S, S_c, H = probs.size()
                    probs = probs.view(B * S * S_c, H)
                    flattened = True
                else:
                    flattened = False

                # For every i_token, collect top_k token info and format messages
                top_k_probs, top_k_indices = torch.topk(probs, top_k)
                for i_idx, i_prob in zip(
                    top_k_indices.flatten(), top_k_probs.flatten()
                ):
                    decoded_token = self.model.embedding_model.decode(i_idx.view(1, -1))
                    message += f"Token: {decoded_token}, Probability: {i_prob.item():.4f}, Index: {i_idx.item()}\n"

                idx_next = torch.multinomial(probs, num_samples=1)

                # check if byte-level and if so, unflatten
                if flattened:
                    idx_next = idx_next.view(B, S)
                elif idx_next == self.model.embedding_model.eot_token:
                    break

                if flattened:
                    idx_next = idx_next.unsqueeze(0)
                idx = torch.cat((idx, idx_next), dim=1)

                messages.append(message)

            return self.model.embedding_model.decode(idx.tolist()), messages

        finally:
            # Always restore the original training state
            self.model.train(original_mode)
            # logger.info(f"Restored model training mode to: {'train' if original_mode else 'eval'}")

    def forward(self, x):
        """Call the underlying model"""
        return self.model(x)

    def embed(self, x):
        """Embed the input"""
        return self.model.embed(x)


def build_generator(model, generate_cfg):
    """Build the generator"""
    return StandardGenerator(model, generate_cfg)
