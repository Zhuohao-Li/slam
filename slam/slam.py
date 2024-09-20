import torch

def slice2d(x, start, end):
    return x[:, :, start:end, ...]

def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]

def slice1d(x, start, end):
    return x[:, start:end, ...]

DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}

class StartRecentKVCache:
    def __init__(self, start_size=4, recent_size=512, reward_size=128, num_heads=8, pi_phases=3, k_seq_dim=2, v_seq_dim=2):
        """
        Initialize the cache manager.

        Args:
            start_size (int): Size of the attention sink tokens to retain.
            recent_size (int): Size of the attention hysteresis tokens.
            reward_size (int): Size of the attention reward tokens.
            num_heads (int): Number of attention heads.
            pi_phases (int): Period for re-evaluating the reward tokens.
            k_seq_dim (int): The dimension along which key sequences are sliced.
            v_seq_dim (int): The dimension along which value sequences are sliced.
        """
        print(f"StartRecentKVCache: start_size={start_size}, recent_size={recent_size}, reward_size={reward_size}")
        self.start_size = start_size
        self.recent_size = recent_size
        self.reward_size = reward_size
        self.num_heads = num_heads
        self.pi_phases = pi_phases
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]
        
        # Cache state initialization
        self.attention_scores = []  # Store attention scores for reward computation
        self.reward_tokens = set()  # Track current reward tokens
        self.cache_size = start_size + recent_size + reward_size

    def __call__(self, past_key_values):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values
        
        # Select attention sink (initial tokens) and attention hysteresis (recent tokens)
        result = [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),  # Attention sink
                        self.k_slice(k, seq_len - self.recent_size, seq_len)  # Attention hysteresis
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),  # Attention sink
                        self.v_slice(v, seq_len - self.recent_size, seq_len)  # Attention hysteresis
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

        # Update reward tokens based on accumulated attention scores
        if self.attention_scores:
            self.update_reward_tokens(result)

        return result

    def update_attention_scores(self, attention_probs):
        """
        Update attention scores for reward tokens.
        
        Args:
            attention_probs (Tensor): Attention probabilities of shape [num_heads, seq_len, seq_len].
        """
        self.attention_scores.append(attention_probs.sum(dim=0))  # Sum over heads to get scores per token
        if len(self.attention_scores) > self.pi_phases:
            self.attention_scores.pop(0)  # Keep scores within the pi_phases window

    def update_reward_tokens(self, past_key_values):
        """
        Update reward tokens based on accumulated attention scores.
        
        Args:
            past_key_values: The current state of key-value pairs in the cache.
        """
        accumulated_scores = torch.stack(self.attention_scores, dim=0).mean(dim=0)  # [seq_len] average scores
        threshold = self.compute_threshold(accumulated_scores)
        reward_candidates = (accumulated_scores >= threshold).nonzero(as_tuple=True)[0]

        # Maintain the top lr tokens as reward tokens
        if len(reward_candidates) > self.reward_size:
            reward_candidates = reward_candidates[:self.reward_size]

        # Update reward token set
        self.reward_tokens = set(reward_candidates.tolist())

        # Evict tokens if necessary
        self.evict_for_space(past_key_values, len(self.reward_tokens))

    def compute_threshold(self, accumulated_scores):
        """
        Compute the threshold Ti for reward tokens.
        
        Args:
            accumulated_scores (Tensor): Accumulated attention scores for each token.
        
        Returns:
            float: The computed threshold Ti.
        """
        seq_len = accumulated_scores.size(0)
        threshold = (accumulated_scores.mean().item()) / self.num_heads
        return threshold

    def evict_for_space(self, past_key_values, num_coming):
        """
        Evict tokens to manage cache space when new tokens are coming.

        Args:
            past_key_values: The current key-value cache state.
            num_coming (int): Number of new tokens to accommodate.
        
        Returns:
            Updated key-value cache state after eviction.
        """
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(
                            k, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(
                            v, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

    def evict_range(self, past_key_values, start, end):
        """
        Evict tokens in a specified range from the cache.

        Args:
            past_key_values: The current key-value cache state.
            start (int): Start index of the range.
            end (int): End index of the range.
        
        Returns:
            Updated key-value cache state after eviction.
        """
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        assert start <= end and end <= seq_len
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, start),
                        self.k_slice(k, end, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, start),
                        self.v_slice(v, end, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]
