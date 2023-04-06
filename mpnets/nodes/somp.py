import itertools
from mpnets.layers.growing_layer import GrowingLayer
from mpnets.layers.base_dynamic_multi_input_encoder import DynamicMultiInputProcessor
from mpnets.layers.leaky_spiking_bucket import LeakySpikingBucket
from mpnets.layers.torch_dynamic_multi_input_encoder import TorchDynamicMultiInputEncoder
from mpnets.nodes.node import Node
from mpnets.utils.misc import select
from mpnets.utils.nn import soft_v
import torch
import torch.nn as nn
import torch.optim as optim


class SOMP(Node):

    use_batch_norm = False
    bias = None
    leaky_spiking_bucket_hparams = {}
    lr = 0.001
    momentum = 0.9
    ETA_DEFAULT = 1.0
    eta_STDP = ETA_DEFAULT
    STDP_frequencies = [1, 2, 4, 8, 16]
    eta_CD = ETA_DEFAULT
    eta_SP = ETA_DEFAULT
    eta_IP = ETA_DEFAULT
    eta_temporal_VIC_var = ETA_DEFAULT
    eta_temporal_VIC_covar = ETA_DEFAULT
    eta_temporal_VIC_invar = ETA_DEFAULT
    eta_L2W = ETA_DEFAULT
    eta_MR = ETA_DEFAULT
    eta_SR = ETA_DEFAULT
    eta_SL = ETA_DEFAULT
    eta_LWTA = ETA_DEFAULT
    eta_soft_WTA = ETA_DEFAULT
    eta_oja = ETA_DEFAULT
    eta_L1 = ETA_DEFAULT
    eta_WC = ETA_DEFAULT
    eta_VIC_input_var = ETA_DEFAULT
    eta_VIC_input_covar = ETA_DEFAULT
    eta_VIC_input_invar = ETA_DEFAULT
    eta_FA = ETA_DEFAULT
    eta_FF = ETA_DEFAULT
    enable_STDP = True
    enable_CD = True
    enable_SP = True
    enable_IP = True
    enable_temporal_VIC = True
    enable_weight_reg = True
    enable_MR = True
    enable_SR = True
    enable_SL = True
    enable_LWTA = True
    enable_soft_WTA = True
    enable_oja = True
    enable_L1 = True
    enable_WC = True
    enable_VIC_input = True
    enable_FA = True
    enable_FF = True

    def __init__(
        self,
        output_dim,
        head=None,
        bu_input_sizes: dict[str, int] = None,
        td_feedback_sizes: dict[str, int] = None,
        tags: list[str] = [],
        **hparams,
    ):
        super().__init__(tags=tags)
        self.__dict__.update(hparams)

        # inference-related
        self.dim = output_dim
        if self.bias:
            self.bias = nn.Parameter(torch.randn(self.dim))
        if self.batch_norm:
            self.batch_norm = nn.BatchNorm(self.dim)
        self.bu_input_encoder = TorchDynamicMultiInputEncoder(
            encoder_factory=lambda example: nn.Linear(
                example.size[-1], self.dim, bias=self.bias
            )
        )
        self.head = head or LeakySpikingBucket(self.leaky_spiking_bucket_hparams)

        # learning-related
        if self.enable_STDP:
            self.STDP_frequencies = sorted(self.STDP_frequencies)
        if self.enable_VIC_input:
            self.bu_input_projectors = nn.ModuleDict({})
        if self.enable_FA:
            self.td_feedback_encoder = TorchDynamicMultiInputEncoder(
                encoder_factory=lambda example: nn.Linear(
                    example.size[-1], self.dim, bias=False
                )
            )

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, momentum=self.momentum)

        if bu_input_sizes is not None:
            for name, size in bu_input_sizes.items():
                self.add_bu_input(name, torch.randn(1, size))
        if td_feedback_sizes is not None:
            for name, size in td_feedback_sizes.items():
                self.add_td_feedback(name, torch.randn(1, size))

    @property
    def bu_input_keys(self) -> list[str]:
        return self.bu_input_encoder.encoders.keys()

    @property
    def td_feedback_keys(self) -> list[str]:
        return self.td_feedback_encoder.encoders.keys()

    def add_bu_input(self, name, example):
        self.bu_input_encoder.add_input(name, example)
        self.optimizer.add_param_group(
            {"params": self.bu_input_encoder.encoders[name].parameters()}
        )
        if self.enable_VIC_input:
            self.bu_input_projectors[name] = nn.Linear(
                example.size[-1], self.dim, bias=False
            )
            self.optimizer.add_param_group(
                {"params": self.bu_input_projectors[name].parameters()}
            )

    def add_td_feedback(self, name, example):
        self.td_feedback_encoder.add_input(name, example)
        self.optimizer.add_param_group(
            {"params": self.td_feedback_encoder.encoders[name].parameters()}
        )

    def __call__(self, *, reward=None, **all_inputs):

        bu_inputs = select(all_inputs, self.bu_input_keys)
        Z = self.bu_encoder(**bu_inputs)
        if self.bias:
            Z = Z + self.bias
        if self.enable_batch_norm:
            Z = self.batch_norm(Z)
        Y = self.head(Z)

        if self.training:

            # TODO: I need to update these rules, since the weights are now buried in the input_encoders. Also make sure it works for inputs shaped (batch..., time, dim)

            # Unsupervised update rules
            loss = 0.0

            if self.enable_STDP:
                # 1. STDP: spike-timing dependent plasticity
                r"""
                **1. Spike Timing Dependant Plasticity (STDP)** makes synapse $w^{ji}$ decrease when $y_i$ activates before $x_j$ (since the connection must've not been important) and increase when $x_i$ activates after $y_j$ (since the connection made a significant contribution). It has no effect when one of the values is $0$ and an inverse effect when one value is positive and the other is negative (or vice versa):

                $$\Delta w_{STDP,ji} = \hat{x}_{t-1,j} \hat{y}_{t,i} - \hat{x}_{t,j} \hat{y}_{t-1,i}$$

                with $\hat{\cdot{}}$ representing $\cdot{} \div \alpha$ where $\alpha$ is corresponding to the layer that produced the activation in heterogeneous architectures. In the extended temporal case, STDP is expressed using Python-style slicing as:
                $$\Delta w_{STDP,ji} = \hat{x}_{:-1,j} \hat{y}_{1:,i} - \hat{x}_{1:,j} \hat{y}_{:-1,i}$$

                (Note: we also compute STDP for frequencies other than $1$ to capture longer-term dependencies.)
                """
                for k in inputs.keys():

                    assert inputs[k].shape[-2] >= 2 * max(
                        self.STDP_frequencies
                    ), f"STDP requires at least twice as many time steps as the STDP maximum frequency: {max(self.STDP_frequencies)}"
                    assert (
                        inputs[k].shape[-1] % len(self.STDP_frequencies) == 0
                    ), f"STDP requires the number of input dimensions to be cleanly divisible by the number of frequencies: {len(self.STDP_frequencies)}"

                    output_partition_size = Y.shape[-1] // len(self.STDP_frequencies)
                    dW_stdp_subsets = []
                    for i, freq in enumerate(self.STDP_frequencies):
                        output_subset = inputs[k][
                            ...,
                            i * output_partition_size : (i + 1) * output_partition_size,
                        ]
                        prev_input = inputs[k][..., :-freq]
                        next_input = inputs[k][..., freq:]
                        prev_output = output_subset[..., :-freq]
                        next_output = output_subset[..., freq:]
                        dW_stdp_input_subset = torch.einsum(
                            "bti,btj->btij", prev_input, next_output
                        ) - torch.einsum("bti,btj->btij", next_input, prev_output)
                    dW_stdp = torch.cat(
                        dW_stdp_subsets, dim=-1
                    )  # cat([input_size, output_size//len(self.frequencies_STDP)]) -> [input_size, output_size]
                    self.processed_inputs[k].backward(-self.lr_stdp * dW_stdp)

            if self.enable_CD:
                # 2. CD: covariance decay
                r"""
                **2. Covariance Decay (CD)** makes synapse $w^{ji}$ decay or grow in nonlinear proportion to the absolute covariance between $x_j$ and $y_i$ computed via $\beta_{cd}$ rolling means:

                $$\sigma_{ji} = \Cov(\hat{x}_{:-1,j} \hat{y}_{1:,i}) = \E_{\beta_{cd}}{[ \hat{x}_{t-1,j}\hat{y}_{t,i} ]} - \E_{\beta_{cd}}{[ \hat{x}_{t,j} ]} \E_{\beta_{cd}}{[ \hat{y}_{t,i} ]}$$

                The absolute of this covariance factor $|\sigma|$ is next scaled around $1$ by a learned parameter $a_{cd}$ giving the covariance decay coefficient: $$c_{ji} = a_{cd}(|\sigma_{ji}|-1)+1$$ Ideally, the covariance decay coefficient $c_{ji}$ could be applied to directly onto its corresponding weight as $w_{ji} \leftarrow c_{ji}w_{ji}$. However to remain compatible with gradient-based update paradigms, this coefficient is finally expressed as a weight increment to be compounded with other gradients:

                $$\Delta w_{CD,ji} = (1-c_{ji}) w_{ji}$$
                """
                for k in inputs.keys():
                    xhat = inputs[k] / torch.mean(inputs[k], dim=-2, keepdim=True)
                    yhat = Y / torch.mean(Y, dim=-2, keepdim=True)
                    cov = (
                        torch.mean(
                            xhat[..., :-1, :, None] * yhat[..., 1:, None, :], dim=-3
                        )
                        - torch.mean(xhat, dim=-2)[..., :, :, None]
                        * torch.mean(yhat, dim=-2)[..., :, None, :]
                    )
                    coef = self.a_cd * torch.abs(cov) - self.a_cd + 1
                    dW_cd = -(1 - coef) * self.input_processors[k]
                    self.processed_inputs[k].backward(-self.lr_cd * dW_cd)

            if self.enable_SP:
                # 3. SP: structural plasticity
                r"""
                **3. Structural Plasticity (SP)** randomly adds small values synapses between unconnected neurons by Bernoili probability factor $p_{SP}=\frac{a_{sp}}{N_x N_y}$ which scales inversely quatratically with respect to the number of input $N_x$ and output $N_y$ dimensions.

                $$\Delta w_{SP,ji} = d, \; \; \;d \sim \mathcal{B}(\ \cdot\ ; p=\small{\frac{a_{sp}}{N_x N_y}})$$

                $a_{sp}$ is made differentiable by the reparametrization trick
                """
                for k in inputs.keys():
                    p_sp = self.a_sp / (inputs[k].shape[-1] * self.dim)
                    # TODO: check this why is this variable not used?
                    dW_sp = torch.bernoulli(self.p_sp) * self.input_processors[k].mean()
                    self.processed_inputs[k].backward(-self.lr_sp * dW_sp)

            if self.enable_IP:
                # 4. IP: intrinsic plasticity
                r"""
                **4. Intrinsic plasticity (IP)** homeostatically shift the inputs maintain a mean firing rate $H_{IP} \sim \mathcal{N}(\mu_{IP}=0.1, \Sigma_{IP}^2 = 0)$

                $$\Delta \mu_{\zeta} = \eta_{IP}[x(t) - H_{IP}]$$
                """
                self.bias.backward(-self.lr_ip * (Y - 0.1))

            if self.enable_temporal_VIC:
                # 5. VIC: variance-invariance-covariance
                r"""
                **5. Variance-Invariance-Covariance Sequence Regularization (VIC)** aims to build representations that are progressively insensitive to time by applying the following regularization penalties:

                - local temporal element invariance: Assuming that sequence elements in a local neighborhood represent the same information, maximize the similarity $s$ between $y_{\dots,t,:}$ and $y_{\dots,t-1,:},y_{\dots,t-2,:},\dots$ assigning exponential weight to nearer elements

                - batch
                """
                sim = torch.sum(Y[..., :-1] * Y[..., 1:], dim=-1) / (
                    torch.norm(Y[..., :-1], dim=-1) * torch.norm(Y[..., 1:], dim=-1)
                )
                sim = sim.mean()

                y_mean = torch.mean(Y, dim=-2, keepdim=True)
                y_hat = Y - y_mean
                covar = torch.mean(
                    y_hat[..., :-1, :, None] * y_hat[..., 1:, None, :], dim=-3
                )
                covar = torch.sum(covar**2)

                invar = torch.mean((Y[..., :-1] - Y[..., 1:]) ** 2)

                loss += (
                    self.lambda_vic_sim * sim
                    + self.lambda_vic_covar * covar
                    + self.lambda_vic_invar * invar
                )

            if self.enable_weight_reg:
                # 6. Elementwise Squared Weight Regularization
                r"""
                **6. Elementwise Squared Magnitude regularization** prevents weights from growing excessively large unless they serve a meaningful statistical purpose. Formally,

                $$ \mathcal{L}_{L2,W} = \frac{1}{2} \sum_{ji} w_{ji}^2$$
                $$ \mathcal{L}_{L2,b} = \frac{1}{2} \sum_{i} b_{i}^2$$
                """
                for k in inputs.keys():
                    l2_W = 0.5 * sum(
                        torch.sum(param**2)
                        for param in self.input_processors[k].parameters()
                    )
                    self.add_loss(self.eta_L2W * l2_W)
                l2_b = 0.5 * torch.sum(self.bias**2)
                loss += self.eta_L2b * l2_b

            if self.enable_MR:
                # 7. Mean regularization
                r"""
                **7. Mean Regularization (MR)** aim to preserve a mean activation across between SOMPCells by:

                $$\mathcal{L}_{MR} = (\sum \hat{y}-\mu_{MR})^2$$
                """
                loss_MR = (Y.sum() - self.mu_MR) ** 2
                loss += self.eta_MR * loss_MR

            if self.enable_SR:
                # 8. Sparsity regularization (applied only to sparse activation functions)
                r"""
                **8. Sparsity Regularization (SR)** (which applies only to sparse activation `SOMPCell`'s) aims to tune sparsity by penalizing activation KL-divergence from a Bernoilii distribution

                $$\mathcal{L}_{SR} = \sum \biggl[ |\hat{y}| \log { \frac{ |\hat{y}| }{ a_{SR} } } - (1 - \hat{y})| \log { \frac{ 1 - |\hat{y}| }{ 1 - a_{SR} } } \biggr]$$
                """
                if self.activation == "sparse":
                    kl_SR = Y * torch.log(Y / self.a_SR) - (1 - Y) * torch.log(
                        (1 - Y) / (1 - self.a_SR)
                    )
                    loss_SR = kl_SR.sum()
                    loss += self.eta_SR * loss_SR

            if self.enable_LWTA:
                # 9. Local Winner-Take-All (LWTA) competition
                r"""
                **9. Local Winner-Take-All (LWTA) competition**: This rule implements a competition mechanism among the output neurons, ensuring that only the neuron with the highest activation is selected as the winner. The winner's output is set to 1, while all other outputs are set to 0. The weight update is then calculated based on the winner's input using the following equation:

                $$dW_{LWTA} = \frac{1}{n}\sum_{i}^{n} (y_i^{LWTA}\otimes x_i)$$

                where $y_i^{LWTA}$ is the output of the $i$th neuron after the LWTA competition, $x_i$ is the input to the $i$th neuron, $\otimes$ is the outer product operator, and $n$ is the number of output neurons.

                """
                _, max_indices = torch.max(Y, dim=-1, keepdim=True)
                Y_lwta = torch.zeros_like(Y)
                Y_lwta.scatter_(-1, max_indices, 1.0)
                dW_lwta = (Y_lwta[..., None, :] * inputs[k]).mean(dim=-2)
                self.processed_inputs[k].backward(-self.lr_lwta * dW_lwta)

            if self.enable_soft_WTA:
                # 10. Soft Winner-Take-All (WTA) competition
                r"""
                **10. Soft Winner-Take-All (WTA) competition**: This rule is similar to LWTA, but instead of a binary activation, it uses a softmax function to produce a probability distribution over the output neurons. The weight update is then calculated based on this probability distribution using the following equation:

                $$dW_{WTA} = \frac{1}{n}\sum_{i}^{n} (y_i^{WTA}\otimes x_i)$$

                where $y_i^{WTA}$ is the output probability of the $i$th neuron after the WTA competition, $x_i$ is the input to the $i$th neuron, $\otimes$ is the outer product operator, and $n$ is the number of output neurons.
                """
                Y_exp = torch.exp(Y / self.soft_wta_temperature)
                Y_softmax = Y_exp / Y_exp.sum(dim=-1, keepdim=True)
                dW_wta = torch.einsum("...i,...j->...ij", Y_softmax, inputs[k]).mean(
                    dim=-2
                )
                self.processed_inputs[k].backward(-self.lr_soft_wta * dW_wta)

            if self.enable_oja:
                # 11. Oja's rule
                r"""
                **11. Oja's rule**: This rule implements a learning rule for the input weights based on the output neuron's activation and the input signal. The weight update is calculated as the outer product of the output and input signals, with a correction term to ensure that the weight vector remains orthogonal to the output neuron. The weight update rule can be expressed as:

                $$dW_{oja} = y\otimes(x - yW_{oja})$$

                where $y$ is the output of the neuron, $x$ is the input to the neuron, $W_{oja}$ is the current weight vector, $\otimes$ is the outer product operator, and $dW_{oja}$ is the weight update.
                """
                for k in inputs.keys():
                    dW_oja = Y[..., None, :] * (
                        inputs[k] - (Y * self.input_processors[k].T)
                    )
                    self.processed_inputs[k].backward(-self.lr_oja * dW_oja.mean(dim=-2))

            if self.enable_L1:
                # 12. Layerwise L1 weight normalization
                r"""
                **12. Layerwise L1 weight normalization**: This rule normalizes the input weights by their Euclidean norm to ensure that the weight vectors have unit length. The weight normalization can be expressed as:

                $$W_{norm} = \frac{W}{\left\lVert W\right\rVert}$$

                where $W$ is the weight vector and $\left\lVert W\right\rVert$ is the Euclidean norm of the weight vector.
                """
                for k in inputs.keys():
                    self.input_processors[k] /= torch.norm(
                        self.input_processors[k], dim=-1, keepdim=True
                    )

            if self.enable_WC:
                # 13. Weight clipping
                r"""
                **13. Weight clipping**: This rule clips the input weights and biases to a maximum absolute value to prevent them from becoming too large. The weight clipping can be expressed as:

                $$W_{clip} = \text{clip}(W, -c, c)$$

                where $W_{clip}$ is the clipped weight vector, $W$ is the original weight vector, $c$ is the maximum absolute value for the weights and biases, and $\text{clip}$ is the clip function.
                """
                for k in inputs.keys():
                    self.input_processors[k] = torch.clamp(
                        self.input_processors[k],
                        -self.weight_clip,
                        self.weight_clip,
                    )
                self.bias = torch.clamp(self.bias, -self.weight_clip, self.weight_clip)

            if self.enable_VIC_input:
                # 14. VIC for inputs
                r"""
                **14. VIC for inputs**: This rule implements a regularization term to ensure that the input representations are invariant to the particular input that they came from (think: augmentation).

                https://www.youtube.com/watch?v=MzKDNmOJ67Q
                """
                projected = {
                    k: self.bu_input_projectors[k](self.processed_inputs[k])
                    for k in inputs.keys()
                }  # k: [B..., T, D]
                projected_stacked = torch.stack(
                    [projected[k] for k in inputs.keys()], dim=-2
                )  # [B..., T, k, D]
                # Variance: We want to keep the mean variance in (0, 1)
                # var_loss = sum[max[0, lambda_var - sqrt(var + epsilon)]]
                batch_seq_dims = tuple(range(len(projected_stacked.shape) - 2))
                input_variance = torch.var(
                    projected_stacked, dim=batch_seq_dims
                )  # shape: [k, D]
                loss += self.eta_VIC_input_var * torch.sum(
                    (1 / input_variance.shape[-1])
                    * torch.max(
                        torch.zeros_like(input_variance),
                        self.target_input_variance - (input_variance + 1e-3) ** 0.5,
                    )
                )
                # Invariance: We want to make the projected representation invariant from whichever input (augmentation) it was derived
                # inv_loss = (1/k) * sum[eta_inv * sum[sum[dist(x_i, x_j)]]]
                dists = torch.nn.functional.pairwise_distance(
                    projected_stacked, p=2
                )  # shape: [B, T, k, k]
                invariance_sum += (
                    self.eta_VIC_input_invar
                    * torch.sum(dists)
                    / (projected_stacked[..., 0].numel())
                )
                # Covariance: We want to minimize non-diagonal covariances to minimize redundancy
                # cov_loss = (1/(k-1)) * eta_cov * Cov(X) * (1-I)
                means = torch.mean(projected_stacked, dim=batch_seq_dims)  # shape: [k, D]
                # TODO: finish this

            if self.enable_FA:
                # 15. Feedback Alignment
                r"""
                **15. Feedback Alignment**: Treat the feedback alignment signals as ground truth gradients.

                This works on the premise that "an asymmetric feedback path can provide learning by aligning the back-propagated and forward propagated gradients with it's own, under the assumption of constant update directions for each data point".

                https://proceedings.neurips.cc/paper_files/paper/2016/file/d490d7b4576290fa60eb31b5fc917ad1-Paper.pdf
                """
                td_feedback = select(all_inputs, self.td_feedback_keys)
                grads = self.td_feedback_encoder(**td_feedback)
                Y.grad += self.eta_FA * grads

            if self.enable_FF:
                # 16. Forward-Forward Alignment
                r"""
                We are boasting signals that happen when the reward is high, and penalizing signals that happen when the reward is low.
                """
                Y.grad += self.eta_FF * self.reward * Y**2

            # we want to decrease the loss
            loss.grad = -1.0

            # reward accelerates learning
            if reward:
                self.optimizer.lr = self.lr * soft_v(reward)
            else:
                self.optimizer.lr = self.lr

            # TODO: somehow make all the gradients backpropagate to the inputs
            # TODO: this only backward's the loss, but I manually attached grads to several other params. How does the optimizer know which params to update and how far to propagate the grads? Should I use stopgrad to prevent grads from leaking out the inputs?

            # apply the gradients
            self.optimizer.step()  # TODO: maybe don't optimize here. Instead optimize at the global level, but just constrain the .backward pass to only take 10 hops

            # zero the gradients for the next iteration
            self.optimizer.zero_grad()

        return Y
