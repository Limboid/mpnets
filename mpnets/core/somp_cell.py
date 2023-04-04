import torch
import torch.nn as nn
import torch.optim as optim


class SOMPCell(nn.Module):

    DEFAULT_HPARAMS = {
        "alpha_sparse": 1.0,
        "alpha_dense": 1.0,
        "a_b_decay": 0.9,
        "a_b_discharge": 0.1,
        "noise_scale": 0.1,
        "threshold": 1.0,
        "eta_STDP": 0.01,
        "eta_CD": 0.01,
        "eta_SP": 0.01,
        "eta_IP": 0.01,
        "lambda_VIC_sim": 0.01,
        "lambda_VIC_covar": 0.01,
        "lambda_VIC_invar": 0.01,
        "eta_L2W": 0.01,
        "eta_MR": 0.01,
        "eta_SR": 0.01,
        "eta_SL": 0.01,
        "lr": 0.01,
        "enable_stdp": True,
        "enable_cd": True,
        "enable_sp": True,
        "enable_ip": True,
        "enable_vic": True,
        "enable_l2w": True,
        "enable_mr": True,
        "enable_sr": True,
        "enable_sl": True,
        "enable_lwta": True,
        "enable_soft_wta": True,
        "enable_oja": True,
        "enable_l1": True,
        "enable_wc": True,
        "input_bias": True,
        "q_input_bias": False,
        "gamma": 0.99,
    }

    def __init__(
        self,
        output_dim,
        activation_function=torch.nn.tanh,
        pooling_function=torch.mean,
        use_batch_norm=False,
        use_layer_norm=False,
        use_dropout=0.07,
        use_bucket=True,
        use_q_learning=False,
        input_sizes: dict[str, int] = None,
        **hparams
    ):
        super(SOMPCell, self).__init__()
        self.output_dim = output_dim
        self.activation_function = activation_function
        self.pooling_function = pooling_function
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_bucket = use_bucket
        self.use_q_learning = use_q_learning

        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr)
        self.hparams = {**SOMPCell.DEFAULT_HPARAMS, **hparams}

        self.batch_norms = {}
        self.layer_norms = {}
        self.input_layers = {}
        self.b = nn.Parameter(torch.randn(output_dim))
        self.activation_function = activation_function
        self.bucket = torch.zeros(output_dim)
        self.prev_output = torch.zeros((1, 1, output_dim))

        if self.use_q_learning:
            self.q_input_layers = {}
            self.q_head = nn.Linear(output_dim + output_dim, 1)

        if input_sizes is not None:
            for name, size in input_sizes.items():
                self.add_input(name, size)

        if use_dropout:
            self.dropout_layer = nn.Dropout(use_dropout)

    def sparse_threshold(self, Z, threshold=1):
        return torch.sign(Z) * torch.relu(torch.abs(Z) - threshold)

    def add_input(self, name, size):
        if self.use_batch_norm:
            self.batch_norms[name] = nn.BatchNorm1d(size)
            self.optimizer.param_groups[0]["params"].extend(
                self.batch_norms[name].parameters()
            )
        if self.use_layer_norm:
            self.layer_norms[name] = nn.LayerNorm(size)
            self.optimizer.param_groups[0]["params"].extend(
                self.layer_norms[name].parameters()
            )
        self.input_layers[name] = nn.Linear(
            size, self.output_dim, bias=self.hparams.input_bias
        )
        self.optimizer.param_groups[0]["params"].append(self.input_layers[name])
        if self.use_q_learning:
            self.q_input_layers[name] = nn.Linear(
                size, self.output_dim, bias=self.hparams.q_input_bias
            )
            self.optimizer.param_groups[0]["params"].append(self.q_input_layers[name])

    def forward(self, reward=None, **inputs):
        for k, input in inputs.items():
            if k not in self.input_layers:
                self.add_input(k, input.shape[-1])

        if self.use_batch_norm:
            inputs = {k: self.batch_norms[k](input) for k, input in inputs.items()}
        if self.use_layer_norm:
            inputs = {k: self.layer_norms[k](input) for k, input in inputs.items()}

        Z = (
            self.pooling_function(inputs[k] @ self.input_layers[k] for k in inputs.keys())
            + self.b
        )
        Z += torch.randn_like(Z) * self.hparams.noise_scale
        if self.dropout_layer:
            Z = self.dropout_layer(Z)

        if self.use_bucket:
            self.bucket = (
                self.hparams.a_b_decay * self.bucket
                + Z
                - self.hparams.a_b_discharge * torch.abs(self.prev)
            )
            Z = self.bucket

            if self.activation_function == "sparse":
                Y = self.hparams.alpha_sparse * self.sparse_threshold(
                    Z, alpha=self.hparams.threshold
                )
            else:
                Y = self.hparams.alpha_dense * self.activation_function(Z)
        else:
            # non-spiking
            Y = self.hparams.alpha_dense * self.activation_function(Z)

        if self.hparams.training:

            # Unsupervised update rules
            loss = 0.0

            if self.hparams.enable_stdp:
                # 1. STDP: spike-timing dependent plasticity
                for k in inputs.keys():
                    dW_stdp = (
                        inputs[k][..., :-1, :, None] * Y[..., 1:, None, :]
                        - inputs[k][..., 1:, :, None] * Y[..., :-1, None, :]
                    )
                    self.input_layers[k].grad -= self.hparams.lr_stdp * dW_stdp

            if self.hparams.enable_cd:
                # 2. CD: covariance decay
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
                    coef = self.hparams.a_cd * torch.abs(cov) - self.hparams.a_cd + 1
                    dW_cd = -(1 - coef) * self.input_layers[k]
                    self.input_layers[k].grad -= self.hparams.lr_cd * dW_cd

            if self.hparams.enable_sp:
                # 3. SP: structural plasticity
                for k in inputs.keys():
                    p_sp = self.hparams.a_sp / (inputs[k].shape[-1] * self.output_dim)
                    # TODO: check this why is this variable not used?
                    dW_sp = (
                        torch.bernoulli(self.hparams.p_sp) * self.input_layers[k].mean()
                    )
                    self.input_layers[k].grad -= self.hparams.lr_sp * dW_sp

            if self.hparams.enable_ip:
                # 4. IP: intrinsic plasticity
                self.b.grad -= self.hparams.lr_ip * (Y - 0.1)

            if self.hparams.enable_vic:
                # 5. VIC: variance-invariance-covariance
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
                    self.hparams.lambda_vic_sim * sim
                    + self.hparams.lambda_vic_covar * covar
                    + self.hparams.lambda_vic_invar * invar
                )

            if self.hparams.enable_l2:
                # 6. L2 regularization
                for k in inputs.keys():
                    l2_W = 0.5 * torch.sum(self.input_layers[k] ** 2)
                    self.add_loss(self.hparams.eta_L2W * l2_W)
                l2_b = 0.5 * torch.sum(self.b**2)
                loss += self.hparams.eta_L2b * l2_b

            if self.hparams.enable_mr:
                # 7. Mean regularization
                loss_MR = (Y.sum() - self.hparams.mu_MR) ** 2
                loss += self.hparams.eta_MR * loss_MR

            if self.hparams.enable_sr:
                # 8. Sparsity regularization (applied only to sparse activation functions)
                if self.activation_function == "sparse":
                    kl_SR = Y * torch.log(Y / self.hparams.a_SR) - (1 - Y) * torch.log(
                        (1 - Y) / (1 - self.hparams.a_SR)
                    )
                    loss_SR = kl_SR.sum()
                    loss += self.hparams.eta_SR * loss_SR

            if self.hparams.enable_lwta:
                # 9. Local Winner-Take-All (LWTA) competition
                _, max_indices = torch.max(Y, dim=-1, keepdim=True)
                Y_lwta = torch.zeros_like(Y)
                Y_lwta.scatter_(-1, max_indices, 1.0)
                dW_lwta = (Y_lwta[..., None, :] * inputs[k]).mean(dim=-2)
                self.input_layers[k].grad -= self.hparams.lr_lwta * dW_lwta

            if self.hparams.enable_soft_wta:
                # 10. Soft Winner-Take-All (WTA) competition
                Y_exp = torch.exp(Y / self.hparams.soft_wta_temperature)
                Y_softmax = Y_exp / Y_exp.sum(dim=-1, keepdim=True)
                dW_wta = torch.einsum("...i,...j->...ij", Y_softmax, inputs[k]).mean(
                    dim=-2
                )
                self.input_layers[k].grad -= self.hparams.lr_soft_wta * dW_wta

            if self.hparams.enable_oja:
                # 11. Oja's rule
                for k in inputs.keys():
                    dW_oja = Y[..., None, :] * (inputs[k] - (Y * self.input_layers[k].T))
                    self.input_layers[k].grad -= self.hparams.lr_oja * dW_oja.mean(dim=-2)

            if self.hparams.enable_l1:
                # 12. Weight normalization
                for k in inputs.keys():
                    self.input_layers[k] /= torch.norm(
                        self.input_layers[k], dim=-1, keepdim=True
                    )

            if self.hparams.enable_wc:
                # 13. Weight clipping
                for k in inputs.keys():
                    self.input_layers[k] = torch.clamp(
                        self.input_layers[k],
                        -self.hparams.weight_clip,
                        self.hparams.weight_clip,
                    )
                self.b = torch.clamp(
                    self.b, -self.hparams.weight_clip, self.hparams.weight_clip
                )

            if self.hparams.enable_qlearning:
                # Q-learning
                assert reward is not None, "Reward must be provided for Q-learning"
                # Update critic:
                prev_inputs = {k: inputs[k][..., :-1, :] for k in inputs.keys()}
                prev_inputs_emb = sum(
                    self.q_input_layers[k](prev_inputs[k]) for k in self.q_input_layers
                )
                prev_output = (
                    self.prev_output
                )  # already shifted by one timestep since it is the output of the previous iteration
                predicted = self.q_head(torch.cat([prev_inputs_emb, prev_output], dim=-1))
                error = reward - predicted
                # TODO: this should be implemented as a wrapper around the entire SOMPCell class. Same goes for SAC and other RL algorithms
                # Update the Q-value for the previous state-action pair
                target = reward + self.hparams.gamma * torch.max(predicted, dim=-1)[0]
                current_inputs = {k: inputs[k] for k in inputs.keys()}
                current_inputs_emb = sum(
                    self.q_input_layers[k](current_inputs[k]) for k in self.q_input_layers
                )
                q_value = self.q_head(current_inputs_emb)
                q_error = (target - q_value) ** 2
                self.q_optimizer.zero_grad()
                q_error.backward()
                self.q_optimizer.step()

                # Update the policy network using the Q-value as a target
                policy_loss = -torch.mean(
                    self.policy_head(torch.cat([prev_inputs_emb, prev_output], dim=-1))
                    * current_output.detach()[0]
                )
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                self.prev_output = current_output.detach()

            # apply the gradients
            self.optimizer.step()

            # zero the gradients for the next iteration
            self.optimizer.zero_grad()

        self.prev_output = Y
        return Y
