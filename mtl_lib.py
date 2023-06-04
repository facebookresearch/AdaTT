# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
from dataclasses import dataclass, field
from math import sqrt
from typing import List, Optional, Union

import torch
import torch.nn as nn


logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class MtlConfigs:
    mtl_model: str = "att_sp"  # consider using enum
    num_task_experts: int = 1
    num_shared_experts: int = 1
    expert_out_dims: List[List[int]] = field(default_factory=list)
    self_exp_res_connect: bool = False
    expert_archs: Optional[List[List[int]]] = None
    gate_archs: Optional[List[List[int]]] = None
    num_experts: Optional[int] = None


@dataclass(frozen=True)
class ArchInputs:
    num_task: int = 3

    task_mlp: List[int] = field(default_factory=list)

    mtl_configs: Optional[MtlConfigs] = field(default=None)

    # Parameters related to activation function
    activation_type: str = "RELU"


class AdaTTSp(nn.Module):
    """
    paper title: "AdaTT: Adaptive Task-to-Task Fusion Network for Multitask Learning in Recommendations"
    paper link: https://doi.org/10.1145/3580305.3599769
    Call Args:
        inputs: inputs is a tensor of dimension
            [batch_size, self.num_tasks, self.input_dim].
            Experts in the same module share the same input.
        outputs dimensions: [B, T, D_out]

    Example::
        AdaTTSp(
            input_dim=256,
            expert_out_dims=[[128, 128]],
            num_tasks=8,
            num_task_experts=2,
            self_exp_res_connect=True,
        )
    """

    def __init__(
        self,
        input_dim: int,
        expert_out_dims: List[List[int]],
        num_tasks: int,
        num_task_experts: int,
        self_exp_res_connect: bool = True,
        activation: str = "RELU",
    ) -> None:
        super().__init__()
        if len(expert_out_dims) == 0:
            logger.warning(
                "AdaTTSp is noop! size of expert_out_dims which is the number of "
                "extraction layers should be at least 1."
            )
            return
        self.num_extraction_layers: int = len(expert_out_dims)
        self.num_tasks = num_tasks
        self.num_task_experts = num_task_experts
        self.total_experts_per_layer: int = num_task_experts * num_tasks
        self.self_exp_res_connect = self_exp_res_connect
        self.experts = torch.nn.ModuleList()
        self.gate_weights = torch.nn.ModuleList()

        self_exp_weight_list = []
        layer_input_dim = input_dim
        for expert_out_dim in expert_out_dims:
            self.experts.append(
                torch.nn.ModuleList(
                    [
                        MLP(layer_input_dim, expert_out_dim, activation)
                        for i in range(self.total_experts_per_layer)
                    ]
                )
            )

            self.gate_weights.append(
                torch.nn.ModuleList(
                    [
                        torch.nn.Sequential(
                            torch.nn.Linear(
                                layer_input_dim, self.total_experts_per_layer
                            ),
                            torch.nn.Softmax(dim=-1),
                        )
                        for _ in range(num_tasks)
                    ]
                )
            )  # self.gate_weights is of shape L X T, after we loop over all layers.

            if self_exp_res_connect and num_task_experts > 1:
                params = torch.empty(num_tasks, num_task_experts)
                scale = sqrt(1.0 / num_task_experts)
                torch.nn.init.uniform_(params, a=-scale, b=scale)
                self_exp_weight_list.append(torch.nn.Parameter(params))

            layer_input_dim = expert_out_dim[-1]

        self.self_exp_weights = nn.ParameterList(self_exp_weight_list)

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        for layer_i in range(self.num_extraction_layers):
            # all task expert outputs.
            experts_out = torch.stack(
                [
                    expert(inputs[:, expert_i // self.num_task_experts, :])
                    for expert_i, expert in enumerate(self.experts[layer_i])
                ],
                dim=1,
            )  # [B * E (total experts) * D_out]

            gates = torch.stack(
                [
                    gate_weight(
                        inputs[:, task_i, :]
                    )  #  W ([B, D]) * S ([D, E]) -> G, dim is [B, E]
                    for task_i, gate_weight in enumerate(self.gate_weights[layer_i])
                ],
                dim=1,
            )  # [B, T, E]
            fused_experts_out = torch.bmm(
                gates,
                experts_out,
            )  # [B, T, E] X [B * E (total experts) * D_out] -> [B, T, D_out]

            if self.self_exp_res_connect:
                if self.num_task_experts > 1:
                    # residual from the linear combination of tasks' own experts.
                    self_exp_weighted = torch.einsum(
                        "te,bted->btd",
                        self.self_exp_weights[layer_i],
                        experts_out.view(
                            experts_out.size(0),
                            self.num_tasks,
                            self.num_task_experts,
                            -1,
                        ),  # [B * E (total experts) * D_out] -> [B * T * E_task * D_out]
                    )  #  bmm: [T * E_task] X [B * T * E_task * D_out] -> [B, T, D_out]

                    fused_experts_out = (
                        fused_experts_out + self_exp_weighted
                    )  # [B, T, D_out]
                else:
                    fused_experts_out = fused_experts_out + experts_out

            inputs = fused_experts_out

        return inputs


class AdaTTWSharedExps(nn.Module):
    """
    paper title: "AdaTT: Adaptive Task-to-Task Fusion Network for Multitask Learning in Recommendations"
    paper link: https://doi.org/10.1145/3580305.3599769
    Call Args:
        inputs: inputs is a tensor of dimension
            [batch_size, self.num_tasks, self.input_dim].
            Experts in the same module share the same input.
        outputs dimensions: [B, T, D_out]

    Example::
        AdaTTWSharedExps(
            input_dim=256,
            expert_out_dims=[[128, 128]],
            num_tasks=8,
            num_shared_experts=1,
            num_task_experts=2,
            self_exp_res_connect=True,
        )
    """

    def __init__(
        self,
        input_dim: int,
        expert_out_dims: List[List[int]],
        num_tasks: int,
        num_shared_experts: int,
        num_task_experts: Optional[int] = None,
        num_task_expert_list: Optional[List[int]] = None,
        # Set num_task_expert_list for experimenting with a flexible number of
        # experts for different task_specific units.
        self_exp_res_connect: bool = True,
        activation: str = "RELU",
    ) -> None:
        super().__init__()
        if len(expert_out_dims) == 0:
            logger.warning(
                "AdaTTWSharedExps is noop! size of expert_out_dims which is the number of "
                "extraction layers should be at least 1."
            )
            return
        self.num_extraction_layers: int = len(expert_out_dims)
        self.num_tasks = num_tasks
        assert (num_task_experts is None) ^ (num_task_expert_list is None)
        if num_task_experts is not None:
            self.num_expert_list = [num_task_experts for _ in range(num_tasks)]
        else:
            # num_expert_list is guaranteed to be not None here.
            # pyre-ignore
            self.num_expert_list: List[int] = num_task_expert_list
        self.num_expert_list.append(num_shared_experts)

        self.total_experts_per_layer: int = sum(self.num_expert_list)
        self.self_exp_res_connect = self_exp_res_connect
        self.experts = torch.nn.ModuleList()
        self.gate_weights = torch.nn.ModuleList()

        layer_input_dim = input_dim
        for layer_i, expert_out_dim in enumerate(expert_out_dims):
            self.experts.append(
                torch.nn.ModuleList(
                    [
                        MLP(layer_input_dim, expert_out_dim, activation)
                        for i in range(self.total_experts_per_layer)
                    ]
                )
            )

            num_full_active_modules = (
                num_tasks
                if layer_i == self.num_extraction_layers - 1
                else num_tasks + 1
            )

            self.gate_weights.append(
                torch.nn.ModuleList(
                    [
                        torch.nn.Sequential(
                            torch.nn.Linear(
                                layer_input_dim, self.total_experts_per_layer
                            ),
                            torch.nn.Softmax(dim=-1),
                        )
                        for _ in range(num_full_active_modules)
                    ]
                )
            )  # self.gate_weights is a 2d module list of shape L X T (+ 1), after we loop over all layers.

            layer_input_dim = expert_out_dim[-1]

        self_exp_weight_list = []
        if self_exp_res_connect:
            # If any tasks have number of experts not equal to 1, we learn linear combinations of native experts.
            if any(num_experts != 1 for num_experts in self.num_expert_list):
                for i in range(num_tasks + 1):
                    num_full_active_layer = (
                        self.num_extraction_layers - 1
                        if i == num_tasks
                        else self.num_extraction_layers
                    )
                    params = torch.empty(
                        num_full_active_layer,
                        self.num_expert_list[i],
                    )
                    scale = sqrt(1.0 / self.num_expert_list[i])
                    torch.nn.init.uniform_(params, a=-scale, b=scale)
                    self_exp_weight_list.append(torch.nn.Parameter(params))

        self.self_exp_weights = nn.ParameterList(self_exp_weight_list)

        self.expert_input_idx: List[int] = []
        for i in range(num_tasks + 1):
            self.expert_input_idx.extend([i for _ in range(self.num_expert_list[i])])

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        for layer_i in range(self.num_extraction_layers):
            num_full_active_modules = (
                self.num_tasks
                if layer_i == self.num_extraction_layers - 1
                else self.num_tasks + 1
            )
            # all task expert outputs.
            experts_out = torch.stack(
                [
                    expert(inputs[:, self.expert_input_idx[expert_i], :])
                    for expert_i, expert in enumerate(self.experts[layer_i])
                ],
                dim=1,
            )  # [B * E (total experts) * D_out]

            # gate weights for fusing all experts.
            gates = torch.stack(
                [
                    gate_weight(inputs[:, i, :])  #  [B, D] * [D, E] -> [B, E]
                    for i, gate_weight in enumerate(self.gate_weights[layer_i])
                ],
                dim=1,
            )  # [B, T (+ 1), E]

            # add all expert gate weights with native expert weights.
            if self.self_exp_res_connect:
                prev_idx = 0
                use_unit_naive_weights = all(
                    num_expert == 1 for num_expert in self.num_expert_list
                )
                for module_i in range(num_full_active_modules):
                    next_idx = self.num_expert_list[module_i] + prev_idx
                    if use_unit_naive_weights:
                        gates[:, module_i, prev_idx:next_idx] += torch.ones(
                            1, self.num_expert_list[module_i]
                        )
                    else:
                        gates[:, module_i, prev_idx:next_idx] += self.self_exp_weights[
                            module_i
                        ][layer_i].unsqueeze(0)
                    prev_idx = next_idx

            fused_experts_out = torch.bmm(
                gates,
                experts_out,
            )  # [B, T (+ 1), E (total)] X [B * E (total) * D_out] -> [B, T (+ 1), D_out]

            inputs = fused_experts_out

        return inputs


class MLP(nn.Module):
    """
    Args:
        input_dim (int):
        mlp_arch (List[int]):
        activation (str):

    Call Args:
        input (torch.Tensor): tensor of shape (B, I)

    Returns:
        output (torch.Tensor): MLP result

    Example::

        mlp = MLP(100, [100])

    """

    def __init__(
        self,
        input_dim: int,
        mlp_arch: List[int],
        activation: str = "RELU",
        bias: bool = True,
    ) -> None:
        super().__init__()

        mlp_net = []
        for mlp_dim in mlp_arch:
            mlp_net.append(
                nn.Linear(in_features=input_dim, out_features=mlp_dim, bias=bias)
            )
            if activation == "RELU":
                mlp_net.append(nn.ReLU())
            else:
                raise ValueError("only RELU is included currently")
            input_dim = mlp_dim
        self.mlp_net = nn.Sequential(*mlp_net)

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        return self.mlp_net(input)


class SharedBottom(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dims: List[int], num_tasks: int, activation: str
    ) -> None:
        super().__init__()
        self.bottom_projection = MLP(input_dim, hidden_dims, activation)
        self.num_tasks: int = num_tasks

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        # input dim [T, D_in]
        # output dim [B, T, D_out]
        return self.bottom_projection(input).unsqueeze(1).expand(-1, self.num_tasks, -1)


class CrossStitch(torch.nn.Module):
    """
    cross-stitch
    paper title: "Cross-stitch Networks for Multi-task Learning".
    paper link: https://openaccess.thecvf.com/content_cvpr_2016/papers/Misra_Cross-Stitch_Networks_for_CVPR_2016_paper.pdf
    """

    def __init__(
        self,
        input_dim: int,
        expert_archs: List[List[int]],
        num_tasks: int,
        activation: str = "RELU",
    ) -> None:
        super().__init__()
        self.num_layers: int = len(expert_archs)
        self.num_tasks = num_tasks
        self.experts = torch.nn.ModuleList()
        self.stitchs = torch.nn.ModuleList()

        expert_input_dim = input_dim
        for layer_ind in range(self.num_layers):
            self.experts.append(
                torch.nn.ModuleList(
                    [
                        MLP(
                            expert_input_dim,
                            expert_archs[layer_ind],
                            activation,
                        )
                        for _ in range(self.num_tasks)
                    ]
                )
            )

            self.stitchs.append(
                torch.nn.Linear(
                    self.num_tasks,
                    self.num_tasks,
                    bias=False,
                )
            )

            expert_input_dim = expert_archs[layer_ind][-1]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        input dim [B, T, D_in]
        output dim [B, T, D_out]
        """
        x = input

        for layer_ind in range(self.num_layers):
            expert_out = torch.stack(
                [
                    expert(x[:, expert_ind, :])  # [B, D_out]
                    for expert_ind, expert in enumerate(self.experts[layer_ind])
                ],
                dim=1,
            )  # [B, T, D_out]

            stitch_out = self.stitchs[layer_ind](expert_out.transpose(1, 2)).transpose(
                1, 2
            )  # [B, T, D_out]

            x = stitch_out

        return x


class MLMMoE(torch.nn.Module):
    """
    Multi-level Multi-gate Mixture of Experts
    This code implements a multi-level extension of the MMoE model, as described in the
    paper titled "Modeling Task Relationships in Multi-task Learning with Multi-gate
    Mixture-of-Experts".
    Paper link: https://dl.acm.org/doi/10.1145/3219819.3220007
    To run the original MMoE, use only one fusion level. For example, set expert_archs as
    [[96, 48]].
    To configure multiple fusion levels, set expert_archs as something like [[96], [48]].
    """

    def __init__(
        self,
        input_dim: int,
        expert_archs: List[List[int]],
        gate_archs: List[List[int]],
        num_tasks: int,
        num_experts: int,
        activation: str = "RELU",
    ) -> None:
        super().__init__()
        self.num_layers: int = len(expert_archs)
        self.num_tasks: int = num_tasks
        self.num_experts = num_experts
        self.experts = torch.nn.ModuleList()
        self.gates = torch.nn.ModuleList()

        expert_input_dim = input_dim
        for layer_ind in range(self.num_layers):
            self.experts.append(
                torch.nn.ModuleList(
                    [
                        MLP(
                            expert_input_dim,
                            expert_archs[layer_ind],
                            activation,
                        )
                        for _ in range(self.num_experts)
                    ]
                )
            )
            self.gates.append(
                torch.nn.ModuleList(
                    [
                        torch.nn.Sequential(
                            MLP(
                                input_dim,
                                gate_archs[layer_ind],
                                activation,
                            ),
                            torch.nn.Linear(
                                gate_archs[layer_ind][-1]
                                if gate_archs[layer_ind]
                                else input_dim,
                                self.num_experts,
                            ),
                            torch.nn.Softmax(dim=-1),
                        )
                        for _ in range(
                            self.num_experts
                            if layer_ind < self.num_layers - 1
                            else self.num_tasks
                        )
                    ]
                )
            )
            expert_input_dim = expert_archs[layer_ind][-1]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        input dim [B, D_in]
        output dim [B, T, D_out]
        """
        x = input.unsqueeze(1).expand([-1, self.num_experts, -1])  # [B, E, D_in]

        for layer_ind in range(self.num_layers):
            expert_out = torch.stack(
                [
                    expert(x[:, expert_ind, :])  # [B, D_out]
                    for expert_ind, expert in enumerate(self.experts[layer_ind])
                ],
                dim=1,
            )  # [B, E, D_out]

            gate_out = torch.stack(
                [
                    gate(input)  # [B, E]
                    for gate_ind, gate in enumerate(self.gates[layer_ind])
                ],
                dim=1,
            )  # [B, T, E]

            gated_out = torch.matmul(gate_out, expert_out)  # [B, T, D_out]

            x = gated_out
        return x


class PLE(nn.Module):
    """
    PLE module is based on the paper "Progressive Layered Extraction (PLE): A
    Novel Multi-Task Learning (MTL) Model for Personalized Recommendations".
    Paper link: https://doi.org/10.1145/3383313.3412236
    PLE aims to address negative transfer and seesaw phenomenon in multi-task
    learning. PLE distinguishes shared and task-specic experts explicitly and
    adopts a progressive routing mechanism to extract and separate deeper
    semantic knowledge gradually. When there is only one extraction layer, PLE
    falls back to CGC.

    Args:
        input_dim: input embedding dimension
        expert_out_dims (List[List[int]]): dimension of an expert's output at
            each layer. This list's length equals the number of extraction
            layers
        num_tasks: number of tasks
        num_task_experts: number of experts for each task module at each layer.
            * If the number of experts is the same for all tasks, use an
            integer here.
            * If the number of experts is different for different tasks, use a
            list of integers here.
        num_shared_experts: number of experts for shared module at each layer

    Call Args:
        inputs: inputs is a tensor of dimension [batch_size, self.num_tasks + 1,
        self.input_dim]. Task specific module inputs are placed first, followed
        by shared module input. (Experts in the same module share the same input)

    Returns:
        output: output of extraction layer to be feed into task-specific tower
            networks. It's a list of tensors, each of which is for one task.

    Example::
        PLE(
            input_dim=256,
            expert_out_dims=[[128]],
            num_tasks=8,
            num_task_experts=2,
            num_shared_experts=2,
        )

    """

    def __init__(
        self,
        input_dim: int,
        expert_out_dims: List[List[int]],
        num_tasks: int,
        num_task_experts: Union[int, List[int]],
        num_shared_experts: int,
        activation: str = "RELU",
    ) -> None:
        super().__init__()
        if len(expert_out_dims) == 0:
            raise ValueError("Expert out dims cannot be empty list")
        self.num_extraction_layers: int = len(expert_out_dims)
        self.num_tasks = num_tasks
        self.num_task_experts = num_task_experts
        if type(num_task_experts) is int:
            self.total_experts_per_layer: int = (
                num_task_experts * num_tasks + num_shared_experts
            )
        else:
            self.total_experts_per_layer: int = (
                sum(num_task_experts) + num_shared_experts
            )
            assert len(num_task_experts) == num_tasks
        self.num_shared_experts = num_shared_experts
        self.experts = nn.ModuleList()
        expert_input_dim = input_dim
        for expert_out_dim in expert_out_dims:
            self.experts.append(
                nn.ModuleList(
                    [
                        MLP(expert_input_dim, expert_out_dim, activation)
                        for i in range(self.total_experts_per_layer)
                    ]
                )
            )
            expert_input_dim = expert_out_dim[-1]

        self.gate_weights = nn.ModuleList()
        selector_dim = input_dim
        for i in range(self.num_extraction_layers):
            expert_out_dim = expert_out_dims[i]
            # task specific gates.
            if type(num_task_experts) is int:
                gate_weights_in_layer = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(
                                selector_dim, num_task_experts + num_shared_experts
                            ),
                            nn.Softmax(dim=-1),
                        )
                        for i in range(num_tasks)
                    ]
                )
            else:
                gate_weights_in_layer = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(
                                selector_dim, num_task_experts[i] + num_shared_experts
                            ),
                            nn.Softmax(dim=-1),
                        )
                        for i in range(num_tasks)
                    ]
                )
            # Shared module gates. Note last layer has only task specific module gates for task towers later.
            if i != self.num_extraction_layers - 1:
                gate_weights_in_layer.append(
                    nn.Sequential(
                        nn.Linear(selector_dim, self.total_experts_per_layer),
                        nn.Softmax(dim=-1),
                    )
                )
            self.gate_weights.append(gate_weights_in_layer)

            selector_dim = expert_out_dim[-1]

        if type(self.num_task_experts) is list:
            experts_idx_2_task_idx = []
            for i in range(num_tasks):
                # pyre-ignore
                experts_idx_2_task_idx += [i] * self.num_task_experts[i]
            experts_idx_2_task_idx += [num_tasks] * num_shared_experts
            self.experts_idx_2_task_idx: List[int] = experts_idx_2_task_idx

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        for layer_i in range(self.num_extraction_layers):
            # all task specific and shared experts' outputs.
            # Note first num_task_experts * num_tasks experts are task specific,
            # last num_shared_experts experts are shared.
            if type(self.num_task_experts) is int:
                experts_out = torch.stack(
                    [
                        self.experts[layer_i][expert_i](
                            inputs[
                                :,
                                # pyre-ignore
                                min(expert_i // self.num_task_experts, self.num_tasks),
                                :,
                            ]
                        )
                        for expert_i in range(self.total_experts_per_layer)
                    ],
                    dim=1,
                )  # [B * E (num experts) * D_out]
            else:
                experts_out = torch.stack(
                    [
                        self.experts[layer_i][expert_i](
                            inputs[
                                :,
                                self.experts_idx_2_task_idx[expert_i],
                                :,
                            ]
                        )
                        for expert_i in range(self.total_experts_per_layer)
                    ],
                    dim=1,
                )  # [B * E (num experts) * D_out]

            gates_out = []
            # Loop over all the gates in the layer. Note for the last layer,
            # there is no shared gating network.
            prev_idx = 0
            for gate_i in range(len(self.gate_weights[layer_i])):
                # This is for shared gating network, which uses all the experts.
                if gate_i == self.num_tasks:
                    selected_matrix = experts_out  # S_share
                # This is for task gating network, which only uses shared and its own experts.
                else:
                    if type(self.num_task_experts) is int:
                        task_experts_out = experts_out[
                            :,
                            # pyre-ignore
                            (gate_i * self.num_task_experts) : (gate_i + 1)
                            # pyre-ignore
                            * self.num_task_experts,
                            :,
                        ]  # task specific experts
                    else:
                        # pyre-ignore
                        next_idx = prev_idx + self.num_task_experts[gate_i]
                        task_experts_out = experts_out[
                            :,
                            prev_idx:next_idx,
                            :,
                        ]  # task specific experts
                        prev_idx = next_idx
                    shared_experts_out = experts_out[
                        :,
                        -self.num_shared_experts :,
                        :,
                    ]  # shared experts
                    selected_matrix = torch.concat(
                        [task_experts_out, shared_experts_out], dim=1
                    )  # S_k with dimension of [B * E_selected * D_out]

                gates_out.append(
                    torch.bmm(
                        self.gate_weights[layer_i][gate_i](
                            inputs[:, gate_i, :]
                        ).unsqueeze(dim=1),
                        selected_matrix,
                    )
                    #  W * S -> G
                    #  [B, 1, E_selected] X [B * E_selected * D_out] -> [B, 1, D_out]
                )
            inputs = torch.cat(gates_out, dim=1)  # [B, T, D_out]

        return inputs


class CentralTaskArch(nn.Module):
    def __init__(
        self,
        mtl_configs: MtlConfigs,
        opts: ArchInputs,
        input_dim: int,
    ) -> None:
        super().__init__()
        self.opts = opts

        assert len(mtl_configs.expert_out_dims) > 0, "expert_out_dims is empty."
        self.num_tasks: int = opts.num_task

        self.mtl_model: str = mtl_configs.mtl_model
        logger.info(f"mtl_model is {mtl_configs.mtl_model}")
        expert_out_dims: List[List[int]] = mtl_configs.expert_out_dims
        # AdaTT-sp
        # consider consolidating the implementation of att_sp and att_g.
        if mtl_configs.mtl_model == "att_sp":
            self.mtl_arch: nn.Module = AdaTTSp(
                input_dim=input_dim,
                expert_out_dims=expert_out_dims,
                num_tasks=self.num_tasks,
                num_task_experts=mtl_configs.num_task_experts,
                self_exp_res_connect=mtl_configs.self_exp_res_connect,
                activation=opts.activation_type,
            )
        # AdaTT-general
        elif mtl_configs.mtl_model == "att_g":
            self.mtl_arch: nn.Module = AdaTTWSharedExps(
                input_dim=input_dim,
                expert_out_dims=expert_out_dims,
                num_tasks=self.num_tasks,
                num_task_experts=mtl_configs.num_task_experts,
                num_shared_experts=mtl_configs.num_shared_experts,
                self_exp_res_connect=mtl_configs.self_exp_res_connect,
                activation=opts.activation_type,
            )
        # PLE
        elif mtl_configs.mtl_model == "ple":
            self.mtl_arch: nn.Module = PLE(
                input_dim=input_dim,
                expert_out_dims=expert_out_dims,
                num_tasks=self.num_tasks,
                num_task_experts=mtl_configs.num_task_experts,
                num_shared_experts=mtl_configs.num_shared_experts,
                activation=opts.activation_type,
            )
        # cross-stitch
        elif mtl_configs.mtl_model == "cross_st":
            self.mtl_arch: nn.Module = CrossStitch(
                input_dim=input_dim,
                expert_archs=mtl_configs.expert_out_dims,
                num_tasks=self.num_tasks,
                activation=opts.activation_type,
            )
        # multi-layer MMoE or MMoE
        elif mtl_configs.mtl_model == "mmoe":
            self.mtl_arch: nn.Module = MLMMoE(
                input_dim=input_dim,
                expert_archs=mtl_configs.expert_out_dims,
                gate_archs=[[] for i in range(len(mtl_configs.expert_out_dims))],
                num_tasks=self.num_tasks,
                num_experts=mtl_configs.num_shared_experts,
                activation=opts.activation_type,
            )
        # shared bottom
        elif mtl_configs.mtl_model == "share_bottom":
            self.mtl_arch: nn.Module = SharedBottom(
                input_dim,
                [dim for dims in expert_out_dims for dim in dims],
                self.num_tasks,
                opts.activation_type,
            )
        else:
            raise ValueError("invalid model type")

        task_modules_input_dim = expert_out_dims[-1][-1]
        self.task_modules: nn.ModuleList = nn.ModuleList(
            [
                nn.Sequential(
                    MLP(
                        task_modules_input_dim, self.opts.task_mlp, opts.activation_type
                    ),
                    torch.nn.Linear(self.opts.task_mlp[-1], 1),
                )
                for i in range(self.num_tasks)
            ]
        )

    def forward(
        self,
        task_arch_input: torch.Tensor,
    ) -> List[torch.Tensor]:
        if self.mtl_model in ["att_sp", "cross_st"]:
            task_arch_input = task_arch_input.unsqueeze(1).expand(
                -1, self.num_tasks, -1
            )
        elif self.mtl_model in ["att_g", "ple"]:
            task_arch_input = task_arch_input.unsqueeze(1).expand(
                -1, self.num_tasks + 1, -1
            )

        task_specific_outputs = self.mtl_arch(task_arch_input)

        task_arch_output = [
            task_module(task_specific_outputs[:, i, :])
            for i, task_module in enumerate(self.task_modules)
        ]

        return task_arch_output