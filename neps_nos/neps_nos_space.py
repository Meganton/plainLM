"""
Neural Optimizer Search (NOS) space definitions.

This module defines the search space for neural optimizers,
including variable definitions, operations, and optimizer construction logic.
"""

import torch
import neps
from functools import partial
from typing import Callable, Tuple, Literal


def scale_by_constant(constant):
    """Scale tensor by a constant value."""
    return partial(torch.mul, other=constant)


def clamp_by_constant(constant):
    """Clamp tensor values within [-constant, constant]."""
    return partial(torch.clamp, min=-constant, max=constant)


def interpolate(constant):
    """Interpolate between tensors with given weight."""
    return partial(torch.lerp, weight=constant)


def resolve_expression(expr, var_dict):
    """
    Recursively evaluate nested tuples representing operations.

    Args:
        expr: Expression to resolve (string variable name or tuple operation)
        var_dict: Dictionary mapping variable names to values

    Returns:
        Resolved value
    """
    if isinstance(expr, str):
        return var_dict.get(expr, expr)

    op, *args = expr

    # Resolve nested args first (bottom-up)
    args = [resolve_expression(a, var_dict) for a in args]
    for n, arg in enumerate(args):
        if isinstance(arg, str):
            args[n] = var_dict[arg]


    return op(*args)


def _to_runtime_tensor(value, like_tensor):
    """Convert scalar-like runtime metadata to a tensor matching parameter dtype/device."""
    if isinstance(value, torch.Tensor):
        return value.to(device=like_tensor.device, dtype=like_tensor.dtype)
    try:
        return torch.tensor(float(value), dtype=like_tensor.dtype, device=like_tensor.device)
    except Exception:
        return torch.tensor(0.0, dtype=like_tensor.dtype, device=like_tensor.device)


def _runtime_symbol_dict(optimizer, group, like_tensor):
    """Return optional runtime symbols with backwards-compatible defaults."""
    t = getattr(optimizer, "_nos_t", 0.0)

    depth = group.get("depth", 0.0)
    layer_type_attention = group.get("layer_type_attention", 0.0)

    return {
        "t": _to_runtime_tensor(t, like_tensor),
        "depth": _to_runtime_tensor(depth, like_tensor),
        "layer_type_attention": _to_runtime_tensor(layer_type_attention, like_tensor),
    }


class PremadeBlocks(neps.PipelineSpace):
    """
    Neural Optimizer Search space with variable number of lines and fixed last line updating u.

    This space allows for flexible optimizer definitions with up to max_lines
    of update rules, where each line can assign to either v1 or v2, and the last line always updates u.
    """

    @staticmethod
    def function_wrapper(function: Callable, *inputs):
        """Package function with its inputs."""
        return function, *inputs

    @staticmethod
    def first_moment(beta, m, g):
        """First moment estimation block."""
        return beta * m + (1 - beta) * g

    @staticmethod
    def second_moment(beta, v, g):
        """Second moment estimation block."""
        return beta * v + (1 - beta) * (g * g)

    @staticmethod
    def bias_correction(moment, beta, t):
        """Bias correction for moment estimates."""
        return moment / (1 - beta ** t)


    def __init__(
        self,
        n_lines: Tuple[int, int] = (1, 10),
        fidelity: Tuple[int, int] | None = None,
        learning_rate: Tuple[float, float] | None = None,
        weight_decay: Tuple[float, float] | None = None,
        **_,
    ):
        """
        Initialize the NOS space.

        Args:
            n_lines: Tuple indicating (min_lines, max_lines)
            fidelity: Optional fidelity range (lower, upper)
            learning_rate: Optional learning rate range (lower, upper)
            weight_decay: Optional weight decay range (lower, upper)
        """
        assert (
            n_lines[0] >= 0 and n_lines[1] >= n_lines[0]
        ), f"Invalid n_lines range {n_lines}"

        if fidelity is not None:
            assert (
                len(fidelity) == 2 and 0 <= fidelity[0] < fidelity[1]
            ), f"Invalid fidelity range {fidelity}"
            self.fidelity = neps.IntegerFidelity(lower=fidelity[0], upper=fidelity[1])

        if learning_rate is not None:
            assert (
                len(learning_rate) == 2 and 0 < learning_rate[0] < learning_rate[1]
            ), f"Invalid learning rate range {learning_rate}"
            self.learning_rate = neps.Float(
                lower=learning_rate[0], upper=learning_rate[1], log=True
            )

        if weight_decay is not None:
            weight_decay = (max(weight_decay[0], 1e-8), weight_decay[1])
            assert (
                1e-8 <= weight_decay[0] < weight_decay[1]
            ), f"Invalid weight decay range {weight_decay}"
            self.weight_decay = neps.Float(
                lower=weight_decay[0], upper=weight_decay[1], log=True
            )

        # Define variables that can be used in optimizer
        self._input_variables = neps.Categorical(choices=("w", "g", "v1", "v2"))
        self._output_variables = neps.Categorical(choices=("v1", "v2"))

        # Define constants that can be used in operations
        self._constants = neps.Categorical(choices=(10, 1, 0, 0.1, 0.01, 0.9, 0.99))

        self._first_moment = neps.Operation(
            operator=self.first_moment,
        )

        self._second_moment = neps.Operation(
            operator=self.second_moment,
        )

        self._clamp_by_constant = neps.Operation(
            operator=clamp_by_constant,
            kwargs={"constant": self._constants.resample()},
        )

        self._bias_correction = neps.Operation(
            operator=self.bias_correction,
        )

        self._first_moment_rh = neps.Operation(
            operator=self.function_wrapper,
            args=(self.first_moment, "b1", "m", "g")
        )

        self._second_moment_rh = neps.Operation(
            operator=self.function_wrapper,
            args=(self.second_moment, "b2", "v", "g")
        )

        self._clamp_gradient_rh = neps.Operation(
            operator=self.function_wrapper,
            args=(self._clamp_by_constant.resample(), "g")
        )

        self._fmoment_bias_correction_rh = neps.Operation(
            operator=self.function_wrapper,
            args=(self.bias_correction, "m", "b1", "t")
        )

        self._smoment_bias_correction_rh = neps.Operation(
            operator=self.function_wrapper,
            args=(self.bias_correction, "v", "b2", "t")
        )

        self._premade_line = neps.Categorical(
            choices=(
                ("m", self._first_moment_rh),
                ("v", self._second_moment_rh),
                ("g", self._clamp_gradient_rh.resample()),
                ("m", self._fmoment_bias_correction_rh.resample()),
                ("v", self._smoment_bias_correction_rh.resample())
            ),
        )


        # Create shared line pool
        self._shared_lines = [
            self._premade_line.resample()
            for _ in range(n_lines[1])
        ]

        # Create line choices (0 to max_lines)
        self._line_choices = tuple(
            tuple(self._shared_lines[:i]) for i in range(n_lines[0], n_lines[1] + 1)
        )

        # Define the optimizer class creation operation
        self.optimizer_cls = neps.Operation(
            operator=self.create_optimizer,
            args=(neps.Categorical(choices=self._line_choices)),
            kwargs={"last_line": self._premade_line.resample()},
        )


    @staticmethod
    def create_optimizer(*lines, last_line):
        """
        Create a custom optimizer class from the given lines.

        Args:
            *lines: Variable number of update rule lines
            last_line: The last line which always updates 'u'

        Returns:
            Custom optimizer class
        """

        class CustomOptimizer(torch.optim.Optimizer):
            """Custom optimizer with flexible update rules."""

            def __init__(
                self, params, lr=1e-3, betas=(0.9, 0.999), variables=(0.1, 0.1), weight_decay=0.0, **kwargs
            ):
                defaults = dict(lr=lr, vars=variables, weight_decay=weight_decay)
                super().__init__(params, defaults)
                self.lr = lr
                self.weight_decay = weight_decay
                self.betas = betas

                # Initialize state for each parameter
                for group in self.param_groups:
                    for p in group.get("params", []):
                        state = self.state.setdefault(p, {})
                        if "v1" not in state:
                            state["v1"] = torch.ones_like(p.data) * variables[0]
                        if "v2" not in state:
                            state["v2"] = torch.ones_like(p.data) * variables[1]
                        if "u" not in state:
                            state["u"] = torch.zeros_like(p.data)

            def step(self, closure=None):
                """Perform a single optimization step."""
                loss = None
                if closure is not None:
                    with torch.enable_grad():
                        loss = closure()

                for group in self.param_groups:
                    for p in group.get("params", []):
                        # Get gradient (or zero if missing)
                        if p.grad is None:
                            d_p = torch.zeros_like(p.data)
                        else:
                            d_p = p.grad + self.weight_decay * p.data

                        state = self.state.setdefault(p, {})

                        def as_tensor(x):
                            """Convert to tensor matching parameter."""
                            if isinstance(x, torch.Tensor):
                                try:
                                    return x.to(device=p.data.device, dtype=p.data.dtype)
                                except Exception:
                                    return x
                            else:
                                return torch.tensor(
                                    x, dtype=p.data.dtype, device=p.data.device
                                )

                        # Build variable dictionary
                        var_dict = {
                            "g": d_p,
                            "w": p.data,
                            "u": state.get("u", torch.zeros_like(p.data)),
                            "v1": state.get("v1", torch.zeros_like(p.data)),
                            "v2": state.get("v2", torch.zeros_like(p.data)),
                            "m": state.get("m", torch.zeros_like(p.data)),
                            "v": state.get("v", torch.zeros_like(p.data)),
                            "b1": self.betas[0],
                            "b2": self.betas[1],
                        }
                        var_dict.update(_runtime_symbol_dict(self, group, p.data))

                        # Evaluate each line + last line and update state
                        for line in lines + (last_line,):
                            target_var, expr = line
                            result = resolve_expression(expr, var_dict)
                            result = as_tensor(result)
                            state[target_var] = result
                            var_dict[target_var] = result

                        # Apply update to parameter
                        p.data = (
                            p.data - self.lr * state["u"]
                        )

                return loss

            def __repr__(self) -> str:
                """String representation of the optimizer."""
                string = f"{self.__class__.__name__}(\n"
                for group in self.param_groups:
                    string += f"  Parameter group:\n"
                    for k, v in group.items():
                        if k != "params":
                            string += f"    {k}: {v}\n"
                string += ")\nLines:\n"
                for line in lines + (last_line,):
                    target_var = line[0]
                    expression = line[1]
                    if isinstance(expression, tuple):
                        string += f"  {target_var:>2} = "
                        expr = expression
                        if callable(expr[0]):
                            if isinstance(expr[0], partial):
                                string += f"{expr[0].func.__name__}("
                                if expr[0].keywords:
                                    string += "{"
                                    for k, v in expr[0].keywords.items():
                                        string += f"{k}={v}, "
                                    string = string.rstrip(", ")
                                    string += "}, "
                            else:
                                string += f"{expr[0].__name__}("
                            for arg in expr[1:]:
                                string += f"{arg}, "
                        else:
                            string += f"{expr[0]}"
                        string = string.rstrip(", ")
                        string += ")\n"
                    else:
                        string += f"  {target_var:>2} = {expression}\n"
                return string

            def get_lines(self):
                """Get the optimizer update lines."""
                return lines + (last_line,)

        return CustomOptimizer



class AdamWExtend(neps.PipelineSpace):
    """
    Neural Optimizer Search space with variable number of lines that compute an additional term for the Adam update.

    This space allows for flexible optimizer definitions with up to max_lines
    of update rules, where each line can assign to either v1 or v2, and the last line always updates u.
    """

    def __init__(
        self,
        term_mode: Literal["add", "mul"] = "add",
        n_lines: Tuple[int, int] = (1, 10),
        fidelity: Tuple[int, int] | None = None,
        learning_rate: Tuple[float, float] | None = None,
        weight_decay: Tuple[float, float] | None = None,
        **_,
    ):
        """
        Initialize the NOS space.

        Args:
            n_lines: Tuple indicating (min_lines, max_lines)
            fidelity: Optional fidelity range (lower, upper)
            learning_rate: Optional learning rate range (lower, upper)
            weight_decay: Optional weight decay range (lower, upper)
        """
        assert (
            n_lines[0] >= 0 and n_lines[1] >= n_lines[0]
        ), f"Invalid n_lines range {n_lines}"

        if fidelity is not None:
            assert (
                len(fidelity) == 2 and 0 <= fidelity[0] < fidelity[1]
            ), f"Invalid fidelity range {fidelity}"
            self.fidelity = neps.IntegerFidelity(lower=fidelity[0], upper=fidelity[1])

        if learning_rate is not None:
            assert (
                len(learning_rate) == 2 and 0 < learning_rate[0] < learning_rate[1]
            ), f"Invalid learning rate range {learning_rate}"
            self.learning_rate = neps.Float(
                lower=learning_rate[0], upper=learning_rate[1], log=True
            )

        if weight_decay is not None:
            weight_decay = (max(weight_decay[0], 0), weight_decay[1])
            assert (
                weight_decay[0] < weight_decay[1] <= 1
            ), f"Invalid weight decay range {weight_decay}"
            self.weight_decay = neps.Float(
                lower=weight_decay[0], upper=weight_decay[1], log=True
            )

        # Define variables that can be used in optimizer
        self._input_variables = neps.Categorical(choices=("w", "g", "v1", "v2"))
        self._output_variables = neps.Categorical(choices=("v1", "v2"))

        # Define constants that can be used in operations
        self._constants = neps.Categorical(choices=(10, 1, 0, 0.1, 0.01, 0.9, 0.99))

        # Define unary operations
        self._unary_funct = neps.Categorical(
            choices=(
                neps.Operation(
                    scale_by_constant,
                    kwargs={"constant": self._constants.resample()},
                ).resample(),
                neps.Operation(
                    clamp_by_constant,
                    kwargs={"constant": self._constants.resample()},
                ).resample(),
                torch.reciprocal,
                torch.square,
                torch.exp,
                torch.sqrt,
                torch.log,
                torch.neg,
            )
        )

        # Define binary operations
        self._binary_funct = neps.Categorical(
            choices=(
                torch.add,
                torch.mul,
                neps.Operation(
                    interpolate,
                    kwargs={"constant": self._constants.resample()},
                ).resample(),
            )
        )

        self._unary_right_hand = neps.Operation(
            operator=self.unaryFunction,
            args=(
                self._unary_funct.resample(),
                self._input_variables.resample(),
            ),
        )

        self._binary_right_hand = neps.Operation(
            operator=self.binaryFunction,
            args=(
                self._binary_funct.resample(),
                self._input_variables.resample(),
                self._input_variables.resample(),
            ),
        )

        # Define possible line structures
        self._line_right_hand = neps.Categorical(
            choices=(
                self._unary_right_hand.resample(),
                self._binary_right_hand.resample(),
            )
        )

        self._u_line_right_hand = neps.Categorical(
            choices=(
                self._output_variables.resample(),
                self._unary_right_hand.resample(),
                self._binary_right_hand.resample(),
            )
        )

        # Create shared line pool
        self._shared_lines = [
            (self._output_variables.resample(), self._line_right_hand.resample())
            for _ in range(n_lines[1])
        ]

        # Create line choices (0 to max_lines)
        self._line_choices = tuple(
            tuple(self._shared_lines[:i]) for i in range(n_lines[0], n_lines[1] + 1)
        )

        # Define the optimizer class creation operation
        self.optimizer_cls = neps.Operation(
            operator=self.create_optimizer,
            args=(neps.Categorical(choices=self._line_choices)),
            kwargs={"last_line": ("u", self._u_line_right_hand),
                    "term_mode": term_mode},
        )

    @staticmethod
    def unaryFunction(operation: Callable, input_value):
        """Package unary operation with its input."""
        return operation, input_value

    @staticmethod
    def binaryFunction(operation: Callable, input1, input2):
        """Package binary operation with its inputs."""
        return operation, input1, input2

    @staticmethod
    def create_optimizer(*lines, last_line, term_mode: Literal["add", "mul"] = "add"):
        """
        Create a custom optimizer class from the given lines.

        Args:
            *lines: Variable number of lines
            last_line: The last line which always updates 'u'

        Returns:
            Custom optimizer class
        """

        class CustomOptimizer(torch.optim.Optimizer):
            """Custom optimizer with flexible update rules."""

            def __init__(
                self, params, lr=0.001, betas=(0.9, 0.95), eps=1e-08, weight_decay=0.01, variables=(0.1, 0.1), **kwargs
            ):
                defaults = dict(lr=lr, vars=variables, weight_decay=weight_decay)
                super().__init__(params, defaults)
                self.lr = lr
                self.weight_decay = weight_decay
                self.beta1 = betas[0]
                self.beta2 = betas[1]
                self.eps = eps

                # Initialize state for each parameter
                for group in self.param_groups:
                    for p in group.get("params", []):
                        state = self.state.setdefault(p, {})
                        if "v1" not in state:
                            state["v1"] = torch.ones_like(p.data) * variables[0]
                        if "v2" not in state:
                            state["v2"] = torch.ones_like(p.data) * variables[1]
                        if "u" not in state:
                            state["u"] = torch.zeros_like(p.data)
                        if "m" not in state:
                            state["m"] = torch.zeros_like(p.data)
                        if "v" not in state:
                            state["v"] = torch.zeros_like(p.data)
                        

            def step(self, closure=None):
                """Perform a single optimization step."""
                loss = None
                if closure is not None:
                    with torch.enable_grad():
                        loss = closure()

                for group in self.param_groups:
                    for p in group.get("params", []):
                        # Get gradient (or zero if missing)
                        if p.grad is None:
                            d_p = torch.zeros_like(p.data)
                        else:
                            d_p = p.grad

                        state = self.state.setdefault(p, {})

                        def as_tensor(x):
                            """Convert to tensor matching parameter."""
                            if isinstance(x, torch.Tensor):
                                try:
                                    return x.to(device=p.data.device, dtype=p.data.dtype)
                                except Exception:
                                    return x
                            else:
                                return torch.tensor(
                                    x, dtype=p.data.dtype, device=p.data.device
                                )

                        # Build variable dictionary
                        var_dict = {
                            "g": d_p,
                            "w": p.data,
                            "u": state.get("u", torch.zeros_like(p.data)),
                            "v1": state.get("v1", torch.zeros_like(p.data)),
                            "v2": state.get("v2", torch.zeros_like(p.data)),
                        }
                        var_dict.update(_runtime_symbol_dict(self, group, p.data))

                        # Apply weight decay
                        p.data -= self.weight_decay * p.data

                        # First perform an AdamW like update
                        state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * d_p
                        state["v"] = self.beta2 * state["v"] + (1 - self.beta2) * (d_p * d_p)
                        m_hat = state["m"] / (1 - self.beta1)
                        v_hat = state["v"] / (1 - self.beta2)

                        # Evaluate each line + last line and update state
                        for line in lines + (last_line,):
                            target_var, expr = line
                            result = resolve_expression(expr, var_dict)
                            result = as_tensor(result)
                            state[target_var] = result
                            var_dict[target_var] = result

                        # Apply update to parameter
                        p.data = (
                            (p.data - self.lr * (m_hat / (torch.sqrt(v_hat) + self.eps)) * state["u"]) if term_mode=="mul" 
                            else (p.data - self.lr * (m_hat / (torch.sqrt(v_hat) + self.eps) + state["u"]))
                        )

                return loss

            def __repr__(self) -> str:
                """String representation of the optimizer."""
                string = f"{self.__class__.__name__}(\n"
                for group in self.param_groups:
                    string += f"  Parameter group:\n"
                    for k, v in group.items():
                        if k != "params":
                            string += f"    {k}: {v}\n"
                string += ")\nLines:\n"
                string += "(fixed) m = beta1 * m + (1 - beta1) * g\n"
                string += "(fixed) v = beta2 * v + (1 - beta2) * (g * g)\n"
                string += "(fixed) m_hat = m / (1 - beta1)\n"
                string += "(fixed) v_hat = v / (1 - beta2)\n"
                for line in lines + (last_line,):
                    target_var = line[0]
                    expression = line[1]
                    if isinstance(expression, tuple):
                        string += f"  {target_var:>2} = "
                        expr = expression
                        if callable(expr[0]):
                            if isinstance(expr[0], partial):
                                string += f"{expr[0].func.__name__}("
                                if expr[0].keywords:
                                    string += "{"
                                    for k, v in expr[0].keywords.items():
                                        string += f"{k}={v}, "
                                    string = string.rstrip(", ")
                                    string += "}, "
                            else:
                                string += f"{expr[0].__name__}("
                            for arg in expr[1:]:
                                string += f"{arg}, "
                        else:
                            string += f"{expr[0]}"
                        string = string.rstrip(", ")
                        string += ")\n"
                    else:
                        string += f"  {target_var:>2} = {expression}\n"
                string += f"(fixed) w = w - lr * (m_hat / (sqrt(v_hat) + eps){') * u' if term_mode=='mul' else ' + u)'}\n"
                return string

            def get_lines(self):
                """Get the sampled optimizer update lines."""
                return lines + (last_line,)

        return CustomOptimizer



class AdamWMore(neps.PipelineSpace):
    """
    Neural Optimizer Search space with variable number of lines that compute an additional term for the Adam update.

    This space allows for flexible optimizer definitions with up to max_lines
    of update rules, where each line can assign to either v1 or v2, and the last line always updates u.
    """

    def __init__(
        self,
        n_lines: Tuple[int, int] = (1, 10),
        fidelity: Tuple[int, int] | None = None,
        learning_rate: Tuple[float, float] | None = None,
        weight_decay: Tuple[float, float] | None = None,
        special_variables: Tuple[str] = ("t", "depth", "layer_type_attention"),
        term_mode: Literal["add", "mul"] = "add",
        **_,
    ):
        """
        Initialize the NOS space.

        Args:
            n_lines: Tuple indicating (min_lines, max_lines)
            fidelity: Optional fidelity range (lower, upper)
            learning_rate: Optional learning rate range (lower, upper)
            weight_decay: Optional weight decay range (lower, upper)
        """
        assert (
            n_lines[0] >= 0 and n_lines[1] >= n_lines[0]
        ), f"Invalid n_lines range {n_lines}"

        if fidelity is not None:
            assert (
                len(fidelity) == 2 and 0 <= fidelity[0] < fidelity[1]
            ), f"Invalid fidelity range {fidelity}"
            self.fidelity = neps.IntegerFidelity(lower=fidelity[0], upper=fidelity[1])

        if learning_rate is not None:
            assert (
                len(learning_rate) == 2 and 0 < learning_rate[0] < learning_rate[1]
            ), f"Invalid learning rate range {learning_rate}"
            self.learning_rate = neps.Float(
                lower=learning_rate[0], upper=learning_rate[1], log=True
            )

        if weight_decay is not None:
            weight_decay = (max(weight_decay[0], 0), weight_decay[1])
            assert (
                weight_decay[0] < weight_decay[1] <= 1
            ), f"Invalid weight decay range {weight_decay}"
            self.weight_decay = neps.Float(
                lower=weight_decay[0], upper=weight_decay[1], log=True
            )

        # Define variables that can be used in optimizer
        self._input_variables = neps.Categorical(choices=(
            neps.Categorical(choices=("w", "g", "v1", "v2")),
            neps.Categorical(choices=special_variables)
            )
        )
        self._output_variables = neps.Categorical(choices=("v1", "v2"))

        # Define constants that can be used in operations
        self._constants = neps.Categorical(choices=(10, 1, 0, 0.1, 0.01, 0.9, 0.99))

        # Define unary operations
        self._unary_funct = neps.Categorical(
            choices=(
                neps.Operation(
                    scale_by_constant,
                    kwargs={"constant": self._constants.resample()},
                ).resample(),
                neps.Operation(
                    clamp_by_constant,
                    kwargs={"constant": self._constants.resample()},
                ).resample(),
                torch.reciprocal,
                torch.square,
                torch.exp,
                torch.sqrt,
                torch.log,
                torch.neg,
            )
        )

        # Define binary operations
        self._binary_funct = neps.Categorical(
            choices=(
                torch.add,
                torch.mul,
                neps.Operation(
                    interpolate,
                    kwargs={"constant": self._constants.resample()},
                ).resample(),
            )
        )

        self._unary_right_hand = neps.Operation(
            operator=self.unaryFunction,
            args=(
                self._unary_funct.resample(),
                self._input_variables.resample(),
            ),
        )

        self._binary_right_hand = neps.Operation(
            operator=self.binaryFunction,
            args=(
                self._binary_funct.resample(),
                self._input_variables.resample(),
                self._input_variables.resample(),
            ),
        )

        # Define possible line structures
        self._line_right_hand = neps.Categorical(
            choices=(
                self._unary_right_hand.resample(),
                self._binary_right_hand.resample(),
            )
        )

        self._u_line_right_hand = neps.Categorical(
            choices=(
                self._output_variables.resample(),
                self._unary_right_hand.resample(),
                self._binary_right_hand.resample(),
            )
        )

        # Create shared line pools
        self._shared_lines_1 = [
            (self._output_variables.resample(), self._line_right_hand.resample())
            for _ in range(n_lines[1])
        ]

        self._shared_lines_2 = [
            (self._output_variables.resample(), self._line_right_hand.resample())
            for _ in range(n_lines[1])
        ]

        # Create line choices for different line counts
        self._line_choices_1 = tuple(
            tuple(self._shared_lines_1[:i]) for i in range(n_lines[0], n_lines[1] + 1)
        )

        self._line_choices_2 = tuple(
            tuple(self._shared_lines_2[:i]) for i in range(n_lines[0], n_lines[1] + 1)
        )

        self._lines1 = neps.Categorical(choices=self._line_choices_1)
        self._lines2 = neps.Categorical(choices=self._line_choices_2)

        # Define the optimizer class creation operation
        # Pass all lines via args (unpacked) so NEPS resolves Operations properly
        self.optimizer_cls = neps.Operation(
            operator=self.create_optimizer,
            args=self._lines1,#self._lines2),
            kwargs={#"lines1": self._lines1.resample(),
                    "u1": ("u1", self._u_line_right_hand.resample()),
                    "term_mode": term_mode,
                    # "lines2": neps.Categorical(choices=self._line_choices_2),
                    # "u2": self._u_line_right_hand.resample(),
            },
        )

    @staticmethod
    def unaryFunction(operation: Callable, input_value):
        """Package unary operation with its input."""
        return operation, input_value

    @staticmethod
    def binaryFunction(operation: Callable, input1, input2):
        """Package binary operation with its inputs."""
        return operation, input1, input2

    @staticmethod
    def create_optimizer(*lines, u1, term_mode: Literal["add", "mul"] = "add"):
        """
        Create a custom optimizer class from the given lines.

        Args:
            lines: The lines.

        Returns:
            Custom optimizer class
        """

        # from pprint import pprint
        # pprint(lines)
        # pprint(u1)
        lines+=(u1,)


        class CustomOptimizer(torch.optim.Optimizer):
            """Custom optimizer with flexible update rules."""

            def __init__(
                self, params, lr=0.001, betas=(0.9, 0.95), eps=1e-08, weight_decay=0.01, variables=(0.1, 0.1), **kwargs
            ):
                defaults = dict(lr=lr, vars=variables, weight_decay=weight_decay)
                super().__init__(params, defaults)
                self.lr = lr
                self.weight_decay = weight_decay
                self.beta1 = betas[0]
                self.beta2 = betas[1]
                self.eps = eps

                # pprint(lines)
                # pprint(u1)

                # new_lines = []
                # for line in lines1:
                #     print("Processing line:", line)
                #     if not isinstance(line[0], tuple):
                #         print("Line is not a tuple, adding directly:", line)
                #         new_lines.append((line[0], line[1]))
                #     else:
                #         for subline in line:
                #             print("Processing subline:", subline)
                #             print("Append subline to new_lines:", subline[0])
                #             new_lines.append((subline[0], (subline[1])))
                self.opt_lines = lines

                # Initialize state for each parameter
                for group in self.param_groups:
                    for p in group.get("params", []):
                        state = self.state.setdefault(p, {})
                        if "v1" not in state:
                            state["v1"] = torch.ones_like(p.data) * variables[0]
                        if "v2" not in state:
                            state["v2"] = torch.ones_like(p.data) * variables[1]
                        if "u1" not in state:
                            state["u1"] = torch.zeros_like(p.data)
                        if "u2" not in state:
                            state["u2"] = torch.zeros_like(p.data)
                        if "m" not in state:
                            state["m"] = torch.zeros_like(p.data)
                        if "v" not in state:
                            state["v"] = torch.zeros_like(p.data)
                        

            def step(self, closure=None):
                """Perform a single optimization step."""
                loss = None
                if closure is not None:
                    with torch.enable_grad():
                        loss = closure()

                for group in self.param_groups:
                    for p in group.get("params", []):
                        # Get gradient (or zero if missing)
                        if p.grad is None:
                            d_p = torch.zeros_like(p.data)
                        else:
                            d_p = p.grad

                        state = self.state.setdefault(p, {})

                        def as_tensor(x):
                            """Convert to tensor matching parameter."""
                            if isinstance(x, torch.Tensor):
                                try:
                                    return x.to(device=p.data.device, dtype=p.data.dtype)
                                except Exception:
                                    return x
                            else:
                                return torch.tensor(
                                    x, dtype=p.data.dtype, device=p.data.device
                                )

                        # Build variable dictionary
                        var_dict = {
                            "g": d_p,
                            "w": p.data,
                            "u1": state.get("u1", torch.zeros_like(p.data)),
                            "u2": state.get("u2", torch.zeros_like(p.data)),
                            "v1": state.get("v1", torch.zeros_like(p.data)),
                            "v2": state.get("v2", torch.zeros_like(p.data)),
                        }
                        var_dict.update(_runtime_symbol_dict(self, group, p.data))
                        # print(_runtime_symbol_dict(self, group, p.data))

                        # Apply weight decay
                        p.data -= self.weight_decay * p.data

                        # First perform an AdamW like update
                        state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * d_p
                        state["v"] = self.beta2 * state["v"] + (1 - self.beta2) * (d_p * d_p)
                        m_hat = state["m"] / (1 - self.beta1)
                        v_hat = state["v"] / (1 - self.beta2)

                        # Evaluate each line + last line and update state
                        for line in self.opt_lines:
                            # print("Evaluating line:", line)
                            target_var, expr = line
                            result = resolve_expression(expr, var_dict)
                            result = as_tensor(result)
                            state[target_var] = result
                            var_dict[target_var] = result

                        # Apply update to parameter
                        p.data = (
                            (p.data - self.lr * (m_hat / (torch.sqrt(v_hat) + self.eps)) * state["u1"]) if term_mode=="mul" 
                            else (p.data - self.lr * (m_hat / (torch.sqrt(v_hat) + self.eps) + state["u1"]))
                        )

                return loss

            def __repr__(self) -> str:
                """String representation of the optimizer."""
                string = f"{self.__class__.__name__}(\n"
                for group in self.param_groups:
                    string += f"  Parameter group:\n"
                    for k, v in group.items():
                        if k != "params":
                            string += f"    {k}: {v}\n"
                string += ")\nLines:\n"
                string += "(fixed) m = beta1 * m + (1 - beta1) * g\n"
                string += "(fixed) v = beta2 * v + (1 - beta2) * (g * g)\n"
                string += "(fixed) m_hat = m / (1 - beta1)\n"
                string += "(fixed) v_hat = v / (1 - beta2)\n"
                for line in self.opt_lines:
                    target_var = line[0]
                    expression = line[1]
                    if isinstance(expression, tuple):
                        string += f"  {target_var:>2} = "
                        expr = expression
                        if callable(expr[0]):
                            if isinstance(expr[0], partial):
                                string += f"{expr[0].func.__name__}("
                                if expr[0].keywords:
                                    string += "{"
                                    for k, v in expr[0].keywords.items():
                                        string += f"{k}={v}, "
                                    string = string.rstrip(", ")
                                    string += "}, "
                            else:
                                string += f"{expr[0].__name__}("
                            for arg in expr[1:]:
                                string += f"{arg}, "
                        else:
                            string += f"{expr[0]}"
                        string = string.rstrip(", ")
                        string += ")\n"
                    else:
                        string += f"  {target_var:>2} = {expression}\n"
                string += f"(fixed) w = w - lr * (m_hat / (sqrt(v_hat) + eps) + u1) * u2\n"
                return string

            def get_lines(self):
                """Get the sampled optimizer update lines."""
                return self.opt_lines

        return CustomOptimizer



class PremadeModules(neps.PipelineSpace):
    """
    Modular optimizer search space with high-level building blocks.
    
    This space allows construction of optimizers like Adam, RMSProp, Adagrad, etc.
    by selecting variants for each operation type (momentum, second moment, bias correction, etc.).
    """

    def __init__(
        self,
        fidelity: Tuple[int, int] | None = None,
        learning_rate: Tuple[float, float] | None = None,
        weight_decay: Tuple[float, float] | None = None,
        **_,
    ):
        """
        Initialize the modular optimizer space.

        Args:
            fidelity: Optional fidelity range (lower, upper)
            learning_rate: Optional learning rate range (lower, upper)
            weight_decay: Optional weight decay range (lower, upper)
        """
        if fidelity is not None:
            assert (
                len(fidelity) == 2 and 0 <= fidelity[0] < fidelity[1]
            ), f"Invalid fidelity range {fidelity}"
            self.fidelity = neps.IntegerFidelity(lower=fidelity[0], upper=fidelity[1])

        self._bias_correct = neps.Categorical(
            choices=(True, False)
        )

        self._momentum_beta = neps.Float(lower=0.8, upper=0.999)

        # First moment (momentum) configuration
        self._momentum = neps.Categorical(
            choices=(("none", None, False), 
                     ("standard", self._momentum_beta, self._bias_correct.resample()), 
                     ("heavy_ball", self._momentum_beta, self._bias_correct.resample()), 
                     ("nesterov", self._momentum_beta, self._bias_correct.resample()))
        )


        self._second_moment_beta = neps.Float(lower=0.9, upper=0.9999)
        # Second moment configuration
        self._second_moment = neps.Categorical(
            choices=(("none", None, False), 
                     ("ema_squared", self._second_moment_beta, self._bias_correct.resample()), 
                     ("accumulate", self._second_moment_beta, self._bias_correct.resample()), 
                     ("centered", self._second_moment_beta, self._bias_correct.resample()), 
                     ("amsgrad", self._second_moment_beta, self._bias_correct.resample()))
        )

        
        # Weight decay value
        if weight_decay is not None:
            weight_decay = (max(weight_decay[0], 1e-8), weight_decay[1])
            assert (
                1e-8 <= weight_decay[0] < weight_decay[1]
            ), f"Invalid weight decay range {weight_decay}"
            self.weight_decay = neps.Float(
                lower=weight_decay[0], upper=weight_decay[1], log=True
            )
            # Weight decay type
            self._weight_decay_type = neps.Categorical(
                choices=("none", "coupled_l2", "decoupled")
            )
        else:
            self._weight_decay_type = "none"

        # Update rule
        self._update_rule = neps.Categorical(
            choices=(
                "sgd",
                "adaptive_momentum",
                "adaptive_gradient",
                "adadelta",
                "sign_momentum",
                "normalized_gradient",
            )
        )

        # Learning rate
        if learning_rate is not None:
            assert (
                len(learning_rate) == 2 and 0 < learning_rate[0] < learning_rate[1]
            ), f"Invalid learning rate range {learning_rate}"
            self.learning_rate = neps.Float(
                lower=learning_rate[0], upper=learning_rate[1], log=True
            )

        # Epsilon for numerical stability
        self._eps = neps.Float(lower=1e-10, upper=1e-6, log=True)

        # Create optimizer operation
        self.optimizer_cls = neps.Operation(
            operator=self.create_optimizer,
            kwargs={
                "first_moment": self._momentum,
                "second_moment": self._second_moment,
                "weight_decay_type": self._weight_decay_type,
                "update_rule": self._update_rule,
                "eps": self._eps,
            },
        )

    @staticmethod
    def create_optimizer(
        first_moment,
        second_moment,
        weight_decay_type,
        update_rule,
        eps,
    ):
        """
        Create a custom optimizer class from the modular configuration.

        Returns:
            Custom optimizer class
        """

        class ModularOptimizer(torch.optim.Optimizer):
            """Modular optimizer built from high-level blocks."""

            def __init__(self, params, lr=1e-3, weight_decay=0, **kwargs):
                defaults = dict(lr=lr)
                super().__init__(params, defaults)
                self.lr = lr
                self.beta1 = first_moment[1]
                self.beta2 = second_moment[1]
                self.weight_decay = weight_decay
                self.eps = eps

                # Store configuration
                self.first_moment_type = first_moment[0]
                self.second_moment_type = second_moment[0]
                self.first_bias_correction = first_moment[2]
                self.second_bias_correction = second_moment[2]
                self.weight_decay_type = weight_decay_type
                self.update_rule = update_rule

                # Initialize state for each parameter
                for group in self.param_groups:
                    for p in group["params"]:
                        state = self.state[p]
                        state["step"] = 0
                        state["m"] = torch.zeros_like(p.data)  # First moment
                        state["v"] = torch.zeros_like(p.data)  # Second moment
                        if self.second_moment_type == "centered":
                            state["g_avg"] = torch.zeros_like(p.data)  # Gradient average
                        if self.second_moment_type == "amsgrad":
                            state["v_max"] = torch.zeros_like(p.data)
                        if self.update_rule == "adadelta":
                            state["delta_acc"] = torch.zeros_like(p.data)

            def step(self, closure=None):
                """Perform a single optimization step."""
                loss = None
                if closure is not None:
                    loss = closure()

                for group in self.param_groups:
                    for p in group["params"]:
                        if p.grad is None:
                            continue

                        grad = p.grad.data
                        state = self.state[p]
                        state["step"] += 1

                        # 1. Apply coupled weight decay (modify gradient)
                        if self.weight_decay_type == "coupled_l2":
                            grad = grad + self.weight_decay * p.data

                        # 2. Compute first moment (momentum)
                        if self.first_moment_type == "standard":
                            state["m"] = (
                                self.beta1 * state["m"] + (1 - self.beta1) * grad
                            )
                        elif self.first_moment_type == "heavy_ball":
                            state["m"] = self.beta1 * state["m"] + grad
                        elif self.first_moment_type == "nesterov":
                            state["m"] = self.beta1 * state["m"] + grad
                        # else: momentum_type == "none", m stays zero

                        # 3. Compute second moment
                        if self.second_moment_type == "ema_squared":
                            state["v"] = (
                                self.beta2 * state["v"]
                                + (1 - self.beta2) * grad**2
                            )
                        elif self.second_moment_type == "accumulate":
                            state["v"] = state["v"] + grad**2
                        elif self.second_moment_type == "centered":
                            state["g_avg"] = (
                                self.beta2 * state["g_avg"] + (1 - self.beta2) * grad
                            )
                            state["v"] = (
                                self.beta2 * state["v"]
                                + (1 - self.beta2) * grad**2
                            )
                        elif self.second_moment_type == "amsgrad":
                            v_new = (
                                self.beta2 * state["v"]
                                + (1 - self.beta2) * grad**2
                            )
                            state["v_max"] = torch.maximum(state["v_max"], v_new)
                            state["v"] = v_new
                        # else: second_moment_type == "none", v stays zero

                        # 4. Apply bias correction
                        m_hat = state["m"]
                        v_hat = state["v"]

                        if self.first_bias_correction:
                            m_hat = m_hat / (1 - self.beta1 ** state["step"])

                        if self.second_bias_correction:
                            if self.second_moment_type == "amsgrad":
                                v_hat = state["v_max"] / (1 - self.beta2 ** state["step"])
                            else:
                                v_hat = v_hat / (1 - self.beta2 ** state["step"])

                        # For centered RMSProp
                        if self.second_moment_type == "centered":
                            g_avg_hat = state["g_avg"]
                            if self.second_bias_correction:
                                g_avg_hat = g_avg_hat / (1 - self.beta2 ** state["step"])
                            v_hat = v_hat - g_avg_hat**2

                        # For AMSGrad without bias correction
                        if (
                            self.second_moment_type == "amsgrad"
                            and not self.second_bias_correction
                        ):
                            v_hat = state["v_max"]

                        # 5. Compute update based on rule
                        if self.update_rule == "sgd":
                            # Simple SGD with momentum
                            if self.first_moment_type == "nesterov":
                                # Nesterov: use look-ahead position
                                update = -(self.lr * (self.beta1 * m_hat + grad))
                            else:
                                update = -self.lr * m_hat if self.first_moment_type != "none" else -self.lr * grad

                        elif self.update_rule == "adaptive_momentum":
                            # Adam-style: scale momentum by second moment
                            if self.first_moment_type == "nesterov":
                                numerator = self.beta1 * m_hat + grad
                            else:
                                numerator = m_hat if self.first_moment_type != "none" else grad
                            update = -self.lr * numerator / (torch.sqrt(v_hat) + self.eps)

                        elif self.update_rule == "adaptive_gradient":
                            # RMSProp-style: scale gradient by second moment
                            update = -self.lr * grad / (torch.sqrt(v_hat) + self.eps)

                        elif self.update_rule == "adadelta":
                            # AdaDelta: no learning rate, use delta accumulator
                            update = (
                                -(torch.sqrt(state["delta_acc"] + self.eps))
                                / (torch.sqrt(v_hat) + self.eps)
                                * grad
                            )
                            state["delta_acc"] = (
                                self.beta2 * state["delta_acc"]
                                + (1 - self.beta2) * update**2
                            )

                        elif self.update_rule == "sign_momentum":
                            # SignSGD variant
                            if self.first_moment_type != "none":
                                update = -self.lr * torch.sign(m_hat)
                            else:
                                update = -self.lr * torch.sign(grad)

                        elif self.update_rule == "normalized_gradient":
                            # Normalized gradient descent
                            grad_norm = torch.norm(grad)
                            if grad_norm > 0:
                                update = -self.lr * grad / grad_norm
                            else:
                                update = -self.lr * grad

                        else:
                            raise ValueError(f"Unknown update rule: {self.update_rule}")

                        # 6. Apply update
                        p.data.add_(update)

                        # 7. Apply decoupled weight decay (modify weights directly)
                        if self.weight_decay_type == "decoupled":
                            p.data.mul_(1 - self.lr * self.weight_decay)

                return loss

            def __repr__(self) -> str:
                """String representation of the optimizer."""
                config_str = f"{self.__class__.__name__}(\n"
                config_str += f"  First Momentum: {self.first_moment_type} (β1={self.beta1})\n"
                config_str += f"  Second Moment: {self.second_moment_type} (β2={self.beta2})\n"
                config_str += f"  First Bias Correction: {self.first_bias_correction}\n"
                config_str += f"  Second Bias Correction: {self.second_bias_correction}\n"
                config_str += f"  Weight Decay: {self.weight_decay_type} (λ={self.weight_decay})\n"
                config_str += f"  Update Rule: {self.update_rule}\n"
                config_str += f"  Learning Rate: γ={self.lr:.6f}\n"
                config_str += f"  Epsilon: ε={self.eps:.2e}\n"
                config_str += ")"
                return config_str

            def get_config(self):
                """Get optimizer configuration."""
                return {
                    "momentum_type": self.first_moment_type,
                    "momentum_beta": self.beta1,
                    "second_moment_type": self.second_moment_type,
                    "second_moment_beta": self.beta2,
                    "weight_decay_type": self.weight_decay_type,
                    "weight_decay": self.weight_decay,
                    "update_rule": self.update_rule,
                    "lr": self.lr,
                    "eps": self.eps,
                }

        return ModularOptimizer



class NOSSpaceNLinesU(neps.PipelineSpace):
    """
    Neural Optimizer Search space with variable number of lines and fixed last line updating u.

    This space allows for flexible optimizer definitions with up to max_lines
    of update rules, where each line can assign to either v1 or v2, and the last line always updates u.
    """

    def __init__(
        self,
        n_lines: Tuple[int, int] = (1, 10),
        fidelity: Tuple[int, int] | None = None,
        learning_rate: Tuple[float, float] | None = None,
        weight_decay: Tuple[float, float] | None = None,
        **_,
    ):
        """
        Initialize the NOS space.

        Args:
            n_lines: Tuple indicating (min_lines, max_lines)
            fidelity: Optional fidelity range (lower, upper)
            learning_rate: Optional learning rate range (lower, upper)
            weight_decay: Optional weight decay range (lower, upper)
        """
        assert (
            n_lines[0] >= 0 and n_lines[1] >= n_lines[0]
        ), f"Invalid n_lines range {n_lines}"

        if fidelity is not None:
            assert (
                len(fidelity) == 2 and 0 <= fidelity[0] < fidelity[1]
            ), f"Invalid fidelity range {fidelity}"
            self.fidelity = neps.IntegerFidelity(lower=fidelity[0], upper=fidelity[1])

        if learning_rate is not None:
            assert (
                len(learning_rate) == 2 and 0 < learning_rate[0] < learning_rate[1]
            ), f"Invalid learning rate range {learning_rate}"
            self.learning_rate = neps.Float(
                lower=learning_rate[0], upper=learning_rate[1], log=True
            )

        if weight_decay is not None:
            weight_decay = (max(weight_decay[0], 1e-8), weight_decay[1])
            assert (
                1e-8 <= weight_decay[0] < weight_decay[1]
            ), f"Invalid weight decay range {weight_decay}"
            self.weight_decay = neps.Float(
                lower=weight_decay[0], upper=weight_decay[1], log=True
            )

        # Define variables that can be used in optimizer
        self._input_variables = neps.Categorical(choices=("w", "g", "v1", "v2"))
        self._output_variables = neps.Categorical(choices=("v1", "v2"))

        # Define constants that can be used in operations
        self._constants = neps.Categorical(choices=(10, 1, 0, 0.1, 0.01, 0.9, 0.99))

        # Define unary operations
        self._unary_funct = neps.Categorical(
            choices=(
                neps.Operation(
                    scale_by_constant,
                    kwargs={"constant": self._constants.resample()},
                ).resample(),
                neps.Operation(
                    clamp_by_constant,
                    kwargs={"constant": self._constants.resample()},
                ).resample(),
                torch.reciprocal,
                torch.square,
                torch.exp,
                torch.sqrt,
                torch.log,
                torch.neg,
            )
        )

        # Define binary operations
        self._binary_funct = neps.Categorical(
            choices=(
                torch.add,
                torch.mul,
                neps.Operation(
                    interpolate,
                    kwargs={"constant": self._constants.resample()},
                ).resample(),
            )
        )

        self._unary_right_hand = neps.Operation(
            operator=self.unaryFunction,
            args=(
                self._unary_funct.resample(),
                self._input_variables.resample(),
            ),
        )

        self._binary_right_hand = neps.Operation(
            operator=self.binaryFunction,
            args=(
                self._binary_funct.resample(),
                self._input_variables.resample(),
                self._input_variables.resample(),
            ),
        )

        # Define possible line structures
        self._line_right_hand = neps.Categorical(
            choices=(
                self._unary_right_hand.resample(),
                self._binary_right_hand.resample(),
            )
        )

        self._u_line_right_hand = neps.Categorical(
            choices=(
                self._output_variables.resample(),
                self._unary_right_hand.resample(),
                self._binary_right_hand.resample(),
            )
        )

        # Create shared line pool
        self._shared_lines = [
            (self._output_variables.resample(), self._line_right_hand.resample())
            for _ in range(n_lines[1])
        ]

        # Create line choices (0 to max_lines)
        self._line_choices = tuple(
            tuple(self._shared_lines[:i]) for i in range(n_lines[0], n_lines[1] + 1)
        )

        # Define the optimizer class creation operation
        self.optimizer_cls = neps.Operation(
            operator=self.create_optimizer,
            args=(neps.Categorical(choices=self._line_choices)),
            kwargs={"last_line": ("u", self._u_line_right_hand)},
        )

    @staticmethod
    def unaryFunction(operation: Callable, input_value):
        """Package unary operation with its input."""
        return operation, input_value

    @staticmethod
    def binaryFunction(operation: Callable, input1, input2):
        """Package binary operation with its inputs."""
        return operation, input1, input2

    @staticmethod
    def create_optimizer(*lines, last_line):
        """
        Create a custom optimizer class from the given lines.

        Args:
            *lines: Variable number of update rule lines
            last_line: The last line which always updates 'u'

        Returns:
            Custom optimizer class
        """

        class CustomOptimizer(torch.optim.Optimizer):
            """Custom optimizer with flexible update rules."""

            def __init__(
                self, params, lr=1e-3, variables=(0.1, 0.1), weight_decay=0.0, **kwargs
            ):
                defaults = dict(lr=lr, vars=variables, weight_decay=weight_decay)
                super().__init__(params, defaults)
                self.lr = lr
                self.weight_decay = weight_decay

                # Initialize state for each parameter
                for group in self.param_groups:
                    for p in group.get("params", []):
                        state = self.state.setdefault(p, {})
                        if "v1" not in state:
                            state["v1"] = torch.ones_like(p.data) * variables[0]
                        if "v2" not in state:
                            state["v2"] = torch.ones_like(p.data) * variables[1]
                        if "u" not in state:
                            state["u"] = torch.zeros_like(p.data)

            def step(self, closure=None):
                """Perform a single optimization step."""
                loss = None
                if closure is not None:
                    with torch.enable_grad():
                        loss = closure()

                for group in self.param_groups:
                    for p in group.get("params", []):
                        # Get gradient (or zero if missing)
                        if p.grad is None:
                            d_p = torch.zeros_like(p.data)
                        else:
                            d_p = p.grad

                        state = self.state.setdefault(p, {})

                        def as_tensor(x):
                            """Convert to tensor matching parameter."""
                            if isinstance(x, torch.Tensor):
                                try:
                                    return x.to(device=p.data.device, dtype=p.data.dtype)
                                except Exception:
                                    return x
                            else:
                                return torch.tensor(
                                    x, dtype=p.data.dtype, device=p.data.device
                                )

                        # Build variable dictionary
                        var_dict = {
                            "g": d_p,
                            "w": p.data,
                            "u": state.get("u", torch.zeros_like(p.data)),
                            "v1": state.get("v1", torch.zeros_like(p.data)),
                            "v2": state.get("v2", torch.zeros_like(p.data)),
                        }
                        var_dict.update(_runtime_symbol_dict(self, group, p.data))

                        # Apply weight decay
                        p.data -= self.weight_decay * p.data

                        # Evaluate each line + last line and update state
                        for line in lines + (last_line,):
                            target_var, expr = line
                            result = resolve_expression(expr, var_dict)
                            result = as_tensor(result)
                            state[target_var] = result
                            var_dict[target_var] = result

                        # Apply update to parameter
                        p.data = (
                            p.data - self.lr * state["u"]
                        )

                return loss

            def __repr__(self) -> str:
                """String representation of the optimizer."""
                string = f"{self.__class__.__name__}(\n"
                for group in self.param_groups:
                    string += f"  Parameter group:\n"
                    for k, v in group.items():
                        if k != "params":
                            string += f"    {k}: {v}\n"
                string += ")\nLines:\n"
                for line in lines + (last_line,):
                    target_var = line[0]
                    expression = line[1]
                    if isinstance(expression, tuple):
                        string += f"  {target_var:>2} = "
                        expr = expression
                        if callable(expr[0]):
                            if isinstance(expr[0], partial):
                                string += f"{expr[0].func.__name__}("
                                if expr[0].keywords:
                                    string += "{"
                                    for k, v in expr[0].keywords.items():
                                        string += f"{k}={v}, "
                                    string = string.rstrip(", ")
                                    string += "}, "
                            else:
                                string += f"{expr[0].__name__}("
                            for arg in expr[1:]:
                                string += f"{arg}, "
                        else:
                            string += f"{expr[0]}"
                        string = string.rstrip(", ")
                        string += ")\n"
                    else:
                        string += f"  {target_var:>2} = {expression}\n"
                return string

            def get_lines(self):
                """Get the optimizer update lines."""
                return lines + (last_line,)

        return CustomOptimizer


class NOSSpaceMaxLines(neps.PipelineSpace):
    """
    Neural Optimizer Search space with variable number of lines.

    This space allows for flexible optimizer definitions with up to max_lines
    of update rules, where each line can assign to any of the state variables
    (w, g, u, v1, v2).
    """

    def __init__(
        self,
        max_lines=10,
        fidelity: Tuple[int, int] | None = None,
        learning_rate: Tuple[float, float] | None = None,
        weight_decay: Tuple[float, float] | None = None,
        **_,
    ):
        """
        Initialize the NOS space.

        Args:
            max_lines: Maximum number of optimizer update lines
            fidelity: Optional fidelity range (lower, upper)
            learning_rate: Optional learning rate range (lower, upper)
            weight_decay: Optional weight decay range (lower, upper)
        """

        if fidelity is not None:
            assert (
                len(fidelity) == 2 and 0 <= fidelity[0] < fidelity[1]
            ), f"Invalid fidelity range {fidelity}"
            self.fidelity = neps.IntegerFidelity(lower=fidelity[0], upper=fidelity[1])

        if learning_rate is not None:
            assert (
                len(learning_rate) == 2 and 0 < learning_rate[0] < learning_rate[1]
            ), f"Invalid learning rate range {learning_rate}"
            self.learning_rate = neps.Float(
                lower=learning_rate[0], upper=learning_rate[1], log=True
            )

        if weight_decay is not None:
            weight_decay = (max(weight_decay[0], 1e-8), weight_decay[1])
            assert (
                1e-8 <= weight_decay[0] < weight_decay[1]
            ), f"Invalid weight decay range {weight_decay}"
            self.weight_decay = neps.Float(
                lower=weight_decay[0], upper=weight_decay[1], log=True
            )
        # Define variables that can be used in optimizer
        self._variables = neps.Categorical(choices=("w", "g", "u", "v1", "v2"))

        # Define constants that can be used in operations
        self._constants = neps.Categorical(choices=(10, 1, 0, 0.1, 0.01, 0.9, 0.99))

        # Define unary operations
        self._unary_funct = neps.Categorical(
            choices=(
                neps.Operation(
                    scale_by_constant,
                    kwargs={"constant": self._constants.resample()},
                ).resample(),
                neps.Operation(
                    clamp_by_constant,
                    kwargs={"constant": self._constants.resample()},
                ).resample(),
                torch.reciprocal,
                torch.square,
                torch.exp,
                torch.sqrt,
                torch.log,
                torch.neg,
            )
        )

        # Define binary operations
        self._binary_funct = neps.Categorical(
            choices=(
                torch.add,
                torch.mul,
                neps.Operation(
                    interpolate,
                    kwargs={"constant": self._constants.resample()},
                ).resample(),
            )
        )

        # Define possible line structures
        self._line = neps.Categorical(
            choices=(
                (self._variables.resample(), self._variables.resample()),
                (
                    self._variables.resample(),
                    neps.Operation(
                        operator=self.unaryFunction,
                        args=(
                            self._unary_funct.resample(),
                            self._variables.resample(),
                        ),
                    ).resample(),
                ),
                (
                    self._variables.resample(),
                    neps.Operation(
                        operator=self.binaryFunction,
                        args=(
                            self._binary_funct.resample(),
                            self._variables.resample(),
                            self._variables.resample(),
                        ),
                    ).resample(),
                ),
            ),
        )

        # Create shared line pool
        self._shared_lines = [self._line.resample() for _ in range(max_lines)]

        # Create line choices (1 to max_lines)
        self._line_choices = tuple(
            tuple(self._shared_lines[:i]) for i in range(1, max_lines + 1)
        )

        # Define the optimizer class creation operation
        self.optimizer_cls = neps.Operation(
            operator=self.create_optimizer,
            args=(neps.Categorical(choices=self._line_choices)),
        )

    @staticmethod
    def unaryFunction(operation: Callable, input_value):
        """Package unary operation with its input."""
        return operation, input_value

    @staticmethod
    def binaryFunction(operation: Callable, input1, input2):
        """Package binary operation with its inputs."""
        return operation, input1, input2

    def create_optimizer(self, *lines):
        """
        Create a custom optimizer class from the given lines.

        Args:
            *lines: Variable number of update rule lines

        Returns:
            Custom optimizer class
        """

        class MaxLinesOptimizer(torch.optim.Optimizer):
            """Custom optimizer with flexible update rules."""

            def __init__(
                self, params, lr=1e-3, variables=(0.1, 0.1), weight_decay=0.0, **kwargs
            ):
                defaults = dict(lr=lr, vars=variables, weight_decay=weight_decay)
                super().__init__(params, defaults)
                self.lr = lr
                self.weight_decay = weight_decay

                # Initialize state for each parameter
                for group in self.param_groups:
                    for p in group.get("params", []):
                        state = self.state.setdefault(p, {})
                        if "v1" not in state:
                            state["v1"] = torch.ones_like(p.data) * variables[0]
                        if "v2" not in state:
                            state["v2"] = torch.ones_like(p.data) * variables[1]
                        if "u" not in state:
                            state["u"] = torch.zeros_like(p.data)

            def step(self, closure=None):
                """Perform a single optimization step."""
                loss = None
                if closure is not None:
                    with torch.enable_grad():
                        loss = closure()

                for group in self.param_groups:
                    for p in group.get("params", []):
                        # Get gradient (or zero if missing)
                        if p.grad is None:
                            d_p = torch.zeros_like(p.data)
                        else:
                            d_p = p.grad

                        state = self.state.setdefault(p, {})

                        def as_tensor(x):
                            """Convert to tensor matching parameter."""
                            if isinstance(x, torch.Tensor):
                                try:
                                    return x.to(device=p.data.device, dtype=p.data.dtype)
                                except Exception:
                                    return x
                            else:
                                return torch.tensor(
                                    x, dtype=p.data.dtype, device=p.data.device
                                )

                        # Build variable dictionary
                        var_dict = {
                            "g": d_p,
                            "w": p.data,
                            "u": state.get("u", torch.zeros_like(p.data)),
                            "v1": state.get("v1", torch.zeros_like(p.data)),
                            "v2": state.get("v2", torch.zeros_like(p.data)),
                        }
                        var_dict.update(_runtime_symbol_dict(self, group, p.data))

                        # Apply weight decay
                        p.data -= self.weight_decay * p.data

                        # Evaluate each line and update state
                        for line in lines:
                            target_var, expr = line
                            result = resolve_expression(expr, var_dict)
                            result = as_tensor(result)
                            state[target_var] = result
                            var_dict[target_var] = result

                        # Apply update to parameter
                        p.data = (
                            p.data - self.lr * state["u"]
                        )

                return loss

            def __repr__(self) -> str:
                """String representation of the optimizer."""
                string = f"{self.__class__.__name__}(\n"
                for group in self.param_groups:
                    string += f"  Parameter group:\n"
                    for k, v in group.items():
                        if k != "params":
                            string += f"    {k}: {v}\n"
                string += ")\nLines:\n"
                for line in lines:
                    target_var = line[0]
                    expression = line[1]
                    if isinstance(expression, tuple):
                        string += f"  {target_var} = "
                        expr = expression
                        if callable(expr[0]):
                            if isinstance(expr[0], partial):
                                string += f"{expr[0].func.__name__}("
                                if expr[0].keywords:
                                    string += "{"
                                    for k, v in expr[0].keywords.items():
                                        string += f"{k}={v}, "
                                    string = string.rstrip(", ")
                                    string += "}, "
                            else:
                                string += f"{expr[0].__name__}("
                            for arg in expr[1:]:
                                string += f"{arg}, "
                        else:
                            string += f"{expr[0]}"
                        string = string.rstrip(", ")
                        string += ")\n"
                    else:
                        string += f"  {target_var}={expression[0]}\n"
                return string

            def get_lines(self):
                """Get the optimizer update lines."""
                return lines

        return MaxLinesOptimizer


class NOSSpace3Lines(neps.PipelineSpace):
    """
    Neural Optimizer Search space with exactly 3 lines.

    This is a simpler variant with fixed 3-line structure,
    where each line updates v1, v2, and u in sequence.
    """

    def __init__(
        self,
        fidelity: Tuple[int, int] | None = None,
        learning_rate: Tuple[float, float] | None = None,
        weight_decay: Tuple[float, float] | None = None,
        **_,
    ):
        """Initialize the 3-line NOS space.
        Args:
            fidelity: Optional fidelity range (lower, upper)
            learning_rate: Optional learning rate range (lower, upper)
            weight_decay: Optional weight decay range (lower, upper)
        """

        if fidelity is not None:
            assert (
                len(fidelity) == 2 and 0 <= fidelity[0] < fidelity[1]
            ), f"Invalid fidelity range {fidelity}"
            self.fidelity = neps.IntegerFidelity(lower=fidelity[0], upper=fidelity[1])

        if learning_rate is not None:
            assert (
                len(learning_rate) == 2 and 0 < learning_rate[0] < learning_rate[1]
            ), f"Invalid learning rate range {learning_rate}"
            self.learning_rate = neps.Float(
                lower=learning_rate[0], upper=learning_rate[1], log=True
            )

        if weight_decay is not None:
            weight_decay = (max(weight_decay[0], 1e-8), weight_decay[1])
            assert (
                1e-8 <= weight_decay[0] < weight_decay[1]
            ), f"Invalid weight decay range {weight_decay}"
            self.weight_decay = neps.Float(
                lower=weight_decay[0], upper=weight_decay[1], log=True
            )
        # Variables available (no 'u' since it's always the target of line 3)
        self._variables = neps.Categorical(choices=("w", "g", "v1", "v2"))

        # Constants
        self._constants = neps.Categorical(choices=(10, 1, 0, 0.1, 0.01, 0.9, 0.99))

        # Unary operations
        self._unary_funct = neps.Categorical(
            choices=(
                neps.Operation(
                    scale_by_constant,
                    kwargs={"constant": self._constants.resample()},
                ).resample(),
                neps.Operation(
                    clamp_by_constant,
                    kwargs={"constant": self._constants.resample()},
                ).resample(),
                torch.reciprocal,
                torch.square,
                torch.exp,
                torch.sqrt,
                torch.log,
                torch.neg,
            )
        )

        # Binary operations
        self._binary_funct = neps.Categorical(
            choices=(
                torch.add,
                torch.mul,
                neps.Operation(
                    interpolate,
                    kwargs={"constant": self._constants.resample()},
                ).resample(),
            )
        )

        # Line can be: variable, unary(variable), or binary(variable, variable)
        self._line = neps.Categorical(
            choices=(
                self._variables.resample(),
                neps.Operation(
                    operator=self.unaryFunction,
                    args=(
                        self._unary_funct.resample(),
                        self._variables.resample(),
                    ),
                ).resample(),
                neps.Operation(
                    operator=self.binaryFunction,
                    args=(
                        self._binary_funct.resample(),
                        self._variables.resample(),
                        self._variables.resample(),
                    ),
                ).resample(),
            ),
        )

        # Create optimizer with exactly 3 lines
        self.optimizer_cls = neps.Operation(
            operator=self.create_optimizer,
            args=(
                self._line.resample(),
                self._line.resample(),
                self._line.resample(),
            ),
        )

        # Learning rate
        self.learning_rate = neps.Float(lower=1e-5, upper=1e-2, log=True)

    @staticmethod
    def unaryFunction(operation: Callable, input_value):
        """Package unary operation."""
        return operation, input_value

    @staticmethod
    def binaryFunction(operation: Callable, input1, input2):
        """Package binary operation."""
        return operation, input1, input2

    def create_optimizer(self, *lines):
        """Create optimizer with exactly 3 lines (v1, v2, u)."""

        class ThreeLineOptimizer(torch.optim.Optimizer):
            """Optimizer with fixed 3-line structure."""

            def __init__(
                self, params, lr=1e-3, variables=(0.1, 0.1), weight_decay=0.0, **kwargs
            ):
                defaults = dict(lr=lr, vars=variables, weight_decay=weight_decay)
                super().__init__(params, defaults)
                self.lr = lr
                self.weight_decay = weight_decay

                # Initialize state
                for group in self.param_groups:
                    for p in group.get("params", []):
                        state = self.state.setdefault(p, {})
                        if "v1" not in state:
                            state["v1"] = torch.ones_like(p.data) * variables[0]
                        if "v2" not in state:
                            state["v2"] = torch.ones_like(p.data) * variables[1]
                        if "u" not in state:
                            state["u"] = torch.zeros_like(p.data)

            def step(self, closure=None):
                """Perform optimization step."""
                loss = None
                if closure is not None:
                    with torch.enable_grad():
                        loss = closure()

                for group in self.param_groups:
                    for p in group.get("params", []):
                        if p.grad is None:
                            d_p = torch.zeros_like(p.data)
                        else:
                            d_p = p.grad 

                        state = self.state.setdefault(p, {})

                        def as_tensor(x):
                            if isinstance(x, torch.Tensor):
                                try:
                                    return x.to(device=p.data.device, dtype=p.data.dtype)
                                except Exception:
                                    return x
                            else:
                                return torch.tensor(
                                    x, dtype=p.data.dtype, device=p.data.device
                                )

                        # Variable dictionary
                        var_dict = {
                            "g": d_p,
                            "w": p.data,
                            "u": torch.zeros_like(p.data),
                            "v1": state.get("v1", torch.zeros_like(p.data)),
                            "v2": state.get("v2", torch.zeros_like(p.data)),
                        }
                        var_dict.update(_runtime_symbol_dict(self, group, p.data))

                        # Apply weight decay
                        p.data -= self.weight_decay * p.data

                        # Evaluate 3 lines: v1, v2, u
                        for n, line in enumerate(lines):
                            expr = line
                            target_var = ["v1", "v2", "u"][n]
                            result = resolve_expression(expr, var_dict)
                            result = as_tensor(result)
                            state[target_var] = result
                            var_dict[target_var] = result

                        # Apply update
                        p.data = (
                            p.data - self.lr * state["u"]
                        )

                return loss

            def __repr__(self) -> str:
                string = f"{self.__class__.__name__}(\n"
                for group in self.param_groups:
                    string += f"  Parameter group:\n"
                    for k, v in group.items():
                        if k != "params":
                            string += f"    {k}: {v}\n"
                string += ")\nLines:\n"
                for n, line in enumerate(lines):
                    target_var = ["v1", "v2", "u"][n]
                    expression = line
                    if isinstance(expression, tuple):
                        string += f"  {target_var} = "
                        expr = expression
                        if callable(expr[0]):
                            if isinstance(expr[0], partial):
                                string += f"{expr[0].func.__name__}("
                                if expr[0].keywords:
                                    string += "{"
                                    for k, v in expr[0].keywords.items():
                                        string += f"{k} = {v}, "
                                    string = string.rstrip(", ")
                                    string += "}, "
                            else:
                                string += f"{expr[0].__name__}("
                            for arg in expr[1:]:
                                string += f"{arg}, "
                        else:
                            string += f"{expr}"
                        string = string.rstrip(", ")
                        string += ")\n"
                    else:
                        string += f"  {target_var} = {expression}\n"
                return string

            def get_lines(self):
                """Get optimizer lines."""
                return lines

        return ThreeLineOptimizer

import neps
import torch
from typing import Callable, Literal, Tuple
from functools import partial



def scale_by_constant(constant):
    """Scale tensor by a constant value."""
    return partial(torch.mul, other=constant)


def clamp_by_constant(constant):
    """Clamp tensor values within [-constant, constant]."""
    return partial(torch.clamp, min=-constant, max=constant)


def interpolate(constant):
    """Interpolate between tensors with given weight."""
    return partial(torch.lerp, weight=constant)


def resolve_expression(expr, var_dict):
    """
    Recursively evaluate nested tuples representing operations.

    Args:
        expr: Expression to resolve (string variable name or tuple operation)
        var_dict: Dictionary mapping variable names to values

    Returns:
        Resolved value
    """
    if isinstance(expr, str):
        return var_dict.get(expr, expr)

    op, *args = expr
    # Resolve nested args first (bottom-up)
    args = [resolve_expression(a, var_dict) for a in args]

    for n, arg in enumerate(args):
        if isinstance(arg, str):
            args[n] = var_dict[arg]

    return op(*args)



class SmallAdamMul(neps.PipelineSpace):
    """
    A small search space for Adam-like optimizers with multiplicative term.

    This space allows for flexible optimizer definitions with up to max_lines
    of update rules, where each line can assign to v1, and the last line always updates u.
    """

    def __init__(
        self,
        term_mode: Literal["add", "mul"] = "mul",
        n_lines: Tuple[int, int] = (1, 1),
        fidelity: Tuple[int, int] | None = None,
        learning_rate: Tuple[float, float] | None = None,
        weight_decay: Tuple[float, float] | None = None,
        **_,
    ):
        """
        Initialize the NOS space.

        Args:
            n_lines: Tuple indicating (min_lines, max_lines)
            fidelity: Optional fidelity range (lower, upper)
            learning_rate: Optional learning rate range (lower, upper)
            weight_decay: Optional weight decay range (lower, upper)
        """
        assert (
            n_lines[0] >= 0 and n_lines[1] >= n_lines[0]
        ), f"Invalid n_lines range {n_lines}"

        if fidelity is not None:
            assert (
                len(fidelity) == 2 and 0 <= fidelity[0] < fidelity[1]
            ), f"Invalid fidelity range {fidelity}"
            self.fidelity = neps.IntegerFidelity(lower=fidelity[0], upper=fidelity[1])

        if learning_rate is not None:
            assert (
                len(learning_rate) == 2 and 0 < learning_rate[0] < learning_rate[1]
            ), f"Invalid learning rate range {learning_rate}"
            self.learning_rate = neps.Float(
                lower=learning_rate[0], upper=learning_rate[1], log=True
            )

        if weight_decay is not None:
            weight_decay = (max(weight_decay[0], 1e-8), weight_decay[1])
            assert (
                1e-8 <= weight_decay[0] < weight_decay[1]
            ), f"Invalid weight decay range {weight_decay}"
            self.weight_decay = neps.Float(
                lower=weight_decay[0], upper=weight_decay[1], log=True
            )

        # Define variables that can be used in optimizer
        self._input_variables = neps.Categorical(choices=("w", "g", "v1"))
        self._output_variables = neps.Categorical(choices=("v1",))

        # Define constants that can be used in operations
        self._constants = neps.Categorical(choices=(0.1, 0.9))

        # Define unary operations
        self._unary_funct = neps.Categorical(
            choices=(
                torch.exp,
                torch.sign,
            )
        )

        # Define binary operations
        self._binary_funct = neps.Categorical(
            choices=(
                torch.mul,
                torch.sub,
                # neps.Operation(
                #     interpolate,
                #     kwargs={"constant": self._constants.resample()},
                # ).resample(),
            )
        )

        self._unary_right_hand = neps.Operation(
            operator=self.unaryFunction,
            args=(
                self._unary_funct.resample(),
                self._input_variables.resample(),
            ),
        )

        self._binary_right_hand = neps.Operation(
            operator=self.binaryFunction,
            args=(
                self._binary_funct.resample(),
                self._input_variables.resample(),
                self._input_variables.resample(),
            ),
        )

        # Define possible line structures
        self._line_right_hand = neps.Categorical(
            choices=(
                self._unary_right_hand.resample(),
                self._binary_right_hand.resample(),
            )
        )

        self._u_line_right_hand = neps.Categorical(
            choices=(
                # self._output_variables.resample(),
                self._unary_right_hand.resample(),
                self._binary_right_hand.resample(),
            )
        )

        # Create shared line pool
        self._shared_lines = [
            (self._output_variables.resample(), self._line_right_hand.resample())
            for _ in range(n_lines[1])
        ]

        # Create line choices (0 to max_lines)
        self._line_choices = tuple(
            tuple(self._shared_lines[:i]) for i in range(n_lines[0], n_lines[1] + 1)
        )

        # Define the optimizer class creation operation
        self.optimizer_cls = neps.Operation(
            operator=self.create_optimizer,
            args=(neps.Categorical(choices=self._line_choices)),
            kwargs={"last_line": ("u", self._u_line_right_hand),
                    "term_mode": term_mode},
        )

    @staticmethod
    def unaryFunction(operation: Callable, input_value):
        """Package unary operation with its input."""
        return operation, input_value

    @staticmethod
    def binaryFunction(operation: Callable, input1, input2):
        """Package binary operation with its inputs."""
        return operation, input1, input2

    @staticmethod
    def create_optimizer(*lines, last_line, term_mode: Literal["add", "mul"] = "add"):
        """
        Create a custom optimizer class from the given lines.

        Args:
            *lines: Variable number of lines
            last_line: The last line which always updates 'u'

        Returns:
            Custom optimizer class
        """

        class CustomOptimizer(torch.optim.Optimizer):
            """Custom optimizer with flexible update rules."""

            def __init__(
                self, params, lr=0.001, betas=(0.9, 0.95), eps=1e-08, weight_decay=0.01, variables=(0.1, 0.1), **kwargs
            ):
                defaults = dict(lr=lr, vars=variables, weight_decay=weight_decay)
                super().__init__(params, defaults)
                self.lr = lr
                self.weight_decay = weight_decay
                self.beta1 = betas[0]
                self.beta2 = betas[1]
                self.eps = eps

                # Initialize state for each parameter
                for group in self.param_groups:
                    for p in group.get("params", []):
                        state = self.state.setdefault(p, {})
                        if "v1" not in state:
                            state["v1"] = torch.ones_like(p.data) * variables[0]
                        if "v2" not in state:
                            state["v2"] = torch.ones_like(p.data) * variables[1]
                        if "u" not in state:
                            state["u"] = torch.zeros_like(p.data)
                        if "m" not in state:
                            state["m"] = torch.zeros_like(p.data)
                        if "v" not in state:
                            state["v"] = torch.zeros_like(p.data)
                        

            def step(self, closure=None):
                """Perform a single optimization step."""
                loss = None
                if closure is not None:
                    with torch.enable_grad():
                        loss = closure()

                for group in self.param_groups:
                    for p in group.get("params", []):
                        # Get gradient (or zero if missing)
                        if p.grad is None:
                            d_p = torch.zeros_like(p.data)
                        else:
                            d_p = p.grad

                        state = self.state.setdefault(p, {})

                        def as_tensor(x):
                            """Convert to tensor matching parameter."""
                            if isinstance(x, torch.Tensor):
                                try:
                                    return x.to(device=p.data.device, dtype=p.data.dtype)
                                except Exception:
                                    return x
                            else:
                                return torch.tensor(
                                    x, dtype=p.data.dtype, device=p.data.device
                                )

                        # Build variable dictionary
                        var_dict = {
                            "g": d_p,
                            "w": p.data,
                            "u": state.get("u", torch.zeros_like(p.data)),
                            "v1": state.get("v1", torch.zeros_like(p.data)),
                            "v2": state.get("v2", torch.zeros_like(p.data)),
                        }
                        var_dict.update(_runtime_symbol_dict(self, group, p.data))

                        # Apply weight decay
                        p.data -= self.weight_decay * p.data

                        # First perform an AdamW like update
                        state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * d_p
                        state["v"] = self.beta2 * state["v"] + (1 - self.beta2) * (d_p * d_p)
                        m_hat = state["m"] / (1 - self.beta1)
                        v_hat = state["v"] / (1 - self.beta2)

                        # Evaluate each line + last line and update state
                        for line in lines + (last_line,):
                            target_var, expr = line
                            result = resolve_expression(expr, var_dict)
                            result = as_tensor(result)
                            state[target_var] = result
                            var_dict[target_var] = result

                        # Apply update to parameter
                        p.data = (
                            (p.data - self.lr * (m_hat / (torch.sqrt(v_hat) + self.eps)) * state["u"]) if term_mode=="mul" 
                            else (p.data - self.lr * (m_hat / (torch.sqrt(v_hat) + self.eps) + state["u"]))
                        )

                return loss

            def __repr__(self) -> str:
                """String representation of the optimizer."""
                string = f"{self.__class__.__name__}(\n"
                for group in self.param_groups:
                    string += f"  Parameter group:\n"
                    for k, v in group.items():
                        if k != "params":
                            string += f"    {k}: {v}\n"
                string += ")\nLines:\n"
                string += "(fixed) m = beta1 * m + (1 - beta1) * g\n"
                string += "(fixed) v = beta2 * v + (1 - beta2) * (g * g)\n"
                string += "(fixed) m_hat = m / (1 - beta1)\n"
                string += "(fixed) v_hat = v / (1 - beta2)\n"
                for line in lines + (last_line,):
                    target_var = line[0]
                    expression = line[1]
                    if isinstance(expression, tuple):
                        string += f"  {target_var:>2} = "
                        expr = expression
                        if callable(expr[0]):
                            if isinstance(expr[0], partial):
                                string += f"{expr[0].func.__name__}("
                                if expr[0].keywords:
                                    string += "{"
                                    for k, v in expr[0].keywords.items():
                                        string += f"{k}={v}, "
                                    string = string.rstrip(", ")
                                    string += "}, "
                            else:
                                string += f"{expr[0].__name__}("
                            for arg in expr[1:]:
                                string += f"{arg}, "
                        else:
                            string += f"{expr[0]}"
                        string = string.rstrip(", ")
                        string += ")\n"
                    else:
                        string += f"  {target_var:>2} = {expression}\n"
                string += f"(fixed) w = w - lr * (m_hat / (sqrt(v_hat) + eps){') * u' if term_mode=='mul' else ' + u)'}\n"
                return string

            def get_lines(self):
                """Get the sampled optimizer update lines."""
                return lines + (last_line,)

        return CustomOptimizer

