import neps
from neps_nos import neps_nos_space
from typing import Tuple, Any
from neps.space.neps_spaces.neps_space import (
    NepsCompatConverter,
    construct_sampling_path,
    convert_operation_to_callable,
    resolve,
)
from neps.space.neps_spaces import string_formatter
from neps.space.neps_spaces.sampling import OnlyPredefinedValuesSampler
from functools import partial

OPTIMIZERS = {
    "RE": ("neps_regularized_evolution", {"ignore_fidelity": "highest_fidelity"}),
    "RS": ("neps_random_search", {"ignore_fidelity": "highest_fidelity"}),
    "HB": ("neps_hyperband",{}),
    'LI3_r0.1': ("neps_local_and_incumbent", {
        'base': 'hyperband',
        'eta': 2,
        'inc_takeover_mode': 3,
        'random_ratio': 0.1
    }),
    'LI3_r0.3': ("neps_local_and_incumbent", {
        'base': 'hyperband',
        'eta': 2,
        'inc_takeover_mode': 3,
        'random_ratio': 0.3
    }),
    'LI0_r0.3': ("neps_local_and_incumbent", {
        'base': 'hyperband',
        'eta': 2,
        'inc_takeover_mode': 0,
        'random_ratio': 0.3
    }),
    'LI1_r0.3': ("neps_local_and_incumbent", {
        'base': 'hyperband',
        'eta': 2,
        'inc_takeover_mode': 1,
        'random_ratio': 0.3
    }),
    'LI1_r0.1': ("neps_local_and_incumbent", {
        'base': 'hyperband',
        'eta': 2,
        'inc_takeover_mode': 1,
        'random_ratio': 0.1
    }),
    'LI1_r0.5': ("neps_local_and_incumbent", {
        'base': 'hyperband',
        'eta': 2,
        'inc_takeover_mode': 1,
        'random_ratio': 0.5
    }),
    'LI2_r0.3': ("neps_local_and_incumbent", {
        'base': 'hyperband',
        'eta': 2,
        'inc_takeover_mode': 2,
        'random_ratio': 0.3
    }),
    'LI3_r0.5': ("neps_local_and_incumbent", {
        'base': 'hyperband',
        'eta': 2,
        'inc_takeover_mode': 3,
        'random_ratio': 0.5
    }),
    'PB-like': ("neps_priorband", {
        'base': 'hyperband',
        'eta': 2,
    }),
    'PB-like': ("neps_priorband", {
        'base': 'hyperband',
        'eta': 2,
    }),
    'GridSearch': ("neps_exhaustive_search", {"ignore_fidelity": "highest_fidelity", "sampling_density": 3}),
    'LI1_rand0.1': ("neps_local_and_incumbent", {
        'base': 'hyperband',
        'inc_takeover_mode': 1,
        'mutation_mode': ['random', 0.1],
    }),
    'LI1_rand0.3': ("neps_local_and_incumbent", {
        'base': 'hyperband',
        'inc_takeover_mode': 1,
        'mutation_mode': ['random', 0.3],
    }),
    'LI1_rand0.5': ("neps_local_and_incumbent", {
        'base': 'hyperband',
        'inc_takeover_mode': 1,
        'mutation_mode': ['random', 0.5],
    }),
    'LI1_rand0.8': ("neps_local_and_incumbent", {
        'base': 'hyperband',
        'inc_takeover_mode': 1,
        'mutation_mode': ['random', 0.8],
    }),
    'LI1_ratio0.1': ("neps_local_and_incumbent", {
        'base': 'hyperband',
        'inc_takeover_mode': 1,
        'mutation_mode': ['ratio', 0.1],
    }),
    'LI1_ratio0.3': ("neps_local_and_incumbent", {
        'base': 'hyperband',
        'inc_takeover_mode': 1,
        'mutation_mode': ['ratio', 0.3],
    }),
    'LI1_ratio0.5': ("neps_local_and_incumbent", {
        'base': 'hyperband',
        'inc_takeover_mode': 1,
        'mutation_mode': ['ratio', 0.5],
    }),
    'LI1_ratio0.8': ("neps_local_and_incumbent", {
        'base': 'hyperband',
        'inc_takeover_mode': 1,
        'mutation_mode': ['ratio', 0.8],
    }),
    'LI1_fixed1': ("neps_local_and_incumbent", {
        'base': 'hyperband',
        'inc_takeover_mode': 1,
        'mutation_mode': ['fixed', 1],
    }),
    'LI1_fixed3': ("neps_local_and_incumbent", {
        'base': 'hyperband',
        'inc_takeover_mode': 1,
        'mutation_mode': ['fixed', 3],
    }),
    'LI1_fixed5': ("neps_local_and_incumbent", {
        'base': 'hyperband',
        'inc_takeover_mode': 1,
        'mutation_mode': ['fixed', 5],
    }),
    'LI1_fixed8': ("neps_local_and_incumbent", {
        'base': 'hyperband',
        'inc_takeover_mode': 1,
        'mutation_mode': ['fixed', 8],
    }),
    'LI1_rand0.1r1': ("neps_local_and_incumbent", {
        'base': 'hyperband',
        'inc_takeover_mode': 1,
        'mutation_mode': ['random', 0.1],
        'random_ratio': 0.1
    }),
    'LI1_rand0.1r3': ("neps_local_and_incumbent", {
        'base': 'hyperband',
        'inc_takeover_mode': 1,
        'mutation_mode': ['random', 0.1],
        'random_ratio': 0.3
    }),
    'PB_like': ("neps_priorband", {
        'base': 'hyperband',
    }),
    'PB_95': ("neps_priorband", {
        'base': 'hyperband',
        'inc_ratio': 0.95,
    }),
    'PB_80': ("neps_priorband", {
        'base': 'hyperband',
        'inc_ratio': 0.80,
    }),
    'PB_50': ("neps_priorband", {
        'base': 'hyperband',
        'inc_ratio': 0.50,
    }),
}


SPACES = {
    "NLinesU_nf_nl_nw": 
        ("NLinesU_1_10",
        {}),
    "NLinesU_nf_l_nw": 
        ("NLinesU_1_10",
        {"learning_rate": (1e-5, 1e-1)}),
    "NLinesU_f_l_nw": 
        ("NLinesU_1_10",
        {"learning_rate": (1e-5, 1e-1),
        "fidelity": True}),
    "NLinesU_f_nl_nw": 
        ("NLinesU_1_10",
        {"fidelity": True}),
    "AdamExtend_nf_l_nw": 
        ("AdamExtend_1_6",
        {"learning_rate": (1e-5, 1e-1)}),
    "AdamExtend_f_l_nw": 
        ("AdamExtend_1_6",
        {"learning_rate": (1e-5, 1e-1),
        "fidelity": True}),
    "AdamExtend_f_l_w":
        ("AdamExtend_1_6",
        {"learning_rate": (1e-5, 1e-1),
        "weight_decay": (1e-6, 2e-1),
        "fidelity": True}),
    "AdamExtendMul_f_l_nw": 
        ("AdamExtend_1_6_mul",
        {"learning_rate": (1e-5, 1e-1),
        "fidelity": True}),
    "PremadeModules_nf_l_nw":
        ("PremadeModules",
        {"learning_rate": (1e-5, 1e-1)}),
    "PremadeModules_f_l_nw":
        ("PremadeModules",
        {"learning_rate": (1e-5, 1e-1),
        "fidelity": True}),
    "SmallAdam_f_nl_nw":
        ("SmallAdam",
        {"fidelity": True}),
    "AdamMore":
        ("AdamWMore",
        {"learning_rate": (1e-5, 1e-1),
        "weight_decay": (1e-6, 2e-1),
        "fidelity": True}),
    "AdamMore1":
        ("AdamWMore1",
        {"learning_rate": (1e-5, 1e-1),
        "weight_decay": (1e-6, 2e-1),
        "fidelity": True}),
    "AdamMore2":
        ("AdamWMore2",
        {"learning_rate": (1e-5, 1e-1),
        "weight_decay": (1e-6, 2e-1),
        "fidelity": True}),
    "AdamMore3":
        ("AdamWMore3",
        {"learning_rate": (1e-5, 1e-1),
        "weight_decay": (1e-6, 2e-1),
        "fidelity": True}),
}

SPACE_BASES = {
    "NLinesU_1_10": partial(neps_nos_space.NOSSpaceNLinesU, n_lines=(1,10)),
    "AdamExtend_1_6": partial(neps_nos_space.AdamWExtend, n_lines=(1,6)),
    "AdamExtend_1_6_mul": partial(neps_nos_space.AdamWExtend, n_lines=(1,6), term_mode="mul"),
    "AdamWMore": partial(neps_nos_space.AdamWMore, n_lines=(1,6)),
    "PremadeModules": neps_nos_space.PremadeModules,
    "SmallAdam": neps_nos_space.SmallAdamMul,
    "AdamWMore1": partial(neps_nos_space.AdamWMore, special_variables=("t",), n_lines=(1,6)),
    "AdamWMore2": partial(neps_nos_space.AdamWMore, special_variables=("depth",), n_lines=(1,6)),
    "AdamWMore3": partial(neps_nos_space.AdamWMore, special_variables=("layer_type_attention",), n_lines=(1,6)),

}

WARMSTART_PARAMETERS_DEFAULTS = {
    "fidelity": "max",  # "min" for min_fidelity (Warning: Could be smaller then the smallest rung), "max" for max_fidelity, or a specific number
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
}

WARMSTART_PARAMETERS = {
    "SGDM_inter":
    ("SGDM_inter",
    WARMSTART_PARAMETERS_DEFAULTS),
    "SGDM_add":
    ("SGDM_add",
    WARMSTART_PARAMETERS_DEFAULTS),
    "Adam":
    ("Adam",
    WARMSTART_PARAMETERS_DEFAULTS),
}

def get_space_basename_and_kwargs(space_config_name: str) -> Tuple[str, dict[str, Any]]:
    if space_config_name not in SPACES:
        raise ValueError(f"Unknown space config name: {space_config_name}\nAvailable configs: {list(SPACES.keys())}")
    return SPACES[space_config_name]

def get_optimizer_name_and_kwargs(optimizer_name: str) -> Tuple[str, dict[str, Any]]:
    if optimizer_name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer name: {optimizer_name}\nAvailable optimizers: {list(OPTIMIZERS.keys())}")
    return OPTIMIZERS[optimizer_name]

def get_space_base_callable(space_base_name: str):
    if space_base_name not in SPACE_BASES:
        raise ValueError(f"Unknown space base name: {space_base_name}\nAvailable space bases: {list(SPACE_BASES.keys())}")
    return SPACE_BASES[space_base_name]

def resolve_warmstarter_name(warmstarter_str: str) -> Tuple[str, dict[str, Any]]:
    if warmstarter_str not in WARMSTART_PARAMETERS:
        raise ValueError(f"Unknown warmstarter name: {warmstarter_str}\nAvailable warmstarters: {list(WARMSTART_PARAMETERS.keys())}")
    return WARMSTART_PARAMETERS[warmstarter_str]


def get_warmstarter_config(space_base_name: str, space_args: dict[str, Any], pipeline_space: neps.PipelineSpace, warmstarter_name: str, warmstarter_kwargs: dict[str, Any]) -> Tuple[dict[str, Any], dict[str, Any]]:
    if space_base_name not in WARMSTART_CONFIGS:
        raise ValueError(f"Unknown space base name for warmstarter: {space_base_name}")
    warmstarter_configs = WARMSTART_CONFIGS[space_base_name]
    if warmstarter_name not in warmstarter_configs:
        raise ValueError(f"Unknown warmstarter name: {warmstarter_name}")
    
    warmstart_config = warmstarter_configs[warmstarter_name]
    extended_config = warmstart_config.copy()

    for param in space_args.keys():
        if param == "fidelity":
            assert (
                getattr(pipeline_space, param) is not None
            ), "Space has no fidelity parameter to set"
            extended_config[NepsCompatConverter._ENVIRONMENT_PREFIX + param] = (
                warmstarter_kwargs.get(param, WARMSTART_PARAMETERS_DEFAULTS[param])
            )
        else:
            sampling_path = construct_sampling_path(
                path_parts=["Resolvable", param], domain_obj=pipeline_space.__getattribute__(param)
            )
            extended_config[NepsCompatConverter._SAMPLING_PREFIX + sampling_path] = (
                warmstarter_kwargs.get(param, WARMSTART_PARAMETERS_DEFAULTS[param])
            )
    
    converted_dict = NepsCompatConverter.from_neps_config(extended_config)

    pipeline, _ = resolve(
        pipeline_space,
        domain_sampler=OnlyPredefinedValuesSampler(converted_dict.predefined_samplings),
        environment_values=converted_dict.environment_values,
    )

    pipeline_dict = dict(**pipeline.get_attrs())

    for parameter, value in pipeline_dict.items():
        if isinstance(value, neps.Operation):
            # If the operator is a not a string, we convert it to a callable.
            if isinstance(value.operator, str):
                pipeline_dict[parameter] = string_formatter.format_value(value)
            else:
                pipeline_dict[parameter] = convert_operation_to_callable(value)

    return extended_config, pipeline_dict


WARMSTART_CONFIGS = {
    "NLinesU_1_10": {
        "SGDM_inter": {
            "SAMPLING__Resolvable.optimizer_cls.args::categorical__10": 0,
            "SAMPLING__Resolvable.optimizer_cls.args.sampled_value.sequence[0].sequence[0].resampled_categorical::categorical__2": 0,
            "SAMPLING__Resolvable.optimizer_cls.args.sampled_value.sequence[0].sequence[1].resampled_categorical::categorical__2": 1,
            "SAMPLING__Resolvable.optimizer_cls.args.sampled_value.sequence[0].sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_categorical::categorical__3": 2,
            "SAMPLING__Resolvable.optimizer_cls.args.sampled_value.sequence[0].sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.kwargs.mapping_value{constant}.resampled_categorical::categorical__7": 3,
            "SAMPLING__Resolvable.optimizer_cls.args.sampled_value.sequence[0].sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[1].resampled_categorical::categorical__4": 1,
            "SAMPLING__Resolvable.optimizer_cls.args.sampled_value.sequence[0].sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[2].resampled_categorical::categorical__4": 2,
            "SAMPLING__Resolvable.optimizer_cls.kwargs.mapping_value{last_line}.sequence[1]::categorical__3": 0,
            "SAMPLING__Resolvable.optimizer_cls.kwargs.mapping_value{last_line}.sequence[1].sampled_value.resampled_categorical::categorical__2": 0,
        },
    "SGDM_add":{
        'SAMPLING__Resolvable.optimizer_cls.args::categorical__10': 2, 
        'SAMPLING__Resolvable.optimizer_cls.args.sampled_value.sequence[0].sequence[0].resampled_categorical::categorical__2': 0,
        'SAMPLING__Resolvable.optimizer_cls.args.sampled_value.sequence[0].sequence[1].resampled_categorical::categorical__2': 0, 'SAMPLING__Resolvable.optimizer_cls.args.sampled_value.sequence[0].sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_categorical::categorical__8': 0,
        'SAMPLING__Resolvable.optimizer_cls.args.sampled_value.sequence[0].sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.kwargs.mapping_value{constant}.resampled_categorical::categorical__7': 5,
        'SAMPLING__Resolvable.optimizer_cls.args.sampled_value.sequence[0].sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[1].resampled_categorical::categorical__4': 2,
        'SAMPLING__Resolvable.optimizer_cls.args.sampled_value.sequence[1].sequence[0].resampled_categorical::categorical__2': 1,
        'SAMPLING__Resolvable.optimizer_cls.args.sampled_value.sequence[1].sequence[1].resampled_categorical::categorical__2': 0,
        'SAMPLING__Resolvable.optimizer_cls.args.sampled_value.sequence[1].sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_categorical::categorical__8': 0,
        'SAMPLING__Resolvable.optimizer_cls.args.sampled_value.sequence[1].sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.kwargs.mapping_value{constant}.resampled_categorical::categorical__7': 3,
        'SAMPLING__Resolvable.optimizer_cls.args.sampled_value.sequence[1].sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[1].resampled_categorical::categorical__4': 1,
        'SAMPLING__Resolvable.optimizer_cls.args.sampled_value.sequence[2].sequence[0].resampled_categorical::categorical__2': 0,
        'SAMPLING__Resolvable.optimizer_cls.args.sampled_value.sequence[2].sequence[1].resampled_categorical::categorical__2': 1,
        'SAMPLING__Resolvable.optimizer_cls.args.sampled_value.sequence[2].sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_categorical::categorical__3': 0,
        'SAMPLING__Resolvable.optimizer_cls.args.sampled_value.sequence[2].sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[1].resampled_categorical::categorical__4': 2,
        'SAMPLING__Resolvable.optimizer_cls.args.sampled_value.sequence[2].sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[2].resampled_categorical::categorical__4': 3,
        'SAMPLING__Resolvable.optimizer_cls.kwargs.mapping_value{last_line}.sequence[1]::categorical__3': 0,
        'SAMPLING__Resolvable.optimizer_cls.kwargs.mapping_value{last_line}.sequence[1].sampled_value.resampled_categorical::categorical__2': 0,
        },
    },
    "AdamExtend_1_6": {
        "Adam":{
            'SAMPLING__Resolvable.optimizer_cls.args.sampled_value.sequence[0].sequence[0].resampled_categorical::categorical__2': 0,
            'SAMPLING__Resolvable.optimizer_cls.args.sampled_value.sequence[0].sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_categorical.sampled_value.resampled_operation.kwargs.mapping_value{constant}.resampled_categorical::categorical__7': 2,
            'SAMPLING__Resolvable.optimizer_cls.args.sampled_value.sequence[0].sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[0].resampled_categorical::categorical__8': 0,
            'SAMPLING__Resolvable.optimizer_cls.args.sampled_value.sequence[0].sequence[1].resampled_categorical.sampled_value.resampled_operation.args.sequence[1].resampled_categorical::categorical__4': 2,
            'SAMPLING__Resolvable.optimizer_cls.args.sampled_value.sequence[0].sequence[1].resampled_categorical::categorical__2': 0,
            'SAMPLING__Resolvable.optimizer_cls.args::categorical__6': 0,
            'SAMPLING__Resolvable.optimizer_cls.kwargs.mapping_value{last_line}.sequence[1].sampled_value.resampled_categorical::categorical__2': 0,
            'SAMPLING__Resolvable.optimizer_cls.kwargs.mapping_value{last_line}.sequence[1]::categorical__3': 0}
    }
}