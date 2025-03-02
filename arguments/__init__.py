
#
# Edited by: Jingwei Xu, ShanghaiTech University
# Based on the code from: https://github.com/graphdeco-inria/gaussian-splatting
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._colmap_path = ""
        self._model_path = ""
        self.start_frame = None
        self.end_frame = None
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.depth_ratio = 0.0
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 50_000
        self.position_lr_init = 0.000016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 50_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.001
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_dist = 100.0
        self.lambda_normal = 0.05
        self.opacity_cull = 0.005

        self.enable_semantic_loss = True
        self.semantic_loss_ratio = 0.1

        self.densification_interval = 500
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 25_000
        self.densify_grad_threshold = 0.0002

        self.semantic_dist_from_iter = 27_500
        self.normal_consist_from_iter = 30_000

        self.prune_from_iter = 31_000
        self.prune_until_iter = 45_000
        self.prune_interval = 4_000
        self.prune_opacity = 0.3

        self.shrinking_from_iter = 31_000
        self.lambda_shrink = 0.001

        if parser is not None:
            super().__init__(parser, "MaskOptimization Parameters")

class ReOptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 1000
        self.position_lr_init = 0.000016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 1000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_dist = 100.0
        self.lambda_normal = 0.05
        self.enable_geometry_loss = False
        self.geometric_loss_ratio = 0.5
        self.enable_depth_loss = False
        self.depth_loss_ratio = 0.025
        self.enable_semantic_loss = True
        self.semantic_loss_ratio = 0.02
        self.densification_interval = 200
        self.opacity_reset_interval = 400
        self.densify_from_iter = 200
        self.densify_until_iter = 1_500
        self.densify_grad_threshold = 0.0002
        if parser is not None:
            super().__init__(parser, "MaskOptimization Parameters")


def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
