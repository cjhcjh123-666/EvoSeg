from .dist import _init_dist_pytorch, get_dist_info, master_only, get_rank, collect_results_cpu, _init_dist_slurm, barrier


def __getattr__(name):
    if name == 'REFER':
        from .refcoco_refer import REFER
        return REFER
    if name == 'G_REFER':
        from .grefcoco import G_REFER
        return G_REFER
    if name in {'AverageMeter', 'Summary', 'intersectionAndUnionGPU'}:
        from . import utils_refcoco
        return getattr(utils_refcoco, name)
    raise AttributeError(name)
