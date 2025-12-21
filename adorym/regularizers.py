import adorym.wrappers as w
from adorym.util import *


class Regularizer(object):
    """Parent regularizer class.

    :param unknown_type: String. Can be ``'delta_beta'`` or ``'real_imag'``.
    :param device: Device object or ``None``.
    """
    def __init__(self, unknown_type='delta_beta', name=None):
        self.unknown_type = unknown_type
        self.name = name or type(self).__name__

    def get_value(self, obj, device=None, **kwargs):
        pass


class L1Regularizer(Regularizer):
    """L1-norm regularizer.

    :param alpha_d: Weight of l1-norm of delta or real part.
    :param alpha_b: Weight of l1-norm of beta or imaginary part.
    """

    def __init__(self, alpha_d, alpha_b, unknown_type='delta_beta'):
        super().__init__(unknown_type)
        self.alpha_d = alpha_d
        self.alpha_b = alpha_b

    def get_value(self, obj, device=None, **kwargs):
        slicer = [slice(None)] * (len(obj.shape) - 1)
        reg = w.create_variable(0., device=device)
        if self.unknown_type == 'delta_beta':
            if self.alpha_d not in [None, 0]:
                reg = reg + self.alpha_d * w.mean(w.abs(obj[slicer + [0]]))
            if self.alpha_b not in [None, 0]:
                reg = reg + self.alpha_b * w.mean(w.abs(obj[slicer + [1]]))
        elif self.unknown_type == 'real_imag':
            r = obj[slicer + [0]]
            i = obj[slicer + [1]]
            if self.alpha_d not in [None, 0]:
                om = w.sqrt(r ** 2 + i ** 2)
                reg = reg + self.alpha_d * w.mean(w.abs(om - w.mean(om)))
            if self.alpha_b not in [None, 0]:
                reg = reg + self.alpha_b * w.mean(w.abs(w.arctan2(i, r)))
        return reg


class ReweightedL1Regularizer(Regularizer):
    """Reweighted l1-norm regularizer.

    :param alpha_d: Weight of l1-norm of delta or real part.
    :param alpha_b: Weight of l1-norm of beta or imaginary part.
    """
    def __init__(self, alpha_d, alpha_b, unknown_type='delta_beta'):
        super().__init__(unknown_type)
        self.alpha_d = alpha_d
        self.alpha_b = alpha_b
        self.weight_l1 = None

    def update_l1_weight(self, weight_l1):
        self.weight_l1 = weight_l1

    def get_value(self, obj, device=None, **kwargs):
        slicer = [slice(None)] * (len(obj.shape) - 1)
        reg = w.create_variable(0., device=device)
        if self.unknown_type == 'delta_beta':
            if self.alpha_d not in [None, 0]:
                reg = reg + self.alpha_d * w.mean(self.weight_l1[slicer + [0]] * w.abs(obj[slicer + [0]]))
            if self.alpha_b not in [None, 0]:
                reg = reg + self.alpha_b * w.mean(self.weight_l1[slicer + [1]] * w.abs(obj[slicer + [1]]))
        elif self.unknown_type == 'real_imag':
            r = obj[slicer + [0]]
            i = obj[slicer + [1]]
            wr = self.weight_l1[slicer + [0]]
            wi = self.weight_l1[slicer + [1]]
            wm = wr ** 2 + wi ** 2
            if self.alpha_d not in [None, 0]:
                om = w.sqrt(r ** 2 + i ** 2)
                reg = reg + self.alpha_d * w.mean(wm * w.abs(om - w.mean(om)))
            if self.alpha_b not in [None, 0]:
                reg = reg + self.alpha_b * w.mean(wm * w.abs(w.arctan2(i, r)))
        return reg


class TVRegularizer(Regularizer):
    """Total variation regularizer.

    :param gamma: Weight of TV term.
    """
    def __init__(self, gamma, unknown_type='delta_beta'):
        super().__init__(unknown_type)
        self.gamma = gamma

    def get_value(self, obj, distribution_mode=None, device=None, **kwargs):
        slicer = [slice(None)] * (len(obj.shape) - 1)
        reg = w.create_variable(0., device=device)
        if self.unknown_type == 'delta_beta':
            o1 = obj[slicer + [0]]
            o2 = obj[slicer + [1]]
            axis_offset = 0 if distribution_mode is None else 1
            reg = reg + self.gamma * total_variation_3d(o1, axis_offset=axis_offset)
            reg = reg + self.gamma * total_variation_3d(o2, axis_offset=axis_offset)
        elif self.unknown_type == 'real_imag':
            r = obj[slicer + [0]]
            i = obj[slicer + [1]]
            axis_offset = 0 if distribution_mode is None else 1
            reg = reg + self.gamma * total_variation_3d(r ** 2 + i ** 2, axis_offset=axis_offset)
            reg = reg + self.gamma * total_variation_3d(w.arctan2(i, r), axis_offset=axis_offset)
        return reg


class BackgroundTVRegularizer(Regularizer):
    """Total variation regularizer for background map N.

    :param gamma: Weight of TV term.
    """

    def __init__(self, gamma):
        super().__init__(unknown_type=None)
        self.gamma = gamma

    def get_value(self, obj, noise=None, distribution_mode=None, device=None, **kwargs):
        reg = w.create_variable(0., device=device)
        if noise is None or self.gamma in [0, None]:
            return reg
        axis_offset = 0 if distribution_mode is None else 1
        reg = reg + self.gamma * total_variation_3d(noise, axis_offset=axis_offset)
        return reg


class BackgroundMeanPullRegularizer(Regularizer):
    """L2 pull of background map N towards provided mean value.

    :param weight: Weight of the L2 term.
    :param target_mean: Mean background value.
    """

    def __init__(self, weight, target_mean=None):
        super().__init__(unknown_type=None)
        self.weight = weight
        self.target_mean = target_mean

    def get_value(self, obj, noise=None, device=None, **kwargs):
        reg = w.create_variable(0., device=device)
        if noise is None or self.weight in [0, None] or self.target_mean is None:
            return reg
        target_mean = self.target_mean
        if not hasattr(target_mean, 'shape'):
            target_mean = w.create_variable(target_mean, device=device, requires_grad=False)
        reg = reg + self.weight * w.mean((noise - target_mean) ** 2)
        return reg


class ProbeSlaveRatioRegularizer(Regularizer):
    """Keep slave probe magnitude below a fraction of master probe magnitude.

    :param weight: Weight of the ratio penalty term.
    :param max_ratio: Maximum allowed ||P_s|| / ||P_m|| ratio before penalization.
    """

    def __init__(self, weight, max_ratio):
        super().__init__(unknown_type=None)
        self.weight = weight
        self.max_ratio = max_ratio

    def get_value(self, obj, probe_real=None, probe_imag=None, probe_slave_real=None,
                  probe_slave_imag=None, device=None, **kwargs):
        reg = w.create_variable(0., device=device)
        if self.weight in [0, None] or probe_slave_real is None or probe_slave_imag is None:
            return reg
        if probe_real is None or probe_imag is None:
            return reg
        eps = 1e-12
        master_norm = w.vec_norm(w.sqrt(probe_real ** 2 + probe_imag ** 2))
        slave_norm = w.vec_norm(w.sqrt(probe_slave_real ** 2 + probe_slave_imag ** 2))
        ratio = slave_norm / (master_norm + eps)
        reg = reg + self.weight * w.mean(w.clip(ratio - self.max_ratio, 0, np.inf) ** 2)
        return reg


class CorrRegularizer(Regularizer):
    """Pearson correlation regularizer along z axis of object.

    :param gamma: Weight of correlation term.
    """
    def __init__(self, gamma, unknown_type='delta_beta'):
        super().__init__(unknown_type)
        self.gamma = gamma

    def get_value(self, obj, distribution_mode=None, device=None, **kwargs):
        slicer = [slice(None)] * (len(obj.shape) - 1)
        reg = w.create_variable(0., device=device)
        if self.unknown_type == 'delta_beta':
            o1 = obj[slicer + [0]]
            o2 = obj[slicer + [1]]
        elif self.unknown_type == 'real_imag':
            r = obj[slicer + [0]]
            i = obj[slicer + [1]]
            o1 = w.sqrt(r ** 2 + i ** 2)
            o2 = w.arctan2(i, r)
        else:
            raise ValueError('Invalid value for unknown_type.')

        reg = reg + self.gamma * w.pcc(o1)
        reg = reg + self.gamma * w.pcc(o2)
        return reg


class GradCorrRegularizer(Regularizer):
    """Pearson correlation regularizer of the gradients maps of all slices along z axis of object.

    :param gamma: Weight of correlation term.
    """

    def __init__(self, gamma, unknown_type='delta_beta'):
        super().__init__(unknown_type)
        self.gamma = gamma

    def get_value(self, obj, distribution_mode=None, device=None, **kwargs):
        ndim = len(obj.shape)
        slicer = [slice(None)] * (len(obj.shape) - 1)
        reg = w.create_variable(0., device=device)
        if self.unknown_type == 'delta_beta':
            o1 = obj[slicer + [0]]
            o2 = obj[slicer + [1]]
        elif self.unknown_type == 'real_imag':
            r = obj[slicer + [0]]
            i = obj[slicer + [1]]
            o1 = w.sqrt(r ** 2 + i ** 2)
            o2 = w.arctan2(i, r)
        else:
            raise ValueError('Invalid value for unknown_type.')
        o1 = image_gradient(o1, axes=[ndim - 4, ndim - 3])
        o2 = image_gradient(o2, axes=[ndim - 4, ndim - 3])
        reg = reg + self.gamma * w.pcc(o1)
        reg = reg + self.gamma * w.pcc(o2)
        return reg
