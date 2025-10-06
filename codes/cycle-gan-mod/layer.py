import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
from nnabla.parameter import get_parameter_or_create
from nnabla.initializer import ConstantInitializer
from nnabla.parametric_functions import parametric_function_api


@parametric_function_api("in")
def instance_normalization(inp, axes=[1], decay_rate=0.9, eps=1e-5,
                           batch_stat=True, output_stat=False, fix_parameters=False):
    """Instance Normalization (implemented using BatchNormalization)

    Instance normalization is equivalent to the batch normalization if a batch size is one, in
    other words, it normalizes over spatial dimension(s), meaning all dimensions except for
    the batch and feature dimension.

    """
    assert len(axes) == 1
    shape_stat = [1 for _ in inp.shape]
    shape_stat[axes[0]] = inp.shape[axes[0]]
    beta = get_parameter_or_create(
        "beta", shape_stat, ConstantInitializer(0), not fix_parameters)
    gamma = get_parameter_or_create(
        "gamma", shape_stat, ConstantInitializer(1), not fix_parameters)
    mean = get_parameter_or_create(
        "mean", shape_stat, ConstantInitializer(0), False)
    var = get_parameter_or_create(
        "var", shape_stat, ConstantInitializer(0), False)
    return F.batch_normalization(inp, beta, gamma, mean, var, axes,
                                 decay_rate, eps, batch_stat, output_stat)



def convolution_sn(x, n, kernel, stride, pad, init_method=None, scope_name="", with_bias=True):
    with nn.parameter_scope(scope_name):
        if init_method == "paper":
            init = nn.initializer.NormalInitializer(0.02)
        else:
            s = nn.initializer.calc_normal_std_glorot(x.shape[1], n, kernel=kernel)
            init = nn.initializer.NormalInitializer(s)

        x = PF.convolution(x, n, kernel=kernel, stride=stride,
                           pad=pad, with_bias=with_bias, w_init=init,
                           apply_w=lambda w: PF.spectral_norm(w, dim=0))
    return x


def convolution(x, n, kernel, stride, pad, init_method=None, scope_name="", with_bias=True):  
    with nn.parameter_scope(scope_name):
        if init_method == "paper":
            init = nn.initializer.NormalInitializer(0.02)
        else:
            s = nn.initializer.calc_normal_std_glorot(x.shape[1], n, kernel=kernel)
            init = nn.initializer.NormalInitializer(s)
        x = PF.convolution(x, n, kernel=kernel, stride=stride,
                           pad=pad, with_bias=with_bias, w_init=init)
    return x


def deconvolution(x, n, kernel, stride, pad, init_method=None, scope_name=""):
    with nn.parameter_scope(scope_name):
        if init_method == "paper":
            init = nn.initializer.NormalInitializer(0.02)
        else:
            s = nn.initializer.calc_normal_std_glorot(x.shape[1], n, kernel=kernel)
            init = nn.initializer.NormalInitializer(s)
        x = PF.deconvolution(x, n, kernel=kernel, stride=stride,
                             pad=pad, with_bias=True, w_init=init)
    return x



def convblock_sn(x, n=0, k=(4, 4), s=(2, 2), p=(1, 1), leaky=False, init_method=None, scope_name=""):
    with nn.parameter_scope(scope_name):
       
        x = convolution_sn(x, n=n, kernel=k, stride=s, pad=p, init_method=init_method, scope_name="conv")
       
        x = F.leaky_relu(x, alpha=0.2) if leaky else F.relu(x)
    return x


def convblock(x, n=0, k=(4, 4), s=(2, 2), p=(1, 1), leaky=False, init_method=None, scope_name=""): 
    with nn.parameter_scope(scope_name):
        x = convolution(x, n=n, kernel=k, stride=s, pad=p, init_method=init_method, scope_name="conv")
        x = instance_normalization(x, fix_parameters=True)
        x = F.leaky_relu(x, alpha=0.2) if leaky else F.relu(x)
    return x


def unpool_block(x, n=0, k=(4, 4), s=(2, 2), p=(1, 1), leaky=False, unpool=False, init_method=None, scope_name=""):
    with nn.parameter_scope(scope_name):
        if not unpool:
            logger.info("Deconvolution was used.")
            x = deconvolution(x, n=n, kernel=k, stride=s,
                              pad=p, init_method=init_method, scope_name="deconv")
        else:
            logger.info("Unpooling was used.")
            x = F.unpooling(x, kernel=(2, 2))
            x = convolution(x, n, kernel=(3, 3), stride=(1, 1),
                            pad=(1, 1), init_method=init_method, scope_name="conv")
        x = instance_normalization(x, fix_parameters=True)
        x = F.leaky_relu(x, alpha=0.2) if leaky else F.relu(x)
    return x


def channel_wise_attention_block(x, n_filters):
    """
    Channel-wise Attention block.
    """
    
    gap = F.average_pooling(x, kernel=(x.shape[2], x.shape[3]), stride=(x.shape[2], x.shape[3]))

    
    with nn.parameter_scope("attention_mlp"):
        
        fc1 = PF.affine(gap, n_filters, base_axis=1)
        relu = F.relu(fc1)
       
        fc2 = PF.affine(relu, n_filters, base_axis=1)

    
    attention_weights = F.sigmoid(fc2)

   
    attention_weights = F.reshape(attention_weights, (attention_weights.shape[0], attention_weights.shape[1], 1, 1))  # 重みの形状を変更

    
    out = x * attention_weights

    return out


def spatial_attention_block(x):
    """
    Spatial Attention block.
    """
    
    avg_pool = F.mean(x, axis=1, keepdims=True)
    max_pool = F.max(x, axis=1, keepdims=True)

    
    concat = F.concatenate(avg_pool, max_pool, axis=1)

    
    with nn.parameter_scope("spatial_attention_conv"):
        
        
        conv = convolution(concat, 1, kernel=(7, 7), stride=(1, 1), pad=(3, 3), with_bias=False) 
   
    attention_map = F.sigmoid(conv)

    
    out = x * attention_map

    return out, attention_map


def cbam_block(x, n_filters):
    """
    Combined Channel and Spatial Attention block.
    """
    
    x_channel = channel_wise_attention_block(x, n_filters)

   
    x_spatial, attention_map = spatial_attention_block(x_channel)

    return x_spatial, attention_map


def resblock(x, n=256, init_method=None, scope_name=""):
    with nn.parameter_scope(scope_name):
        r = x
        with nn.parameter_scope('block1'):
            r = convolution(r, n, kernel=(3, 3), pad=(1, 1),
                            stride=(1, 1), init_method=init_method, scope_name="conv")
            r = instance_normalization(r, fix_parameters=True)
            r = F.relu(r)
        with nn.parameter_scope('block2'):
            r = convolution(r, n, kernel=(3, 3), pad=(1, 1),
                            stride=(1, 1), init_method=init_method, scope_name="conv")
            r = instance_normalization(r, fix_parameters=True)

        return x + r
