import nnabla as nn
import nnabla.functions as F
# layer モジュールから、必要な全ての関数をファイルの先頭でインポートします
from layer import (convolution, deconvolution, convblock, unpool_block,
                   resblock, cbam_block, convolution_sn, convblock_sn)

class gen_dis:
    def __init__(self, init_method=None, unpool=False, ctx=None):
        self.init_method = init_method
        self.unpool = unpool
        self.ctx = ctx

    # Generatorネットワークの定義 (Spectral Normalizationなし)
    def generator(self, x, scopename, maps=64, unpool=False):
        # ▲▲▲ここから下のブロック全体をインデントしました▲▲▲
        with nn.parameter_scope('generator'):
            with nn.parameter_scope(scopename):
                with nn.parameter_scope('conv1'):
                    x = convblock(x, n=maps, k=(7, 7), s=(1, 1), p=(3, 3),
                                    leaky=False, init_method=self.init_method, scope_name="conv1")
                with nn.parameter_scope('conv2'):
                    x = convblock(x, n=maps*2, k=(3, 3), s=(2, 2), p=(1, 1),
                                    leaky=False, init_method=self.init_method, scope_name="conv2")
                with nn.parameter_scope('conv3'):
                    x = convblock(x, n=maps*4, k=(3, 3), s=(2, 2), p=(1, 1),
                                    leaky=False, init_method=self.init_method, scope_name="conv3")
                for i in range(9):
                    with nn.parameter_scope('res{}'.format(i+1)):
                        x = resblock(x, n=maps*4, init_method=self.init_method, scope_name='res{}'.format(i+1))

                with nn.parameter_scope('attention'):
                    x, attention_map = cbam_block(x, maps * 4)

                with nn.parameter_scope('deconv1'):
                    x = unpool_block(x, n=maps*2, k=(4, 4), s=(2, 2), p=(1, 1),
                                        leaky=False, unpool=unpool, init_method=self.init_method, scope_name="deconv1")
                with nn.parameter_scope('deconv2'):
                    x = unpool_block(x, n=maps, k=(4, 4), s=(2, 2), p=(1, 1),
                                        leaky=False, unpool=unpool, init_method=self.init_method, scope_name="deconv2")
                with nn.parameter_scope('conv4'):
                    x = convolution(x, 3, kernel=(7, 7), stride=(1, 1), pad=(3, 3),
                                    init_method=self.init_method, scope_name="conv4")
                    x = F.tanh(x)
            return x, attention_map

    # Discriminatorネットワークの定義 (Spectral Normalization適用)
    def discriminator(self, x, scopename, maps=64):
        # ▲▲▲ここから下のブロック全体をインデントしました▲▲▲
        with nn.parameter_scope('discriminator'):
            with nn.parameter_scope(scopename):
                with nn.parameter_scope('conv1'):
                    x = convolution_sn(x, maps, kernel=(4, 4), pad=(1, 1), stride=(2, 2),
                                       init_method=self.init_method, scope_name="conv1")
                    x = F.leaky_relu(x, alpha=0.2)
                with nn.parameter_scope('conv2'):
                    x = convblock_sn(x, n=maps*2, k=(4, 4), s=(2, 2), p=(1, 1),
                                     leaky=True, init_method=self.init_method, scope_name="conv2")
                with nn.parameter_scope('conv3'):
                    x = convblock_sn(x, n=maps*4, k=(4, 4), s=(2, 2), p=(1, 1),
                                     leaky=True, init_method=self.init_method, scope_name="conv3")
                with nn.parameter_scope('conv4'):
                    x = convblock_sn(x, n=maps*8, k=(4, 4), s=(1, 1), p=(1, 1),
                                     leaky=True, init_method=self.init_method, scope_name="conv4")
                with nn.parameter_scope('conv5'):
                    x = convolution_sn(x, 1, kernel=(4, 4), pad=(1, 1), stride=(1, 1),
                                       init_method=self.init_method, scope_name="conv5")
        return x

    # Generator A -> B の変換を行うメソッド
    def f(self, x, unpool=False):
        return self.generator(x, 'f', unpool=unpool)

    # Generator B -> A の変換を行うメソッド
    def g(self, x, unpool=False):
        return self.generator(x, 'g', unpool=unpool)

    # Discriminator for domain B のメソッド
    def d_x(self, x):
        return self.discriminator(x, 'x')

    # Discriminator for domain A のメソッド
    def d_y(self, x):
        return self.discriminator(x, 'y')

    # 再構成損失の計算
    def recon_loss(self, x, y):
        return F.mean(F.absolute_error(x, y))

    # LSGAN損失の計算
    def lsgan_loss(self, d_fake, d_real=None, persistent=True):
        if d_real is not None:  # Discriminator loss
            loss_d_real = F.mean(F.pow_scalar(d_real - 1., 2.))
            loss_d_fake = F.mean(F.pow_scalar(d_fake, 2.))
            loss = (loss_d_real + loss_d_fake) * 0.5
        else:  # Generator loss
            loss = F.mean(F.pow_scalar(d_fake - 1., 2.))
            loss.persistent = persistent
        return loss