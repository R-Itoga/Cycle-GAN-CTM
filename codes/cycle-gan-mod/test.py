import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
from nnabla.monitor import Monitor
from nnabla.ext_utils import get_extension_context

from args import get_args, save_args
from cycle_gan_data import cycle_gan_data_source, cycle_gan_data_iterator
from gen_dis import gen_dis  # gen_dis クラスをインポート
from helpers import MonitorImageWithName


def test(args):
    # Settings
    b = args.batch_size
    c, h, w = 3, 256, 256
    beta1 = 0.5
    beta2 = 0.999
    lambda_recon = args.lambda_recon
    lambda_idt = args.lambda_idt
    base_lr = args.learning_rate
    init_method = args.init_method

    # Context
    extension_module = args.context
    if args.context is None:
        extension_module = 'cpu'
    logger.info("Running in %s" % extension_module)
    ctx = get_extension_context(extension_module,
                                device_id=args.device_id, type_config=args.type_config)
    nn.set_default_context(ctx)

    # Inputs
    x_real_test = nn.Variable([b, c, h, w], need_grad=False)
    y_real_test = nn.Variable([b, c, h, w], need_grad=False)

    # Models for test
    model_path = args.model_load_path
    # 指定されたパスがディレクトリの場合、その中のparams.h5を読み込む
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, 'params.h5')
    
    logger.info(f"Loading test model from: {model_path}")
    nn.load_parameters(model_path)

    # gen_dis クラスのインスタンスを作成
    gan = gen_dis(init_method=init_method, unpool=args.unpool) 

    # インスタンスメソッドを呼び出す
    y_fake_test, y_attention_map = gan.g(x_real_test, unpool=args.unpool)
    x_fake_test, x_attention_map = gan.f(y_real_test, unpool=args.unpool)  
    y_fake_test.persistent, x_fake_test.persistent = True, True
    y_attention_map.persistent, x_attention_map.persistent = True, True

    
    # Reconstruct
    x_recon_test, _ = gan.f(y_fake_test, unpool=args.unpool)  
    y_recon_test, _ = gan.g(x_fake_test, unpool=args.unpool)  

    # Datasets
    rng = np.random.RandomState(313)
    ds_test_B = cycle_gan_data_source(
        args.dataset, train=False, domain="B", shuffle=False, rng=rng)
    ds_test_A = cycle_gan_data_source(
        args.dataset, train=False, domain="A", shuffle=False, rng=rng)
    di_test_B = cycle_gan_data_iterator(ds_test_B, args.batch_size)
    di_test_A = cycle_gan_data_iterator(ds_test_A, args.batch_size)

    # Monitors
    monitor = Monitor(args.monitor_path)

    def make_monitor_image(name):
        return MonitorImageWithName(name, monitor, interval=1,
                                    normalize_method=lambda x: x + 1.0)
    monitor_test_gx = make_monitor_image('fake_images_test_A')
    monitor_test_fy = make_monitor_image('fake_images_test_B')
    monitor_test_x_recon = make_monitor_image('fake_images_recon_test_B')
    monitor_test_y_recon = make_monitor_image('fake_images_recon_test_A')

 # Validation for B
    logger.info("Validation for B")
    # ループ回数をデータセットの全件をカバーできるように修正
    for i in range((di_test_A.size + args.batch_size - 1) // args.batch_size):
        y_data, _ = di_test_A.next()
        y_real_test.d = y_data
        y_recon_test.forward(clear_buffer=True)
        
        # バッチ内の各画像についてループ処理
        for j in range(y_data.shape[0]):
            file_index = i * args.batch_size + j
            # データセットのサイズを超えたらスキップ（最後のバッチ対策）
            if file_index >= di_test_A.size:
                continue
            
            name = ds_test_A.filename_list[file_index]
            logger.info("generating a fake of {}".format(name))
            
            # 1枚ずつ画像データを抜き出して保存
            # monitor.addは(N, C, H, W)の形状を期待するため、次元を追加
            fake_b_image = np.expand_dims(x_fake_test.d[j], axis=0)
            recon_a_image = np.expand_dims(y_recon_test.d[j], axis=0)
            
            monitor_test_fy.add(name, fake_b_image)
            monitor_test_y_recon.add(name, recon_a_image)

            # 1. アテンションマップを取得 (バッチからj番目を抜き出す)
            heatmap = x_attention_map.d[j, 0] # 形状は (H, W)

            # 2. 元画像のサイズにリサイズ
            heatmap = cv2.resize(heatmap, (w, h))

            # 3. 0-255の整数に変換
            heatmap = np.uint8(255 * heatmap)

            # 4. カラーマップを適用してヒートマップ画像に変換
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # 5. 元画像も0-255の範囲に戻す
            # y_dataは-1~1なので、0~255に変換
            original_image_np = (y_data[j].transpose(1, 2, 0) + 1) / 2.0 * 255.0
            original_image_np = original_image_np.astype(np.uint8)
            original_image_bgr = cv2.cvtColor(original_image_np, cv2.COLOR_RGB2BGR) # OpenCVはBGR順

            # 6. 元画像とヒートマップをブレンド
            blended_image = cv2.addWeighted(original_image_bgr, 0.5, heatmap_colored, 0.5, 0)

            # 7. 保存
            # モニターパスの下にheatmapディレクトリを作成
            heatmap_dir = os.path.join(args.monitor_path, 'heatmaps_B')
            if not os.path.exists(heatmap_dir):
                os.makedirs(heatmap_dir)
            
            save_path = os.path.join(heatmap_dir, f'{name}_heatmap.png')
            cv2.imwrite(save_path, blended_image)
            logger.info(f"Saved attention heatmap to {save_path}")


    # Validation for A
    logger.info("Validation for A")
    # ループ回数をデータセットの全件をカバーできるように修正
    for i in range((di_test_B.size + args.batch_size - 1) // args.batch_size):
        x_data, _ = di_test_B.next()
        x_real_test.d = x_data
        y_fake_test.forward(clear_buffer=True)
        x_recon_test.forward(clear_buffer=True)

        # バッチ内の各画像についてループ処理
        for j in range(x_data.shape[0]):
            file_index = i * args.batch_size + j
            # データセットのサイズを超えたらスキップ（最後のバッチ対策）
            if file_index >= di_test_B.size:
                continue
            
            name = ds_test_B.filename_list[file_index]
            logger.info("generating a fake of {}".format(name))

            # 1枚ずつ画像データを抜き出して保存
            fake_a_image = np.expand_dims(y_fake_test.d[j], axis=0)
            recon_b_image = np.expand_dims(x_recon_test.d[j], axis=0)

            monitor_test_gx.add(name, fake_a_image)
            monitor_test_x_recon.add(name, recon_b_image)

            # 1. アテンションマップを取得 (バッチからj番目を抜き出す)
            heatmap = y_attention_map.d[j, 0] # 形状は (H, W)

            # 2. 元画像のサイズにリサイズ
            heatmap = cv2.resize(heatmap, (w, h))

            # 3. 0-255の整数に変換
            heatmap = np.uint8(255 * heatmap)

            # 4. カラーマップを適用してヒートマップ画像に変換
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # 5. 元画像も0-255の範囲に戻す
            # y_dataは-1~1なので、0~255に変換
            original_image_np = (x_data[j].transpose(1, 2, 0) + 1) / 2.0 * 255.0
            original_image_np = original_image_np.astype(np.uint8)
            original_image_bgr = cv2.cvtColor(original_image_np, cv2.COLOR_RGB2BGR) # OpenCVはBGR順

            # 6. 元画像とヒートマップをブレンド
            blended_image = cv2.addWeighted(original_image_bgr, 0.5, heatmap_colored, 0.5, 0)

            # 7. 保存
            # モニターパスの下にheatmapディレクトリを作成
            heatmap_dir = os.path.join(args.monitor_path, 'heatmaps_A')
            if not os.path.exists(heatmap_dir):
                os.makedirs(heatmap_dir)
            
            save_path = os.path.join(heatmap_dir, f'{name}_heatmap.png')
            cv2.imwrite(save_path, blended_image)
            logger.info(f"Saved attention heatmap to {save_path}")


def main():
    args = get_args()
    save_args(args)
    test(args)


if __name__ == '__main__':
    main()
