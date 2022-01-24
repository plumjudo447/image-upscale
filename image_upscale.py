import warnings
warnings.filterwarnings("ignore")

import sys
import os

if hasattr(sys, "_MEIPASS"): 
    mei = sys._MEIPASS
    sys.path.append(mei)
else:
    mei = "./"
    sys.path.append("./")

import argparse
import cv2
import glob
from basicsr.archs.rrdbnet_arch import RRDBNet

from mainmodule import RealESRGANer
from mainmodule.archs.srvgg_arch import SRVGGNetCompact

import tkinter as tk
from tkinter import filedialog
tk.Tk().withdraw()

def main():
    """Inference demo for Real-ESRGAN.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default=filedialog.askopenfilename(
        title="Open image folder",
        filetypes=[
            ("JPEG Image File", ".jpg"),
            ("PNG Image File", ".png")
        ]
    ), help='Input image or folder')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='mainganmodule_x4plus',
        help=('Model names: mainganmodule_x4plus | RealESRNet_x4plus | mainganmodule_x4plus_anime_6B | mainganmodule_x2plus'
              'mainganmodulev2-anime-xsx2 | mainganmodulev2-animevideo-xsx2-nousm | mainganmodulev2-animevideo-xsx2'
              'mainganmodulev2-anime-xsx4 | mainganmodulev2-animevideo-xsx4-nousm | mainganmodulev2-animevideo-xsx4'))
    parser.add_argument('-o', '--output', type=str, default=filedialog.askdirectory(title="Output Folder"), help='Output folder')
    parser.add_argument('-s', '--outscale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored image')
    parser.add_argument('-t', '--tile', type=int, default=200, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--face_enhance', default=True, action='store_true', help='Use maingan to enhance face')
    parser.add_argument('--half', action='store_true', help='Use half precision during inference')
    parser.add_argument(
        '--alpha_upsampler',
        type=str,
        default='mainmodule',
        help='The upsampler for the alpha channels. Options: mainmodule | bicubic')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
    args = parser.parse_args()

    # determine models according to model names
    args.model_name = args.model_name.split('.')[0]
    if args.model_name in ['mainganmodule_x4plus', 'RealESRNet_x4plus']:  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif args.model_name in ['mainganmodule_x4plus_anime_6B']:  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
    elif args.model_name in ['mainganmodule_x2plus']:  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
    elif args.model_name in [
            'mainganmodulev2-anime-xsx2', 'mainganmodulev2-animevideo-xsx2-nousm', 'mainganmodulev2-animevideo-xsx2'
    ]:  # x2 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=2, act_type='prelu')
        netscale = 2
    elif args.model_name in [
            'mainganmodulev2-anime-xsx4', 'mainganmodulev2-animevideo-xsx4-nousm', 'mainganmodulev2-animevideo-xsx4'
    ]:  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4

    # determine model paths
    model_path = os.path.join(mei, 'experiments/pretrained_models', args.model_name + '.pth')
    if not os.path.isfile(model_path):
        model_path = os.path.join(mei, 'mainmodule/weights', args.model_name + '.pth')
    if not os.path.isfile(model_path):
        raise ValueError(f'Model {args.model_name} does not exist.')

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=args.half)

    if args.face_enhance:  # Use maingan for face enhancement
        from maingan import mainganer
        face_enhancer = mainganer(
            model_path='https://github.com/TencentARC/maingan/releases/download/v0.2.0/mainganCleanv1-NoCE-C2.pth',
            upscale=args.outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)
    os.makedirs(args.output, exist_ok=True)

    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        print('Processing : ', idx, imgname)

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None

        try:
            if args.face_enhance:
                _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                output, _ = upsampler.enhance(img, outscale=args.outscale)
        except RuntimeError as error:
            print('Error', error)
            # print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        else:
            if args.ext == 'auto':
                extension = extension[1:]
            else:
                extension = args.ext
            if img_mode == 'RGBA':  # RGBA images should be saved in png format
                extension = 'png'
            save_path = os.path.join(args.output, f'{imgname}_{args.suffix}.{extension}')
            cv2.imwrite(save_path, output)


if __name__ == '__main__':
    main()
    sys.exit(0)
