import argparse

from uni3r.utils.path_manager import init_all_submodules
init_all_submodules()

from uni3r.model import LSM_Dust3R
from uni3r.utils.visualization_utils import render_video_from_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_list', type=str, nargs='+', required=True,
                        help='List of input image files or directories')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--n_interp', type=int, default=90)
    parser.add_argument('--fps', type=int, default=30)

    args = parser.parse_args()
    
    # 1. load model
    model = LSM_Dust3R.from_pretrained(args.model_path)
    model.eval()

    # 2. render video
    render_video_from_file(args.file_list, model, args.output_path, resolution=args.resolution, n_interp=args.n_interp, fps=args.fps)