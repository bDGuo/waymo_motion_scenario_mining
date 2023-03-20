from pathlib import Path
import cv2
import argparse
import sys
from rich.progress import track
sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.parameters.tag_parameters import sampling_frequency

#used for presenting the results of the etp and interactor relation

argparser = argparse.ArgumentParser()
argparser.add_argument('--i', type=str, required=True, help='input directory, relative to results folder')
argparser.add_argument('--o', type=str, required=True, help='relative output directory relative to results folder')
argparser.add_argument('--video_name',type=str,required=False,default='video.avi',help='video name')

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]
    args = argparser.parse_args()
    input_dir = ROOT / "results" / "gp1" / "eval" / args.i
    output_dir = ROOT / "results" / "gp1" / "eval" / args.o
    print(f"-i:{str(input_dir)}\n-o:{str(output_dir)}")
    video_name = args.video_name
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    img_list = sorted(input_dir.glob('*.jpg'), key=lambda x: int(x.stem))
    img = cv2.imread(str(img_list[0]))
    height, width, layers = img.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(str(output_dir / video_name), fourcc, int(sampling_frequency/2), (width, height))
    for img_path in track(img_list):
        # print(img_path.name)
        img = cv2.imread(str(img_path))
        video.write(img)
    video.release()
    cv2.destroyAllWindows()