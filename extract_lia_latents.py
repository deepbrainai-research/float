import os
import subprocess
import tempfile
from collections import namedtuple

import cv2
import face_alignment
import torch
import torchvision
from tqdm import tqdm
import albumentations as A
import albumentations.pytorch.transforms as A_pytorch

from models.float.FLOAT import FLOAT
from options.base_options import BaseOptions

normal_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda self: f"{self.shape}_{normal_repr(self)}"

class InferenceOptions(BaseOptions):
    def __init__(self):
        super().__init__()

    def initialize(self, parser):
        super().initialize(parser)

        parser.add_argument("--ref_path",
                default=None, type=str,help='ref')
        parser.add_argument('--aud_path',
                default=None, type=str, help='audio')
        parser.add_argument('--emo',
                default=None, type=str, help='emotion', choices=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])
        parser.add_argument('--no_crop',
                action = 'store_true', help = 'not using crop')
        parser.add_argument('--res_video_path',
                default=None, type=str, help='res video path')
        parser.add_argument('--ckpt_path',
                default="./checkpoints/float.pth", type=str, help='checkpoint path')
        parser.add_argument('--res_dir',
                default="./results", type=str, help='result dir')
        parser.add_argument('--f', help='Dummy argument to ignore ipykernel\'s --f/--file.', default='_dummy_ipykernel_file_')
        parser.add_argument('--directory', help='directory')
        parser.add_argument('--extension', help='mov or mp4')
        
        return parser


opt = InferenceOptions().parse()
opt.rank, opt.ngpus  = 0, 1

face_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False).face_detector
frame_transform = A.Compose([A.Resize(height=opt.input_size, width=opt.input_size, interpolation=cv2.INTER_AREA),
                             A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
                             A_pytorch.ToTensorV2()])


def find_files(directory, extension):
    mov_files = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(extension):
                full_path = os.path.abspath(os.path.join(root, filename))
                mov_files.append(full_path)
    return mov_files

def load_model(opt):
    model = FLOAT(opt)
    state_dict = torch.load(opt.ckpt_path, map_location='cpu', weights_only=True)
    with torch.no_grad():
        for model_name, model_param in model.named_parameters():
            if model_name in state_dict:
                model_param.copy_(state_dict[model_name].to(opt.rank))
            elif "wav2vec2" in model_name: pass
            else:
                print(f"! Warning; {model_name} not found in state_dict.")

    del state_dict
    return model


model = load_model(opt).to(opt.rank)
model.eval()


def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mult = 360. / frame.shape[0]

    resized_img = cv2.resize(frame, dsize=(0, 0), fx = mult, fy = mult, interpolation=cv2.INTER_AREA if mult < 1. else cv2.INTER_CUBIC)
    bboxes = face_detector.detect_from_image(resized_img)
    bboxes = [(int(x1 / mult), int(y1 / mult), int(x2 / mult), int(y2 / mult), score) for (x1, y1, x2, y2, score) in bboxes if score > 0.95]
    bboxes = bboxes[0] # Just use first bbox

    bsy = int((bboxes[3] - bboxes[1]) / 2)
    bsx = int((bboxes[2] - bboxes[0]) / 2)
    my  = int((bboxes[1] + bboxes[3]) / 2)
    mx  = int((bboxes[0] + bboxes[2]) / 2)

    bs = int(max(bsy, bsx) * 1.6)
    img = cv2.copyMakeBorder(frame, bs, bs, bs, bs, cv2.BORDER_CONSTANT, value=0)
    my, mx  = my + bs, mx + bs  	# BBox center y, bbox center x

    crop_img = img[my - bs:my + bs,mx - bs:mx + bs]
    crop_img = cv2.resize(crop_img, dsize=(opt.input_size, opt.input_size), interpolation=cv2.INTER_AREA if mult < 1. else cv2.INTER_CUBIC)
    return crop_img

def extract_latents(frame):
    frame = preprocess_frame(frame)
    frame = frame_transform(image=frame)['image'].unsqueeze(0)
    s_r, r_s_lambda, s_r_feats = model.encode_image_into_latent(frame.to(opt.rank))
    r_s = model.motion_autoencoder.dec.direction(r_s_lambda)

    return s_r.detach().cpu(), r_s.detach().cpu(), [s_r_feat.detach().cpu() for s_r_feat in s_r_feats]

@torch.no_grad()
def process_video(path, force=False):
    if 'latents' in path:
        print(f"Skipping {path}.")
        return

    if '.mov' in path:
        latent_path = path.replace('.mov', '_lia_latents.pt')
    elif '.mp4' in path:
        latent_path = path.replace('.mp4', '_lia_latents.pt')

    if os.path.exists(latent_path) and not force:
        print(f"Skipping {latent_path} as it already exists.")
        return

    cap = cv2.VideoCapture(path)
    assert cap.isOpened(), f"Error: Unable to open the video file {path}"
    motion_latents = []
    first_s_r = None
    first_r_feats = None


    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_count in tqdm(range(total_frames), desc=f"Processing {os.path.basename(path)}", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break  # If there are no more frames, exit the loop

        s_r, r_s, s_r_feats = extract_latents(frame)
        if frame_count == 0:
            first_s_r = s_r
            first_r_feats = s_r_feats
        motion_latents.append(r_s)

    cap.release()
    motion_latents = torch.stack(motion_latents, dim=0)
    latents = {
        'motion_latents': motion_latents,
        's_r': first_s_r,
        's_r_feats': first_r_feats
    }
    torch.save(latents, latent_path)

@torch.no_grad()
def generate_video(path):
    audio_path = path.replace('_lia_latents.pt', '.wav')
    video_path = path.replace('_lia_latents.pt', '_lia_latents.mp4')
    print(f'Working on {video_path}')

    latents = torch.load(path)
    motion_latents = latents['motion_latents']
    if motion_latents.ndim == 2:
        motion_latents = motion_latents[:, None, :]
    s_r = latents['s_r'].to(opt.rank)
    s_r_feats = [s_r_feat.to(opt.rank) for s_r_feat in latents['s_r_feats']]

    batch_size = 100

    d_hats = []
    for i in range(0, motion_latents.shape[0], batch_size):
        motion_latent_batch = motion_latents[i:i+batch_size].to(opt.rank).transpose(0, 1)
        d_hat = model.decode_latent_into_image(s_r, s_r_feats, motion_latent_batch)['d_hat'].clone().detach().cpu()
        d_hats.append(d_hat)

    del motion_latents, s_r, s_r_feats, latents, motion_latent_batch, d_hat
    d_hats = torch.cat(d_hats, dim=0)
    tqdm.write(f"Image generation done. Shape: {d_hats.shape}")
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
        temp_filename = temp_video.name
        d_hats = d_hats.permute(0, 2, 3, 1).clamp(-1, 1)
        d_hats = ((d_hats + 1) / 2 * 255).type('torch.ByteTensor')
        tqdm.write(f"Writing to {temp_filename}")
        torchvision.io.write_video(temp_filename, d_hats, fps=opt.fps)
        del d_hats

        if audio_path is not None:
            with open(os.devnull, 'wb') as f:
                command =  "ffmpeg -i {} -i {} -c:v copy -c:a aac {} -y".format(temp_filename, audio_path, video_path)
                tqdm.write(f"Running {command}")
                ret = subprocess.call(command, shell=True, stdout=f, stderr=f)
                if ret != 0:
                    tqdm.write(f"ffmpeg failed with return code {ret}")
                    os.remove(temp_filename)
                    return None
            if os.path.exists(video_path):
                os.remove(temp_filename)
        else:
            os.rename(temp_filename, video_path)
        tqdm.write(f"Video saved at {video_path}")
        return video_path


for path in find_files(opt.directory, f".{opt.extension}"):
    process_video(path, force=False)