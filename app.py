import gradio as gr
import os
import sys
import shutil
import uuid
import subprocess
from glob import glob
from huggingface_hub import snapshot_download

# Download models
os.makedirs("checkpoints", exist_ok=True)

snapshot_download(
    repo_id="ByteDance/LatentSync",
    local_dir="./checkpoints"  
)

import tempfile
from moviepy.editor import VideoFileClip
from pydub import AudioSegment

# Adjustable limits (set to None for unlimited)
MAX_VIDEO_DURATION = None  # seconds, e.g., 30 for 30 seconds, None for unlimited
MAX_AUDIO_DURATION = None  # seconds, e.g., 25 for 25 seconds, None for unlimited

def process_video(input_video_path, temp_dir="temp_dir"):
    """
    Optionally crop a given MP4 video to a maximum duration.
    """
    os.makedirs(temp_dir, exist_ok=True)
    video = VideoFileClip(input_video_path)
    input_file_name = os.path.basename(input_video_path)
    output_video_path = os.path.join(temp_dir, f"processed_{input_file_name}")

    if MAX_VIDEO_DURATION is not None and video.duration > MAX_VIDEO_DURATION:
        video = video.subclip(0, MAX_VIDEO_DURATION)
    video.write_videofile(output_video_path, codec="libx264", audio_codec="aac")
    return output_video_path

def process_audio(file_path, temp_dir):
    """
    Optionally crop audio to a maximum duration.
    """
    audio = AudioSegment.from_file(file_path)
    if MAX_AUDIO_DURATION is not None:
        max_duration_ms = MAX_AUDIO_DURATION * 1000
        if len(audio) > max_duration_ms:
            audio = audio[:max_duration_ms]
    output_path = os.path.join(temp_dir, "trimmed_audio.wav")
    audio.export(output_path, format="wav")
    return output_path

import argparse
from omegaconf import OmegaConf
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from diffusers.utils.import_utils import is_xformers_available
from accelerate.utils import set_seed
from latentsync.whisper.audio2feature import Audio2Feature

def main(video_path, audio_path, progress=gr.Progress(track_tqdm=True)):
    """
    Perform lip-sync video generation using an input video and a separate audio track.
    """

    inference_ckpt_path = "checkpoints/latentsync_unet.pt"
    unet_config_path = "configs/unet/second_stage.yaml"
    config = OmegaConf.load(unet_config_path)

    print(f"Input video path: {video_path}")
    print(f"Input audio path: {audio_path}")

    # Process video/audio without hard-coded 10s limit
    temp_dir = tempfile.mkdtemp()
    cropped_video_path = process_video(video_path, temp_dir=temp_dir)
    trimmed_audio_path = process_audio(audio_path, temp_dir=temp_dir)
    video_path = cropped_video_path
    audio_path = trimmed_audio_path

    scheduler = DDIMScheduler.from_pretrained("configs")

    if config.model.cross_attention_dim == 768:
        whisper_model_path = "checkpoints/whisper/small.pt"
    elif config.model.cross_attention_dim == 384:
        whisper_model_path = "checkpoints/whisper/tiny.pt"
    else:
        raise NotImplementedError("cross_attention_dim must be 768 or 384")

    audio_encoder = Audio2Feature(model_path=whisper_model_path, device="cuda", num_frames=config.data.num_frames)

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        inference_ckpt_path,
        device="cpu",
    )

    unet = unet.to(dtype=torch.float16)

    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
    ).to("cuda")

    seed = -1
    if seed != -1:
        set_seed(seed)
    else:
        torch.seed()

    print(f"Initial seed: {torch.initial_seed()}")

    unique_id = str(uuid.uuid4())
    video_out_path = f"video_out_{unique_id}.mp4"

    pipeline(
        video_path=video_path,
        audio_path=audio_path,
        video_out_path=video_out_path,
        video_mask_path=video_out_path.replace(".mp4", "_mask.mp4"),
        num_frames=config.data.num_frames,
        num_inference_steps=config.run.inference_steps,
        guidance_scale=1.0,
        weight_dtype=torch.float16,
        width=config.data.resolution,
        height=config.data.resolution,
    )

    shutil.rmtree(temp_dir)
    return video_out_path


css = """
div#col-container{
    margin: 0 auto;
    max-width: 982px;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# LatentSync: Audio Conditioned Latent Diffusion Models for Lip Sync")
        gr.Markdown("LatentSync, an end-to-end lip sync framework based on audio conditioned latent diffusion models.")
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Video Control", format="mp4")
                audio_input = gr.Audio(label="Audio Input", type="filepath")
                submit_btn = gr.Button("Submit")
            with gr.Column():
                video_result = gr.Video(label="Result")

                gr.Examples(
                    examples=[
                        ["assets/demo1_video.mp4", "assets/demo1_audio.wav"],
                        ["assets/demo2_video.mp4", "assets/demo2_audio.wav"],
                        ["assets/demo3_video.mp4", "assets/demo3_audio.wav"],
                    ],
                    inputs=[video_input, audio_input]
                )

    submit_btn.click(
        fn=main,
        inputs=[video_input, audio_input],
        outputs=[video_result]
    )

demo.queue().launch(share=True, debug=True)
