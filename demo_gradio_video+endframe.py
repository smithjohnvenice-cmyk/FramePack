from diffusers_helper.hf_login import login

import os

os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

import gradio as gr
import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import argparse
import math
# 20250506 pftq: Added for video input loading
import decord
# 20250506 pftq: Added for progress bars in video_encode
from tqdm import tqdm
# 20250506 pftq: Normalize file paths for Windows compatibility
import pathlib
# 20250506 pftq: for easier to read timestamp
from datetime import datetime
# 20250508 pftq: for saving prompt to mp4 comments metadata
import imageio_ffmpeg
import tempfile
import shutil
import subprocess

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket

parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
args = parser.parse_args()

print(args)

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60

print(f'Free VRAM {free_mem_gb} GB')
print(f'High-VRAM Mode: {high_vram}')

text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()

vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()
transformer.eval()

if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()

transformer.high_quality_fp32_output_for_inference = True
print('transformer.high_quality_fp32_output_for_inference = True')

transformer.to(dtype=torch.bfloat16)
vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)
transformer.requires_grad_(False)

if not high_vram:
    # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but 3x faster
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    text_encoder.to(gpu)
    text_encoder_2.to(gpu)
    image_encoder.to(gpu)
    vae.to(gpu)
    transformer.to(gpu)

stream = AsyncStream()

outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)

# 20250506 pftq: Added function to encode input video frames into latents
@torch.no_grad()
def video_encode(video_path, resolution, no_resize, vae, vae_batch_size=16, device="cuda", width=None, height=None):
    """
    Encode a video into latent representations using the VAE.
    
    Args:
        video_path: Path to the input video file.
        vae: AutoencoderKLHunyuanVideo model.
        height, width: Target resolution for resizing frames.
        vae_batch_size: Number of frames to process per batch.
        device: Device for computation (e.g., "cuda").
    
    Returns:
        start_latent: Latent of the first frame (for compatibility with original code).
        input_image_np: First frame as numpy array (for CLIP vision encoding).
        history_latents: Latents of all frames (shape: [1, channels, frames, height//8, width//8]).
        fps: Frames per second of the input video.
    """
    # 20250506 pftq: Normalize video path for Windows compatibility
    video_path = str(pathlib.Path(video_path).resolve())
    print(f"Processing video: {video_path}")

    # 20250506 pftq: Check CUDA availability and fallback to CPU if needed
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU")
        device = "cpu"

    try:
        # 20250506 pftq: Load video and get FPS
        print("Initializing VideoReader...")
        vr = decord.VideoReader(video_path)
        fps = vr.get_avg_fps()  # Get input video FPS
        num_real_frames = len(vr)
        print(f"Video loaded: {num_real_frames} frames, FPS: {fps}")

        # Truncate to nearest latent size (multiple of 4)
        latent_size_factor = 4
        num_frames = (num_real_frames // latent_size_factor) * latent_size_factor
        if num_frames != num_real_frames:
            print(f"Truncating video from {num_real_frames} to {num_frames} frames for latent size compatibility")
        num_real_frames = num_frames

        # 20250506 pftq: Read frames
        print("Reading video frames...")
        frames = vr.get_batch(range(num_real_frames)).asnumpy()  # Shape: (num_real_frames, height, width, channels)
        print(f"Frames read: {frames.shape}")

        # 20250506 pftq: Get native video resolution
        native_height, native_width = frames.shape[1], frames.shape[2]
        print(f"Native video resolution: {native_width}x{native_height}")
    
        # 20250506 pftq: Use native resolution if height/width not specified, otherwise use provided values
        target_height = native_height if height is None else height
        target_width = native_width if width is None else width
    
        # 20250506 pftq: Adjust to nearest bucket for model compatibility
        if not no_resize:
            target_height, target_width = find_nearest_bucket(target_height, target_width, resolution=resolution)
            print(f"Adjusted resolution: {target_width}x{target_height}")
        else:
            print(f"Using native resolution without resizing: {target_width}x{target_height}")

        # 20250506 pftq: Preprocess frames to match original image processing
        processed_frames = []
        for i, frame in enumerate(frames):
            #print(f"Preprocessing frame {i+1}/{num_frames}")
            frame_np = resize_and_center_crop(frame, target_width=target_width, target_height=target_height)
            processed_frames.append(frame_np)
        processed_frames = np.stack(processed_frames)  # Shape: (num_real_frames, height, width, channels)
        print(f"Frames preprocessed: {processed_frames.shape}")

        # 20250506 pftq: Save first frame for CLIP vision encoding
        input_image_np = processed_frames[0]
        end_of_input_video_image_np = processed_frames[-1]

        # 20250506 pftq: Convert to tensor and normalize to [-1, 1]
        print("Converting frames to tensor...")
        frames_pt = torch.from_numpy(processed_frames).float() / 127.5 - 1
        frames_pt = frames_pt.permute(0, 3, 1, 2)  # Shape: (num_real_frames, channels, height, width)
        frames_pt = frames_pt.unsqueeze(0)  # Shape: (1, num_real_frames, channels, height, width)
        frames_pt = frames_pt.permute(0, 2, 1, 3, 4)  # Shape: (1, channels, num_real_frames, height, width)
        print(f"Tensor shape: {frames_pt.shape}")
        
        # 20250507 pftq: Save pixel frames for use in worker
        input_video_pixels = frames_pt.cpu()

        # 20250506 pftq: Move to device
        print(f"Moving tensor to device: {device}")
        frames_pt = frames_pt.to(device)
        print("Tensor moved to device")

        # 20250506 pftq: Move VAE to device
        print(f"Moving VAE to device: {device}")
        vae.to(device)
        print("VAE moved to device")

        # 20250506 pftq: Encode frames in batches
        print(f"Encoding input video frames in VAE batch size {vae_batch_size} (reduce if memory issues here or if forcing video resolution)")
        latents = []
        vae.eval()
        with torch.no_grad():
            for i in tqdm(range(0, frames_pt.shape[2], vae_batch_size), desc="Encoding video frames", mininterval=0.1):
                #print(f"Encoding batch {i//vae_batch_size + 1}: frames {i} to {min(i + vae_batch_size, frames_pt.shape[2])}")
                batch = frames_pt[:, :, i:i + vae_batch_size]  # Shape: (1, channels, batch_size, height, width)
                try:
                    # 20250506 pftq: Log GPU memory before encoding
                    if device == "cuda":
                        free_mem = torch.cuda.memory_allocated() / 1024**3
                        #print(f"GPU memory before encoding: {free_mem:.2f} GB")
                    batch_latent = vae_encode(batch, vae)
                    # 20250506 pftq: Synchronize CUDA to catch issues
                    if device == "cuda":
                        torch.cuda.synchronize()
                        #print(f"GPU memory after encoding: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                    latents.append(batch_latent)
                    #print(f"Batch encoded, latent shape: {batch_latent.shape}")
                except RuntimeError as e:
                    print(f"Error during VAE encoding: {str(e)}")
                    if device == "cuda" and "out of memory" in str(e).lower():
                        print("CUDA out of memory, try reducing vae_batch_size or using CPU")
                    raise
        
        # 20250506 pftq: Concatenate latents
        print("Concatenating latents...")
        history_latents = torch.cat(latents, dim=2)  # Shape: (1, channels, frames, height//8, width//8)
        print(f"History latents shape: {history_latents.shape}")

        # 20250506 pftq: Get first frame's latent
        start_latent = history_latents[:, :, :1]  # Shape: (1, channels, 1, height//8, width//8)
        end_of_input_video_latent = history_latents[:, :, -1:]  # Shape: (1, channels, 1, height//8, width//8)
        print(f"Start latent shape: {start_latent.shape}")

        # 20250506 pftq: Move VAE back to CPU to free GPU memory
        if device == "cuda":
            vae.to(cpu)
            torch.cuda.empty_cache()
            print("VAE moved back to CPU, CUDA cache cleared")

        return start_latent, input_image_np, history_latents, fps, target_height, target_width, input_video_pixels, end_of_input_video_latent, end_of_input_video_image_np

    except Exception as e:
        print(f"Error in video_encode: {str(e)}")
        raise

    
# 20250507 pftq: New function to encode a single image (end frame)
@torch.no_grad()
def image_encode(image_np, target_width, target_height, vae, image_encoder, feature_extractor, device="cuda"):
    """
    Encode a single image into a latent and compute its CLIP vision embedding.
    
    Args:
        image_np: Input image as numpy array.
        target_width, target_height: Exact resolution to resize the image to (matches start frame).
        vae: AutoencoderKLHunyuanVideo model.
        image_encoder: SiglipVisionModel for CLIP vision encoding.
        feature_extractor: SiglipImageProcessor for preprocessing.
        device: Device for computation (e.g., "cuda").
    
    Returns:
        latent: Latent representation of the image (shape: [1, channels, 1, height//8, width//8]).
        clip_embedding: CLIP vision embedding of the image.
        processed_image_np: Processed image as numpy array (after resizing).
    """
    # 20250507 pftq: Process end frame with exact start frame dimensions
    print("Processing end frame...")
    try:
        print(f"Using exact start frame resolution for end frame: {target_width}x{target_height}")

        # Resize and preprocess image to match start frame
        processed_image_np = resize_and_center_crop(image_np, target_width=target_width, target_height=target_height)

        # Convert to tensor and normalize
        image_pt = torch.from_numpy(processed_image_np).float() / 127.5 - 1
        image_pt = image_pt.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)  # Shape: [1, channels, 1, height, width]
        image_pt = image_pt.to(device)

        # Move VAE to device
        vae.to(device)

        # Encode to latent
        latent = vae_encode(image_pt, vae)
        print(f"image_encode vae output shape: {latent.shape}")

        # Move image encoder to device
        image_encoder.to(device)

        # Compute CLIP vision embedding
        clip_embedding = hf_clip_vision_encode(processed_image_np, feature_extractor, image_encoder).last_hidden_state

        # Move models back to CPU and clear cache
        if device == "cuda":
            vae.to(cpu)
            image_encoder.to(cpu)
            torch.cuda.empty_cache()
            print("VAE and image encoder moved back to CPU, CUDA cache cleared")

        print(f"End latent shape: {latent.shape}")
        return latent, clip_embedding, processed_image_np

    except Exception as e:
        print(f"Error in image_encode: {str(e)}")
        raise
        
# 20250508 pftq: for saving prompt to mp4 metadata comments
def set_mp4_comments_imageio_ffmpeg(input_file, comments):
    try:
        # Get the path to the bundled FFmpeg binary from imageio-ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Error: Input file {input_file} does not exist")
            return False
            
        # Create a temporary file path
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        
        # FFmpeg command using the bundled binary
        command = [
            ffmpeg_path,                   # Use imageio-ffmpeg's FFmpeg
            '-i', input_file,              # input file
            '-metadata', f'comment={comments}',  # set comment metadata
            '-c:v', 'copy',                # copy video stream without re-encoding
            '-c:a', 'copy',                # copy audio stream without re-encoding
            '-y',                          # overwrite output file if it exists
            temp_file                      # temporary output file
        ]
        
        # Run the FFmpeg command
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode == 0:
            # Replace the original file with the modified one
            shutil.move(temp_file, input_file)
            print(f"Successfully added comments to {input_file}")
            return True
        else:
            # Clean up temp file if FFmpeg fails
            if os.path.exists(temp_file):
                os.remove(temp_file)
            print(f"Error: FFmpeg failed with message:\n{result.stderr}")
            return False
            
    except Exception as e:
        # Clean up temp file in case of other errors
        if 'temp_file' in locals() and os.path.exists(temp_file):
            os.remove(temp_file)
        print(f"Error saving prompt to video metadata, ffmpeg may be required: "+str(e))
        return False

# 20250506 pftq: Modified worker to accept video input, and clean frame count
@torch.no_grad()
def worker(input_video, end_frame, end_frame_weight, prompt, n_prompt, seed, batch, resolution, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, no_resize, mp4_crf, num_clean_frames, vae_batch):
    
    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    try:
        # Clean GPU
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        # Text encoding
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))

        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)  # since we only encode one text - that is one model move and one encode, offload is same time consumption since it is also one load and one encode.
            load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # 20250506 pftq: Processing input video instead of image
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Video processing ...'))))

        # 20250506 pftq: Encode video
        start_latent, input_image_np, video_latents, fps, height, width, input_video_pixels, end_of_input_video_latent, end_of_input_video_image_np = video_encode(input_video, resolution, no_resize, vae, vae_batch_size=vae_batch, device=gpu)

        #Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}.png')) 

        # CLIP Vision
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))

        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
        start_embedding = image_encoder_last_hidden_state
        
        end_of_input_video_output = hf_clip_vision_encode(end_of_input_video_image_np, feature_extractor, image_encoder)
        end_of_input_video_last_hidden_state = end_of_input_video_output.last_hidden_state
        end_of_input_video_embedding = end_of_input_video_last_hidden_state

        # 20250507 pftq: Process end frame if provided
        end_latent = None
        end_clip_embedding = None
        if end_frame is not None:
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'End frame encoding ...'))))
            end_latent, end_clip_embedding, _ = image_encode(
                end_frame, target_width=width, target_height=height, vae=vae,
                image_encoder=image_encoder, feature_extractor=feature_extractor, device=gpu
            )

        # Dtype
        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)
        end_of_input_video_embedding = end_of_input_video_embedding.to(transformer.dtype)

        # 20250509 pftq: Restored original placement of total_latent_sections after video_encode
        total_latent_sections = (total_second_length * fps) / (latent_window_size * 4)
        total_latent_sections = int(max(round(total_latent_sections), 1))

        for idx in range(batch):
            if idx > 0:
                seed = seed + 1
            
            if batch > 1:
                print(f"Beginning video {idx+1} of {batch} with seed {seed} ")
            
            job_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+f"_framepack-videoinput-endframe_{width}-{total_second_length}sec_seed-{seed}_steps-{steps}_distilled-{gs}_cfg-{cfg}"
            
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))
    
            rnd = torch.Generator("cpu").manual_seed(seed)
    
            history_latents = video_latents.cpu()
            history_pixels = None
            total_generated_latent_frames = 0
            previous_video = None
            
            
            # 20250509 Generate backwards with end frame for better end frame anchoring
            latent_paddings = list(reversed(range(total_latent_sections)))
            if total_latent_sections > 4:
                latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

            for section_index, latent_padding in enumerate(latent_paddings):
                is_start_of_video = latent_padding == 0
                is_end_of_video = latent_padding == latent_paddings[0]
                latent_padding_size = latent_padding * latent_window_size

                if stream.input_queue.top() == 'end':
                    stream.output_queue.push(('end', None))
                    return

                if not high_vram:
                    unload_complete_models()
                    move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

                if use_teacache:
                    transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
                else:
                    transformer.initialize_teacache(enable_teacache=False)

                def callback(d):
                    try:
                        preview = d['denoised']
                        preview = vae_decode_fake(preview)
                        preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                        preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')
                        if stream.input_queue.top() == 'end':
                            stream.output_queue.push(('end', None))
                            raise KeyboardInterrupt('User ends the task.')
                        current_step = d['i'] + 1
                        percentage = int(100.0 * current_step / steps)
                        hint = f'Sampling {current_step}/{steps}'
                        desc = f'Total frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (total_generated_latent_frames * 4 - 3) / fps) :.2f} seconds (FPS-{fps}), Seed: {seed}, Video {idx+1} of {batch}. Generating part {total_latent_sections - section_index} of {total_latent_sections} backward...'
                        stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
                    except ConnectionResetError as e:
                        print(f"Suppressed ConnectionResetError in callback: {e}")
                    return

                # 20250509 pftq: Dynamic frame allocation like original num_clean_frames, fix split error
                available_frames = video_latents.shape[2] if is_start_of_video else history_latents.shape[2]
                effective_clean_frames = max(0, num_clean_frames - 1) if num_clean_frames > 1 else 1
                if is_start_of_video:
                    effective_clean_frames = 1 # avoid jumpcuts from input video
                clean_latent_pre_frames = effective_clean_frames 
                num_2x_frames = min(2, max(1, available_frames - clean_latent_pre_frames - 1)) if available_frames > clean_latent_pre_frames + 1 else 1
                num_4x_frames = min(16, max(1, available_frames - clean_latent_pre_frames - num_2x_frames)) if available_frames > clean_latent_pre_frames + num_2x_frames else 1
                total_context_frames = num_2x_frames + num_4x_frames
                total_context_frames = min(total_context_frames, available_frames - clean_latent_pre_frames)

                # 20250511 pftq: Dynamically adjust post_frames based on clean_latents_post
                post_frames = 1 if is_end_of_video and end_latent is not None else effective_clean_frames  # 20250511 pftq: Single frame for end_latent, otherwise padding causes still image
                indices = torch.arange(0, clean_latent_pre_frames + latent_padding_size + latent_window_size + post_frames + num_2x_frames + num_4x_frames).unsqueeze(0)
                clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split(
                    [clean_latent_pre_frames, latent_padding_size, latent_window_size, post_frames, num_2x_frames, num_4x_frames], dim=1
                )
                clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

                # 20250509 pftq: Split context frames dynamically for 2x and 4x only
                context_frames = history_latents[:, :, -(total_context_frames + clean_latent_pre_frames):-clean_latent_pre_frames, :, :] if total_context_frames > 0 else history_latents[:, :, :1, :, :]
                split_sizes = [num_4x_frames, num_2x_frames]
                split_sizes = [s for s in split_sizes if s > 0]
                if split_sizes and context_frames.shape[2] >= sum(split_sizes):
                    splits = context_frames.split(split_sizes, dim=2)
                    split_idx = 0
                    clean_latents_4x = splits[split_idx] if num_4x_frames > 0 else history_latents[:, :, :1, :, :]
                    split_idx += 1 if num_4x_frames > 0 else 0
                    clean_latents_2x = splits[split_idx] if num_2x_frames > 0 and split_idx < len(splits) else history_latents[:, :, :1, :, :]
                else:
                    clean_latents_4x = clean_latents_2x = history_latents[:, :, :1, :, :]

                clean_latents_pre = video_latents[:, :, -min(effective_clean_frames, video_latents.shape[2]):].to(history_latents)  # smoother motion but jumpcuts if end frame is too different, must change clean_latent_pre_frames to effective_clean_frames also
                clean_latents_post = history_latents[:, :, :min(effective_clean_frames, history_latents.shape[2]), :, :] # smoother motion, must change post_frames to effective_clean_frames also
                    
                if is_end_of_video:
                    clean_latents_post = torch.zeros_like(end_of_input_video_latent).to(history_latents)
                
                # 20250509 pftq: handle end frame if available
                if end_latent is not None:
                    #current_end_frame_weight = end_frame_weight * (latent_padding / latent_paddings[0])
                    #current_end_frame_weight = current_end_frame_weight * 0.5 + 0.5
                    current_end_frame_weight = end_frame_weight # changing this over time introduces discontinuity
                    # 20250511 pftq: Removed end frame weight adjustment as it has no effect
                    image_encoder_last_hidden_state = (1 - current_end_frame_weight) * end_of_input_video_embedding + end_clip_embedding * current_end_frame_weight
                    image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)
                    
                    # 20250511 pftq: Use end_latent only
                    if is_end_of_video:
                        clean_latents_post = end_latent.to(history_latents)[:, :, :1, :, :]  # Ensure single frame
                    
                # 20250511 pftq: Pad clean_latents_pre to match clean_latent_pre_frames if needed
                if clean_latents_pre.shape[2] < clean_latent_pre_frames:
                    clean_latents_pre = clean_latents_pre.repeat(1, 1, clean_latent_pre_frames // clean_latents_pre.shape[2], 1, 1)
                # 20250511 pftq: Pad clean_latents_post to match post_frames if needed
                if clean_latents_post.shape[2] < post_frames:
                    clean_latents_post = clean_latents_post.repeat(1, 1, post_frames // clean_latents_post.shape[2], 1, 1)
                 
                clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

                max_frames = min(latent_window_size * 4 - 3, history_latents.shape[2] * 4)
                print(f"Generating video {idx+1} of {batch} with seed {seed}, part {total_latent_sections - section_index} of {total_latent_sections} backward")
                generated_latents = sample_hunyuan(
                    transformer=transformer,
                    sampler='unipc',
                    width=width,
                    height=height,
                    frames=max_frames,
                    real_guidance_scale=cfg,
                    distilled_guidance_scale=gs,
                    guidance_rescale=rs,
                    num_inference_steps=steps,
                    generator=rnd,
                    prompt_embeds=llama_vec,
                    prompt_embeds_mask=llama_attention_mask,
                    prompt_poolers=clip_l_pooler,
                    negative_prompt_embeds=llama_vec_n,
                    negative_prompt_embeds_mask=llama_attention_mask_n,
                    negative_prompt_poolers=clip_l_pooler_n,
                    device=gpu,
                    dtype=torch.bfloat16,
                    image_embeddings=image_encoder_last_hidden_state,
                    latent_indices=latent_indices,
                    clean_latents=clean_latents,
                    clean_latent_indices=clean_latent_indices,
                    clean_latents_2x=clean_latents_2x,
                    clean_latent_2x_indices=clean_latent_2x_indices,
                    clean_latents_4x=clean_latents_4x,
                    clean_latent_4x_indices=clean_latent_4x_indices,
                    callback=callback,
                )

                if is_start_of_video:
                    generated_latents = torch.cat([video_latents[:, :, -1:].to(generated_latents), generated_latents], dim=2)

                total_generated_latent_frames += int(generated_latents.shape[2])
                history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

                if not high_vram:
                    offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                    load_model_as_complete(vae, target_device=gpu)

                real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]
                if history_pixels is None:
                    history_pixels = vae_decode(real_history_latents, vae).cpu()
                else:
                    section_latent_frames = (latent_window_size * 2 + 1) if is_start_of_video else (latent_window_size * 2)
                    overlapped_frames = latent_window_size * 4 - 3
                    current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                    history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

                if not high_vram:
                    unload_complete_models()

                output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')
                save_bcthw_as_mp4(history_pixels, output_filename, fps=fps, crf=mp4_crf)
                print(f"Latest video saved: {output_filename}")
                set_mp4_comments_imageio_ffmpeg(output_filename, f"Prompt: {prompt} | Negative Prompt: {n_prompt}")
                print(f"Prompt saved to mp4 metadata comments: {output_filename}")

                if previous_video is not None and os.path.exists(previous_video):
                    try:
                        os.remove(previous_video)
                        print(f"Previous partial video deleted: {previous_video}")
                    except Exception as e:
                        print(f"Error deleting previous partial video {previous_video}: {e}")
                previous_video = output_filename

                print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')
                stream.output_queue.push(('file', output_filename))

                if is_start_of_video:
                    break

            history_pixels = torch.cat([input_video_pixels, history_pixels], dim=2)
            #overlapped_frames = latent_window_size * 4 - 3
            #history_pixels = soft_append_bcthw(input_video_pixels, history_pixels, overlapped_frames)

            output_filename = os.path.join(outputs_folder, f'{job_id}_final.mp4')
            save_bcthw_as_mp4(history_pixels, output_filename, fps=fps, crf=mp4_crf)
            print(f"Final video with input blend saved: {output_filename}")
            set_mp4_comments_imageio_ffmpeg(output_filename, f"Prompt: {prompt} | Negative Prompt: {n_prompt}")
            print(f"Prompt saved to mp4 metadata comments: {output_filename}")
            stream.output_queue.push(('file', output_filename))
    
            if previous_video is not None and os.path.exists(previous_video):
                try:
                    os.remove(previous_video)
                    print(f"Previous partial video deleted: {previous_video}")
                except Exception as e:
                    print(f"Error deleting previous partial video {previous_video}: {e}")
            previous_video = output_filename

            print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')

            stream.output_queue.push(('file', output_filename))

    except:
        traceback.print_exc()

        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

    stream.output_queue.push(('end', None))
    return
    
# 20250506 pftq: Modified process to pass clean frame count, etc
def process(input_video, end_frame, end_frame_weight, prompt, n_prompt, seed, batch, resolution, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, no_resize, mp4_crf, num_clean_frames, vae_batch):
    global stream, high_vram
    # 20250506 pftq: Updated assertion for video input
    assert input_video is not None, 'No input video!'

    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)

    # 20250507 pftq: Even the H100 needs offloading if the video dimensions are 720p or higher
    if high_vram and (no_resize or resolution>640):
        print("Disabling high vram mode due to no resize and/or potentially higher resolution...")
        high_vram = False
        vae.enable_slicing()
        vae.enable_tiling()
        DynamicSwapInstaller.install_model(transformer, device=gpu)
        DynamicSwapInstaller.install_model(text_encoder, device=gpu)
        
    # 20250508 pftq: automatically set distilled cfg to 1 if cfg is used
    if cfg > 1:
        gs = 1

    stream = AsyncStream()

    # 20250506 pftq: Pass num_clean_frames, vae_batch, etc
    async_run(worker, input_video, end_frame, end_frame_weight, prompt, n_prompt, seed, batch, resolution, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, no_resize, mp4_crf, num_clean_frames, vae_batch)

    output_filename = None

    while True:
        flag, data = stream.output_queue.next()

        if flag == 'file':
            output_filename = data
            yield output_filename, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True)

        if flag == 'progress':
            preview, desc, html = data
            #yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)
            yield output_filename, gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True) # 20250506 pftq: Keep refreshing the video in case it got hidden when the tab was in the background

        if flag == 'end':
            yield output_filename, gr.update(visible=False), desc+' Video complete.', '', gr.update(interactive=True), gr.update(interactive=False)
            break

def end_process():
    stream.input_queue.push('end')

quick_prompts = [
    'The girl dances gracefully, with clear movements, full of charm.',
    'A character doing some simple body movements.',
]
quick_prompts = [[x] for x in quick_prompts]

css = make_progress_bar_css()
block = gr.Blocks(css=css).queue(
    max_size=10  # 20250507 pftq: Limit queue size
)
with block:
    # 20250506 pftq: Updated title to reflect video input functionality
    gr.Markdown('# Framepack with Video Input (Video Extension) + End Frame')
    with gr.Row():
        with gr.Column():
        
            # 20250506 pftq: Changed to Video input from Image
            with gr.Row():
                input_video = gr.Video(sources='upload', label="Input Video", height=320)
                with gr.Column():
                  # 20250507 pftq: Added end_frame + weight
                  end_frame = gr.Image(sources='upload', type="numpy", label="End Frame (Optional) - Reduce context frames if very different from input video or it'll jumpcut", height=320)
                  end_frame_weight = gr.Slider(label="End Frame Weight", minimum=0.0, maximum=1.0, value=1.0, step=0.01, info='Reduce to treat more as a reference image.', visible=False) # no effect
                
            prompt = gr.Textbox(label="Prompt", value='')
            #example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Quick List', samples_per_page=1000, components=[prompt])
            #example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt, show_progress=False, queue=False)

            with gr.Row():
                start_button = gr.Button(value="Start Generation")
                end_button = gr.Button(value="End Generation", interactive=False)

            with gr.Group():
                with gr.Row():
                  use_teacache = gr.Checkbox(label='Use TeaCache', value=False, info='Faster speed, but often makes hands and fingers slightly worse.')
                  no_resize = gr.Checkbox(label='Force Original Video Resolution (No Resizing)', value=False, info='Might run out of VRAM (720p requires > 24GB VRAM).')

                seed = gr.Number(label="Seed", value=31337, precision=0)

                batch = gr.Slider(label="Batch Size (Number of Videos)", minimum=1, maximum=1000, value=1, step=1, info='Generate multiple videos each with a different seed.')

                resolution = gr.Number(label="Resolution (max width or height)", value=640, precision=0, visible=False)

                total_second_length = gr.Slider(label="Additional Video Length to Generate (Seconds)", minimum=1, maximum=120, value=5, step=0.1)
                
                # 20250506 pftq: Reduced default distilled guidance scale to improve adherence to input video
                gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=3.0, step=0.01, info='Prompt adherence at the cost of less details from the input video, but to a lesser extent than Context Frames.')
                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=True, info='Use instead of Distilled for more detail/control + Negative Prompt (make sure Distilled=1). Doubles render time.')  # Should not change
                rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)  # Should not change

                n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=True, info='Requires using normal CFG (undistilled) instead of Distilled (set Distilled=1 and CFG > 1).') 
                
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1, info='Expensive. Increase for more quality, especially if using high non-distilled CFG.')
                
                # 20250506 pftq: Renamed slider to Number of Context Frames and updated description
                num_clean_frames = gr.Slider(label="Number of Context Frames (Adherence to Video)", minimum=2, maximum=10, value=5, step=1, info="Expensive. Retain more video details. Reduce if memory issues or if too stuck to input video (jumpcut to end frame, ignoring prompt).")

                default_vae = 32
                if high_vram:
                    default_vae = 128
                elif free_mem_gb>=20:
                    default_vae = 64
                    
                vae_batch = gr.Slider(label="VAE Batch Size for Input Video", minimum=4, maximum=256, value=default_vae, step=4, info="Expensive. Increase for better quality frames during fast motion. Reduce if running out of memory")

                latent_window_size = gr.Slider(label="Latent Window Size", minimum=9, maximum=49, value=9, step=1, visible=True, info='Expensive. Generate more frames at a time (larger chunks). Less degradation but higher VRAM cost.') 

                gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB) (larger means slower)", minimum=6, maximum=128, value=6, step=0.1, info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed.")

                mp4_crf = gr.Slider(label="MP4 Compression", minimum=0, maximum=100, value=16, step=1, info="Lower means better quality. 0 is uncompressed. Change to 16 if you get black outputs. ")

        with gr.Column():
            preview_image = gr.Image(label="Next Latents", height=200, visible=False)
            result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=False, height=512, loop=True)
            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
            progress_bar = gr.HTML('', elem_classes='no-generating-animation')

    gr.HTML("""
        <div style="text-align:center; margin-top:20px;">Share your results and find ideas at the <a href="https://x.com/search?q=framepack&f=live" target="_blank">FramePack Twitter (X) thread</a></div>
    """)

    # 20250506 pftq: Updated inputs to include num_clean_frames
    ips = [input_video, end_frame, end_frame_weight, prompt, n_prompt, seed, batch, resolution, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, no_resize, mp4_crf, num_clean_frames, vae_batch]
    start_button.click(fn=process, inputs=ips, outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button])
    end_button.click(fn=end_process)

block.launch(
    server_name=args.server,
    server_port=args.port,
    share=args.share,
    inbrowser=args.inbrowser,
)
