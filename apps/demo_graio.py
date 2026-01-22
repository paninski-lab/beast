"""Gradio demo for BEAST video prediction with time selection and FPS control."""

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml
from sklearn.decomposition import PCA

try:
    import gradio as gr
except ImportError:
    raise ImportError("gradio is required for this demo. Install it with: pip install gradio")

from beast.api.model import Model


def get_run_directory_name(base_dir: Path = Path("runs")) -> str:
    """Get a run directory name with timestamp.
    
    Format: run_YYYY-MM-DD_HH-MM-SS
    If a directory with that name exists, append a counter.
    """
    base_dir.mkdir(exist_ok=True)
    
    # Generate timestamp-based name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_name = f"run_{timestamp}"
    
    # Check if directory exists, if so append counter
    run_dir = base_dir / base_name
    if not run_dir.exists():
        return base_name
    
    # If exists, append counter
    counter = 1
    while True:
        run_dir = base_dir / f"{base_name}_{counter:02d}"
        if not run_dir.exists():
            return run_dir.name
        counter += 1


def visualize_patch_embeddings(
    latents_path: str,
    output_video_path: str,
    image_size: int = 224,
    patch_size: int = 16,
    fps: float = 30.0,
) -> str:
    """Create DINO-style patch embedding visualization video.
    
    Parameters
    ----------
    latents_path: Path to .npy file containing latents [B, Patch+1, Dim]
    output_video_path: Path to save the visualization video
    image_size: Size of input image (default: 224)
    patch_size: Size of patches (default: 16)
    fps: FPS for output video
    
    Returns
    -------
    Path to saved visualization video
    """
    # Load latents
    latents = np.load(latents_path)  # Shape: [B, Patch+1, Dim]
    
    # Skip CLS token (first token)
    patch_embeddings = latents[:, 1:, :]  # Shape: [B, Patch, Dim]
    
    B, num_patches, dim = patch_embeddings.shape
    
    # Calculate patch grid dimensions
    patches_per_side = image_size // patch_size  # e.g., 224 // 16 = 14
    
    # Reshape to [B*Patch, Dim] for PCA
    embeddings_flat = patch_embeddings.reshape(-1, dim)  # [B*Patch, Dim]
    
    # Apply PCA to reduce to 3 dimensions
    pca = PCA(n_components=3)
    embeddings_pca = pca.fit_transform(embeddings_flat)  # [B*Patch, 3]
    
    # Reshape back to [B, Patch, 3]
    embeddings_pca = embeddings_pca.reshape(B, num_patches, 3)  # [B, Patch, 3]
    
    # Normalize to 0-1 range (per channel, across all frames and patches)
    min_val = embeddings_pca.min(axis=(0, 1), keepdims=True)
    max_val = embeddings_pca.max(axis=(0, 1), keepdims=True)
    embeddings_pca = (embeddings_pca - min_val) / (max_val - min_val + 1e-8)
    
    # Reshape to [B, H, W, 3] where H=W=patches_per_side
    embeddings_pca = embeddings_pca.reshape(B, patches_per_side, patches_per_side, 3)
    
    # Scale to 0-255 and convert to uint8
    embeddings_pca = (embeddings_pca * 255).astype(np.uint8)
    
    # Upscale to make visualization more visible
    scale_factor = image_size // patches_per_side  # e.g., 224 // 14 = 16
    target_size = (patches_per_side * scale_factor, patches_per_side * scale_factor)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        str(output_video_path),
        fourcc,
        fps,
        target_size,
    )
    
    if not video_writer.isOpened():
        raise ValueError(f'Failed to open video writer for {output_video_path}')
    
    # Write frames
    for i in range(B):
        frame = embeddings_pca[i]  # [patches_per_side, patches_per_side, 3]
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Upscale frame
        frame_upscaled = cv2.resize(
            frame_bgr,
            target_size,
            interpolation=cv2.INTER_NEAREST,  # Use nearest neighbor to preserve patch boundaries
        )
        video_writer.write(frame_upscaled)
    
    video_writer.release()
    
    # Save PCA embeddings as .npy file (before upscaling, original shape)
    pca_embeddings_path = Path(output_video_path).parent / f"{Path(output_video_path).stem}_pca_embeddings.npy"
    # Save in original patch format [B, Patch, 3]
    np.save(pca_embeddings_path, embeddings_pca.reshape(B, num_patches, 3))
    
    return str(output_video_path)


def extract_video_segment(
    input_video: str,
    output_video: str,
    start_time: float,
    end_time: float,
    fps: float,
) -> str:
    """Extract a video segment using ffmpeg.
    
    Parameters
    ----------
    input_video: Path to input video file
    output_video: Path to output video file
    start_time: Start time in seconds
    end_time: End time in seconds
    fps: Target FPS for output video
    
    Returns
    -------
    Path to extracted video file
    """
    duration = end_time - start_time
    
    # Use ffmpeg to extract segment and set FPS
    cmd = [
        "ffmpeg",
        "-i", input_video,
        "-ss", str(start_time),
        "-t", str(duration),
        "-r", str(fps),
        "-c:v", "libx264",
        "-c:a", "copy",
        "-y",  # Overwrite output file if it exists
        output_video,
    ]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )
    
    return output_video


def process_video(
    video_file: str,
    model_dir: str,
    start_time: float,
    end_time: float,
    fps: float,
    batch_size: int = 32,
    progress: Optional[gr.Progress] = None,
) -> Tuple[Optional[str], Optional[str], str]:
    """Process video with BEAST model.
    
    Parameters
    ----------
    video_file: Path to uploaded video file
    model_dir: Directory containing trained model
    start_time: Start time in seconds
    end_time: End time in seconds
    fps: Target FPS for processing
    batch_size: Batch size for inference
    progress: Gradio progress tracker
    
    Returns
    -------
    Tuple of (reconstruction_video_path, patch_embedding_video_path, status_message)
    """
    try:
        # Validate inputs
        if not video_file:
            return None, None, "Error: Please upload a video file."
        
        if not model_dir or not Path(model_dir).exists():
            return None, None, "Error: Please provide a valid model directory."
        
        if start_time >= end_time:
            return None, None, "Error: Start time must be less than end time."
        
        if fps <= 0:
            return None, None, "Error: FPS must be greater than 0."
        
        status_msg = ""
        
        # Get run directory name with timestamp
        if progress:
            progress(0.1, desc="Setting up run directory...")
        base_dir = Path("runs")
        run_dir_name = get_run_directory_name(base_dir)
        run_dir = base_dir / run_dir_name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        status_msg += f"Run directory: {run_dir}\n"
        
        # Extract video segment
        if progress:
            progress(0.2, desc="Extracting video segment...")
        input_video_path = Path(video_file)
        extracted_video = run_dir / f"extracted_segment_{input_video_path.stem}.mp4"
        
        status_msg += f"Extracting video segment (start: {start_time}s, end: {end_time}s, fps: {fps})...\n"
        
        extract_video_segment(
            str(input_video_path),
            str(extracted_video),
            start_time,
            end_time,
            fps,
        )
        
        status_msg += "✓ Video segment extracted.\n"
        
        # Load model
        if progress:
            progress(0.4, desc="Loading model...")
        status_msg += f"Loading model from {model_dir}...\n"
        model = Model.from_dir(model_dir)
        status_msg += "✓ Model loaded.\n"
        
        # Run prediction
        if progress:
            progress(0.6, desc="Running inference...")
        status_msg += f"Running inference (batch_size: {batch_size})...\n"
        
        output_dir = run_dir / "predictions"
        model.predict_video(
            video_file=str(extracted_video),
            output_dir=str(output_dir),
            batch_size=batch_size,
            save_latents=True,
            save_reconstructions=True,
        )
        
        status_msg += "✓ Inference completed.\n"
        
        # Find reconstruction video
        if progress:
            progress(0.9, desc="Finding reconstruction video...")
        reconstruction_video = output_dir / f"{extracted_video.stem}_reconstruction.mp4"
        
        if not reconstruction_video.exists():
            return None, None, status_msg + "\nError: Reconstruction video not found."
        
        status_msg += f"✓ Results saved to: {run_dir}\n"
        status_msg += f"✓ Reconstruction video: {reconstruction_video}\n"
        
        # Find latents file and create patch embedding visualization
        patch_viz_video = None
        latents_file = output_dir / f"{extracted_video.stem}.npy"
        if latents_file.exists():
            status_msg += f"✓ Latents saved to: {latents_file}\n"
            
            # Create patch embedding visualization
            if progress:
                progress(0.95, desc="Creating patch embedding visualization...")
            status_msg += "Creating patch embedding visualization...\n"
            
            try:
                # Get model config to determine image_size and patch_size
                config_path = Path(model_dir) / "config.yaml"
                if config_path.exists():
                    with open(config_path) as f:
                        model_config = yaml.safe_load(f)
                    image_size = model_config['model']['model_params'].get('image_size', 224)
                    patch_size = model_config['model']['model_params'].get('patch_size', 16)
                else:
                    image_size = 224
                    patch_size = 16
                
                patch_viz_video = output_dir / f"{extracted_video.stem}_patch_embeddings.mp4"
                visualize_patch_embeddings(
                    str(latents_file),
                    str(patch_viz_video),
                    image_size=image_size,
                    patch_size=patch_size,
                    fps=fps,
                )
                status_msg += f"✓ Patch embedding video: {patch_viz_video}\n"
            except Exception as e:
                status_msg += f"⚠ Warning: Could not create patch visualization: {str(e)}\n"
        else:
            status_msg += "⚠ Warning: Latents file not found, skipping patch visualization.\n"
        
        if progress:
            progress(1.0, desc="Complete!")
        return str(reconstruction_video), str(patch_viz_video) if patch_viz_video and patch_viz_video.exists() else None, status_msg
        
    except subprocess.CalledProcessError as e:
        error_msg = f"Error during video extraction: {e.stderr}"
        return None, None, error_msg
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return None, None, error_msg


def create_interface():
    """Create and launch the Gradio interface."""
    
    with gr.Blocks(title="BEAST Video Prediction Demo") as demo:
        gr.Markdown(
            """
            # BEAST Video Prediction Demo
            
            Upload a video, select a time segment, and run BEAST inference to generate reconstructions and latents.
            """
        )
        
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(
                    label="Upload Video",
                )
                
                model_dir_input = gr.Textbox(
                    label="Model Directory",
                    placeholder="Path to trained model directory (e.g., runs/2026-01-15/05-02-16)",
                    value="",
                )
                
                with gr.Row():
                    start_time = gr.Number(
                        label="Start Time (seconds)",
                        value=0.0,
                        minimum=0.0,
                    )
                    end_time = gr.Number(
                        label="End Time (seconds)",
                        value=10.0,
                        minimum=0.0,
                    )
                
                fps = gr.Number(
                    label="FPS",
                    value=30.0,
                    minimum=1.0,
                    maximum=120.0,
                )
                
                batch_size = gr.Number(
                    label="Batch Size",
                    value=32,
                    minimum=1,
                    maximum=128,
                    step=1,
                )
                
                process_btn = gr.Button("Process Video", variant="primary")
            
            with gr.Column():
                status_output = gr.Textbox(
                    label="Status",
                    lines=10,
                    interactive=False,
                )
                
                reconstruction_output = gr.Video(
                    label="Reconstruction Video",
                )
                
                patch_embedding_output = gr.Video(
                    label="Patch Embedding Visualization",
                )
        
        process_btn.click(
            fn=process_video,
            inputs=[
                video_input,
                model_dir_input,
                start_time,
                end_time,
                fps,
                batch_size,
            ],
            outputs=[reconstruction_output, patch_embedding_output, status_output],
        )
        
        gr.Markdown(
            """
            ## Instructions
            
            1. Upload a video file
            2. Enter the path to your trained model directory
            3. Set the start and end times for the segment you want to process
            4. Set the target FPS for the output
            5. Click "Process Video" to run inference
            6. The reconstruction video and patch embedding visualization will be displayed below
            
            ## Outputs
            
            - **Reconstruction Video**: Model's reconstruction of the input video
            - **Patch Embedding Visualization**: DINO-style visualization showing patch embeddings reduced to 3D via PCA and colorized
            
            Results (reconstructions, latents, and patch embeddings) are saved in timestamped run folders: `runs/run_YYYY-MM-DD_HH-MM-SS/`, etc.
            """
        )
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True, server_name="0.0.0.0", server_port=None)
