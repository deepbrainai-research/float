"""
FLOAT - Gradio Interface for Audio-Driven Talking Face Generation
"""

import os
import gradio as gr
import datetime
from pathlib import Path

# Import the inference components
from generate import InferenceAgent, InferenceOptions


class GradioInterface:
    def __init__(self):
        # Initialize options with defaults
        self.opt = InferenceOptions().parse()
        self.opt.rank, self.opt.ngpus = 0, 1
        
        # Create results directory
        os.makedirs(self.opt.res_dir, exist_ok=True)
        
        # Initialize the inference agent
        print("Loading FLOAT model...")
        self.agent = InferenceAgent(self.opt)
        print("Model loaded successfully!")
    
    def generate_video(
        self,
        ref_image,
        audio_file,
        emotion,
        a_cfg_scale,
        r_cfg_scale,
        e_cfg_scale,
        nfe,
        seed,
        no_crop,
        progress=gr.Progress()
    ):
        """
        Generate talking face video from reference image and audio
        """
        try:
            progress(0, desc="Preparing inputs...")
            
            # Validate inputs
            if ref_image is None:
                return None, "❌ Please upload a reference image"
            if audio_file is None:
                return None, "❌ Please upload an audio file"
            
            # Generate output filename
            video_name = Path(ref_image).stem
            audio_name = Path(audio_file).stem
            call_time = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            
            res_video_path = os.path.join(
                self.opt.res_dir,
                f"{call_time}-{video_name}-{audio_name}-nfe{nfe}-seed{seed}-acfg{a_cfg_scale}-ecfg{e_cfg_scale}-{emotion}.mp4"
            )
            
            progress(0.3, desc="Running inference...")
            
            # Run inference
            output_path = self.agent.run_inference(
                res_video_path=res_video_path,
                ref_path=ref_image,
                audio_path=audio_file,
                a_cfg_scale=a_cfg_scale,
                r_cfg_scale=r_cfg_scale,
                e_cfg_scale=e_cfg_scale,
                emo=emotion,
                nfe=nfe,
                no_crop=no_crop,
                seed=seed,
                verbose=True
            )
            
            progress(1.0, desc="Complete!")
            
            status_msg = f"✅ Video generated successfully!\n📁 Saved to: {output_path}"
            return output_path, status_msg
            
        except Exception as e:
            error_msg = f"❌ Error during generation: {str(e)}"
            print(error_msg)
            return None, error_msg


def create_interface():
    """Create and configure the Gradio interface"""
    
    interface = GradioInterface()
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    """
    
    with gr.Blocks(css=custom_css, title="FLOAT - Talking Face Generation") as demo:
        
        gr.Markdown(
            """
            # 🎭 FLOAT: Audio-Driven Talking Face Generation
            
            Generate realistic talking face videos from a reference image and audio file.
            Upload your inputs and adjust the parameters below to create your video.
            """
        )
        
        with gr.Row():
            # Left column - Inputs
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Input Files")
                
                ref_image = gr.Image(
                    label="Reference Image",
                    type="filepath",
                    sources=["upload"],
                    height=300
                )
                gr.Markdown("*Upload a clear frontal face image*")
                
                audio_file = gr.Audio(
                    label="Audio File",
                    type="filepath",
                    sources=["upload"]
                )
                gr.Markdown("*Upload the audio/speech file*")
                
                with gr.Accordion("⚙️ Advanced Options", open=False):
                    emotion = gr.Dropdown(
                        choices=['S2E', 'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
                        value='S2E',
                        label="Emotion Control",
                        info="Choose target emotion or 'S2E' for speech-to-emotion"
                    )
                    
                    no_crop = gr.Checkbox(
                        label="Skip Face Cropping",
                        value=False,
                        info="Enable if image is already cropped and aligned"
                    )
                    
                    seed = gr.Slider(
                        minimum=0,
                        maximum=10000,
                        value=25,
                        step=1,
                        label="Random Seed",
                        info="Set seed for reproducible results"
                    )
            
            # Right column - Parameters & Output
            with gr.Column(scale=1):
                gr.Markdown("### 🎛️ Generation Parameters")
                
                with gr.Group():
                    nfe = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=10,
                        step=1,
                        label="Number of Function Evaluations (NFE)",
                        info="Higher = better quality but slower (10-20 recommended)"
                    )
                    
                    a_cfg_scale = gr.Slider(
                        minimum=0.0,
                        maximum=5.0,
                        value=2.0,
                        step=0.1,
                        label="Audio CFG Scale",
                        info="Audio guidance strength (1.5-3.0 recommended)"
                    )
                    
                    r_cfg_scale = gr.Slider(
                        minimum=0.0,
                        maximum=3.0,
                        value=1.0,
                        step=0.1,
                        label="Reference CFG Scale",
                        info="Reference image guidance strength"
                    )
                    
                    e_cfg_scale = gr.Slider(
                        minimum=0.0,
                        maximum=3.0,
                        value=1.0,
                        step=0.1,
                        label="Emotion CFG Scale",
                        info="Emotion control guidance strength"
                    )
                
                generate_btn = gr.Button(
                    "🚀 Generate Video",
                    variant="primary",
                    size="lg"
                )
                
                status_output = gr.Textbox(
                    label="Status",
                    placeholder="Status messages will appear here...",
                    lines=3
                )
                
                video_output = gr.Video(
                    label="Generated Video",
                    height=400
                )
        
        # Parameter presets
        with gr.Accordion("📋 Parameter Presets", open=False):
            gr.Markdown(
                """
                ### Quick Presets
                
                **Fast Preview** (NFE=5, A_CFG=2.0)
                - Quick generation for testing
                - Lower quality but fast
                
                **Balanced** (NFE=10, A_CFG=2.0) ⭐ *Default*
                - Good balance of quality and speed
                - Recommended for most uses
                
                **High Quality** (NFE=20, A_CFG=2.5)
                - Best quality output
                - Slower generation time
                
                **Expressive** (NFE=15, E_CFG=1.5)
                - Enhanced emotional expressions
                - Good for dramatic content
                """
            )
            
            with gr.Row():
                preset_fast = gr.Button("⚡ Fast Preview")
                preset_balanced = gr.Button("⚖️ Balanced")
                preset_quality = gr.Button("💎 High Quality")
                preset_expressive = gr.Button("🎭 Expressive")
        
        # Information section
        with gr.Accordion("ℹ️ Help & Information", open=False):
            gr.Markdown(
                """
                ## How to Use
                
                1. **Upload Reference Image**: Choose a clear, frontal face image (512x512 recommended)
                2. **Upload Audio**: Select the audio file for lip-sync generation
                3. **Adjust Parameters**: Modify generation settings or use presets
                4. **Generate**: Click the generate button and wait for processing
                
                ## Parameter Guide
                
                - **NFE**: Controls generation steps. Higher = better quality but slower
                - **Audio CFG**: Controls how closely video follows audio. Higher = stricter sync
                - **Reference CFG**: Controls identity preservation. Higher = more similar to reference
                - **Emotion CFG**: Controls emotion expression strength
                - **Emotion**: Choose specific emotion or 'S2E' for automatic emotion from speech
                
                ## Tips
                
                - Use high-quality, well-lit reference images for best results
                - Audio should be clear with minimal background noise
                - Start with default parameters and adjust based on results
                - Enable "Skip Face Cropping" only if your image is pre-processed
                
                ## Supported Formats
                
                - **Images**: JPG, PNG
                - **Audio**: WAV, MP3, M4A, FLAC
                """
            )
        
        # Event handlers
        def set_fast_preset():
            return 5, 2.0, 1.0, 1.0
        
        def set_balanced_preset():
            return 10, 2.0, 1.0, 1.0
        
        def set_quality_preset():
            return 20, 2.5, 1.0, 1.0
        
        def set_expressive_preset():
            return 15, 2.0, 1.0, 1.5
        
        # Connect preset buttons
        preset_fast.click(
            fn=set_fast_preset,
            outputs=[nfe, a_cfg_scale, r_cfg_scale, e_cfg_scale]
        )
        
        preset_balanced.click(
            fn=set_balanced_preset,
            outputs=[nfe, a_cfg_scale, r_cfg_scale, e_cfg_scale]
        )
        
        preset_quality.click(
            fn=set_quality_preset,
            outputs=[nfe, a_cfg_scale, r_cfg_scale, e_cfg_scale]
        )
        
        preset_expressive.click(
            fn=set_expressive_preset,
            outputs=[nfe, a_cfg_scale, r_cfg_scale, e_cfg_scale]
        )
        
        # Connect generate button
        generate_btn.click(
            fn=interface.generate_video,
            inputs=[
                ref_image,
                audio_file,
                emotion,
                a_cfg_scale,
                r_cfg_scale,
                e_cfg_scale,
                nfe,
                seed,
                no_crop
            ],
            outputs=[video_output, status_output]
        )
        
        # Examples
        gr.Markdown("### 📚 Example Configurations")
        gr.Examples(
            examples=[
                ["S2E", 2.0, 1.0, 1.0, 10, 25, False],
                ["happy", 2.5, 1.0, 1.5, 15, 42, False],
                ["neutral", 2.0, 1.2, 1.0, 20, 100, False],
            ],
            inputs=[emotion, a_cfg_scale, r_cfg_scale, e_cfg_scale, nfe, seed, no_crop],
            label="Try these parameter combinations"
        )
    
    return demo


def parse_launch_args():
    """Parse command-line arguments for Gradio launch"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Launch FLOAT Gradio Interface')
    
    # Server options
    parser.add_argument('--port', type=int, default=7860,
                        help='Port to run the server on (default: 7860)')
    parser.add_argument('--server_name', type=str, default="0.0.0.0",
                        help='Server name/IP to bind to (default: 0.0.0.0)')
    parser.add_argument('--share', action='store_true',
                        help='Create a public share link')
    parser.add_argument('--auth', type=str, default=None,
                        help='Username and password for authentication, format: "username:password"')
    parser.add_argument('--auth_message', type=str, default="Please login to access FLOAT",
                        help='Message to display on login page')
    
    # Model options
    parser.add_argument('--ckpt_path', type=str, 
                        default="./checkpoints/float.pth",
                        help='Path to model checkpoint')
    parser.add_argument('--res_dir', type=str, default="./results",
                        help='Directory to save generated videos')
    parser.add_argument('--wav2vec_model_path', type=str, default="facebook/wav2vec2-base-960h",
                        help='Path to wav2vec2 model')
    
    # Interface options
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with detailed error messages')
    parser.add_argument('--queue', action='store_true', default=True,
                        help='Enable request queuing (default: True)')
    parser.add_argument('--max_threads', type=int, default=4,
                        help='Maximum number of concurrent threads (default: 4)')
    parser.add_argument('--inbrowser', action='store_true',
                        help='Automatically open in browser')
    parser.add_argument('--prevent_thread_lock', action='store_true',
                        help='Prevent thread lock (useful for debugging)')
    
    # Advanced options
    parser.add_argument('--ssl_keyfile', type=str, default=None,
                        help='Path to SSL key file for HTTPS')
    parser.add_argument('--ssl_certfile', type=str, default=None,
                        help='Path to SSL certificate file for HTTPS')
    parser.add_argument('--ssl_keyfile_password', type=str, default=None,
                        help='Password for SSL key file')
    parser.add_argument('--favicon_path', type=str, default=None,
                        help='Path to custom favicon')
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse launch arguments
    args = parse_launch_args()
    
    # Override inference options with command-line args if provided
    import sys
    inference_args = []
    if args.ckpt_path:
        inference_args.extend(['--ckpt_path', args.ckpt_path])
    if args.res_dir:
        inference_args.extend(['--res_dir', args.res_dir])
    if args.wav2vec_model_path:
        inference_args.extend(['--wav2vec_model_path', args.wav2vec_model_path])
    
    # Temporarily modify sys.argv for InferenceOptions
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]] + inference_args
    
    # Create and launch the interface
    demo = create_interface()
    
    # Restore original argv
    sys.argv = original_argv
    
    # Parse authentication if provided
    auth_tuple = None
    if args.auth:
        try:
            username, password = args.auth.split(':', 1)
            auth_tuple = (username, password)
            print(f"🔒 Authentication enabled for user: {username}")
        except ValueError:
            print("⚠️ Invalid auth format. Use 'username:password'")
    
    # Print launch information
    print("\n" + "="*60)
    print("🚀 Launching FLOAT Gradio Interface")
    print("="*60)
    print(f"📍 Server: {args.server_name}:{args.port}")
    print(f"🔗 Local URL: http://localhost:{args.port}")
    if args.share:
        print("🌐 Public sharing: Enabled")
    if auth_tuple:
        print(f"🔒 Authentication: Enabled")
    print(f"💾 Results directory: {args.res_dir}")
    print(f"🤖 Model checkpoint: {args.ckpt_path}")
    print("="*60 + "\n")
    
    # Launch configuration
    launch_kwargs = {
        'server_name': args.server_name,
        'server_port': args.port,
        'share': args.share,
        'show_error': True,
        'inbrowser': args.inbrowser,
        'prevent_thread_lock': args.prevent_thread_lock,
    }
    
    # Add optional parameters
    if auth_tuple:
        launch_kwargs['auth'] = auth_tuple
        launch_kwargs['auth_message'] = args.auth_message
    
    if args.ssl_keyfile and args.ssl_certfile:
        launch_kwargs['ssl_keyfile'] = args.ssl_keyfile
        launch_kwargs['ssl_certfile'] = args.ssl_certfile
        if args.ssl_keyfile_password:
            launch_kwargs['ssl_keyfile_password'] = args.ssl_keyfile_password
        print("🔐 HTTPS enabled")
    
    if args.favicon_path:
        launch_kwargs['favicon_path'] = args.favicon_path
    
    if args.debug:
        launch_kwargs['debug'] = True
        print("🐛 Debug mode enabled")
    
    # Launch the demo
    try:
        if args.queue:
            demo.queue(max_size=args.max_threads)
        
        demo.launch(**launch_kwargs)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"\n❌ Error launching interface: {e}")
        if args.debug:
            raise
