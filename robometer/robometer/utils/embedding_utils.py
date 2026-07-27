import torch
import numpy as np
from PIL import Image
from tqdm import tqdm


def compute_video_embeddings(
    frames_array: np.ndarray,
    dinov2_model,
    dinov2_processor,
    batch_size: int = 32,
    use_autocast: bool = True,
    use_tqdm: bool = False,
) -> torch.Tensor:
    """
    Compute DINOv2 embeddings for video frames.

    Args:
        frames_array: Video frames as numpy array (T, H, W, C)
        dinov2_model: DINOv2 model instance
        dinov2_processor: DINOv2 processor instance
        batch_size: Batch size for processing frames
        use_autocast: Whether to use mixed precision (autocast) for computation

    Returns:
        Video embeddings as torch tensor (T, D) where D is embedding dimension

    Raises:
        ValueError: If DINOv2 model or processor is not initialized
    """
    if dinov2_model is None or dinov2_processor is None:
        raise ValueError("DINOv2 model not initialized. Set precompute_embeddings=True in config.")

    device = next(dinov2_model.parameters()).device
    embeddings = []

    # Process frames in batches
    num_frames = frames_array.shape[0]

    autocast_context = torch.amp.autocast(device_type="cuda", enabled=use_autocast and torch.cuda.is_available())

    with torch.no_grad(), autocast_context:
        for i in tqdm(range(0, num_frames, batch_size), disable=not use_tqdm):
            end_idx = min(i + batch_size, num_frames)
            batch_frames = frames_array[i:end_idx]

            # Convert numpy frames to PIL images for processing
            if batch_frames.dtype != np.uint8:
                batch_frames = (batch_frames * 255).astype(np.uint8)

            pil_images = [Image.fromarray(frame) for frame in batch_frames]

            # Process with DINOv2
            inputs = dinov2_processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

            outputs = dinov2_model(**inputs)
            # Use pooler_output for global representation
            batch_embeddings = outputs.pooler_output  # (batch_size, embedding_dim)
            embeddings.append(batch_embeddings.cpu())

    # Concatenate all embeddings
    video_embeddings = torch.cat(embeddings, dim=0)  # (T, embedding_dim)
    return video_embeddings


def compute_text_embeddings(
    text: str,
    sentence_model,
    use_autocast: bool = True,
    show_progress_bar: bool = False,
) -> torch.Tensor:
    """
    Compute sentence transformer embeddings for text.

    Args:
        text: Text description
        sentence_model: Sentence transformer model instance
        use_autocast: Whether to use mixed precision (autocast) for computation
        show_progress_bar: Whether to show progress bar during encoding

    Returns:
        Text embedding as torch tensor (D,) where D is embedding dimension

    Raises:
        ValueError: If sentence model is not initialized
    """
    if sentence_model is None:
        raise ValueError("Sentence model not initialized. Set precompute_embeddings=True in config.")

    autocast_context = torch.amp.autocast(device_type="cuda", enabled=use_autocast and torch.cuda.is_available())

    with torch.no_grad(), autocast_context:
        embedding = sentence_model.encode(
            text,
            convert_to_tensor=True,
            show_progress_bar=show_progress_bar,
            batch_size=1,
        )
        return embedding.cpu()
