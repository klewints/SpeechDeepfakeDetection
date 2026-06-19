"""
Enhanced telephony simulation module.

Simulates realistic telephony degradations:
- Codec compression (μ-law companding)
- Packet loss
- Reverberation
- Volume scaling
- Background noise
"""

import numpy as np
import random
from scipy import signal


def apply_mulaw_companding(y, quantize_bits=8):
    """
    Simulate μ-law companding (ITU-T G.711 codec).
    
    Args:
        y: Audio signal (numpy array)
        quantize_bits: Number of bits for quantization (default 8 for telephone)
    
    Returns:
        Compressed audio signal
    """
    # Normalize to [-1, 1]
    y = np.clip(y, -1.0, 1.0)
    
    # μ-law compression
    mu = 255
    y_compressed = np.sign(y) * np.log(1 + mu * np.abs(y)) / np.log(1 + mu)
    
    # Quantization
    max_val = 2 ** (quantize_bits - 1)
    y_quantized = np.round(y_compressed * max_val) / max_val
    
    # Expansion
    y_expanded = np.sign(y_quantized) * (1 / mu) * (np.power(1 + mu, np.abs(y_quantized)) - 1)
    
    return np.clip(y_expanded, -1.0, 1.0)


def apply_packet_loss(y, loss_probability=0.05, packet_duration_ms=20, sr=8000):
    """
    Simulate VoIP packet loss by zeroing out random chunks.
    
    Args:
        y: Audio signal
        loss_probability: Probability of packet loss (0-1)
        packet_duration_ms: Duration of each packet in milliseconds
        sr: Sampling rate
    
    Returns:
        Audio with simulated packet loss
    """
    y = y.copy()
    packet_samples = int(sr * packet_duration_ms / 1000)
    
    num_packets = len(y) // packet_samples
    
    for i in range(num_packets):
        if random.random() < loss_probability:
            start = i * packet_samples
            end = start + packet_samples
            y[start:end] = 0
    
    return np.clip(y, -1.0, 1.0)


def apply_reverberation(y, room_scale=0.3, delay_samples=1600):
    """
    Simulate mild room reverberation using simple echo/convolution.
    
    Args:
        y: Audio signal
        room_scale: Scale of reverb effect (0-1)
        delay_samples: Delay in samples for echo
    
    Returns:
        Audio with reverberation
    """
    # Create simple impulse response (early reflections)
    ir = np.zeros(delay_samples + 1)
    ir[0] = 1.0  # Direct sound
    ir[delay_samples] = room_scale * 0.5  # First reflection
    
    # Add secondary reflections
    ir[delay_samples // 2] = room_scale * 0.3
    ir[int(delay_samples * 1.5)] = room_scale * 0.2
    
    # Convolve with audio (trim to original length)
    y_reverb = signal.fftconvolve(y, ir, mode='same')
    
    return np.clip(y_reverb, -1.0, 1.0)


def apply_telephony_effects(y, sr=8000, apply_codec=True, apply_packet_loss=True, 
                           apply_reverb=True, apply_noise=True, apply_volume=True):
    """
    Apply comprehensive telephony simulation effects.
    
    This function randomly applies telephony degradations to simulate
    realistic voice communication over VoIP/telephone networks.
    
    Args:
        y: Audio signal (numpy array)
        sr: Sampling rate
        apply_codec: Whether to apply codec compression
        apply_packet_loss: Whether to simulate packet loss
        apply_reverb: Whether to add reverberation
        apply_noise: Whether to add background noise
        apply_volume: Whether to apply volume scaling
    
    Returns:
        Audio signal with telephony effects applied
    """
    
    # Normalize input
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    
    # =========================
    # VOLUME SCALING
    # =========================
    if apply_volume and random.random() < 0.8:
        volume_scale = random.uniform(0.6, 1.0)
        y = y * volume_scale
    
    # =========================
    # CODEC COMPRESSION
    # =========================
    if apply_codec and random.random() < 0.7:
        y = apply_mulaw_companding(y, quantize_bits=8)
    
    # =========================
    # PACKET LOSS
    # =========================
    if apply_packet_loss and random.random() < 0.5:
        # Random packet loss parameters
        loss_prob = random.uniform(0.02, 0.1)
        packet_ms = random.choice([20, 30, 40])
        y = apply_packet_loss(y, loss_probability=loss_prob, 
                             packet_duration_ms=packet_ms, sr=sr)
    
    # =========================
    # REVERBERATION
    # =========================
    if apply_reverb and random.random() < 0.6:
        room_scale = random.uniform(0.1, 0.4)
        delay = random.randint(800, 1600)
        y = apply_reverberation(y, room_scale=room_scale, delay_samples=delay)
    
    # =========================
    # BACKGROUND NOISE
    # =========================
    if apply_noise and random.random() < 0.8:
        noise_level = random.uniform(0.003, 0.01)
        noise = np.random.normal(0, noise_level, len(y))
        y = y + noise
    
    # =========================
    # FINAL CLIPPING
    # =========================
    y = np.clip(y, -1.0, 1.0)
    
    return y
