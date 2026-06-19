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
    """

    # Normalize
    y = np.clip(y, -1.0, 1.0)

    # μ-law compression
    mu = 255

    y_compressed = (
        np.sign(y)
        * np.log(1 + mu * np.abs(y))
        / np.log(1 + mu)
    )

    # Quantization
    max_val = 2 ** (quantize_bits - 1)

    y_quantized = (
        np.round(y_compressed * max_val)
        / max_val
    )

    # Expansion
    y_expanded = (
        np.sign(y_quantized)
        * (1 / mu)
        * (
            np.power(
                1 + mu,
                np.abs(y_quantized)
            ) - 1
        )
    )

    return np.clip(y_expanded, -1.0, 1.0)


def simulate_packet_loss(
    y,
    loss_probability=0.05,
    packet_duration_ms=20,
    sr=8000
):
    """
    Simulate VoIP packet loss.
    """

    y = y.copy()

    packet_samples = int(
        sr * packet_duration_ms / 1000
    )

    num_packets = len(y) // packet_samples

    for i in range(num_packets):

        if random.random() < loss_probability:

            start = i * packet_samples
            end = start + packet_samples

            y[start:end] = 0

    return np.clip(y, -1.0, 1.0)


def apply_reverberation(
    y,
    room_scale=0.3,
    delay_samples=1600
):
    """
    Simulate mild room reverberation.
    """

    # IMPORTANT FIX
    ir_length = int(delay_samples * 1.5) + 1

    ir = np.zeros(ir_length)

    # Direct sound
    ir[0] = 1.0

    # Reflections
    ir[delay_samples] = room_scale * 0.5

    ir[delay_samples // 2] = room_scale * 0.3

    ir[int(delay_samples * 1.5)] = room_scale * 0.2

    # Convolution
    y_reverb = signal.fftconvolve(
        y,
        ir,
        mode='same'
    )

    return np.clip(y_reverb, -1.0, 1.0)


def apply_telephony_effects(
    y,
    sr=8000,
    enable_codec=True,
    enable_packet_loss=True,
    enable_reverb=True,
    enable_noise=True,
    enable_volume=True
):
    """
    Apply realistic telephony degradations.
    """

    # Normalize safely
    max_abs = np.max(np.abs(y))

    if max_abs > 0:
        y = y / max_abs

    # =========================
    # VOLUME SCALING
    # =========================

    if enable_volume and random.random() < 0.8:

        volume_scale = random.uniform(
            0.6,
            1.0
        )

        y = y * volume_scale

    # =========================
    # CODEC COMPRESSION
    # =========================

    if enable_codec and random.random() < 0.7:

        y = apply_mulaw_companding(
            y,
            quantize_bits=8
        )

    # =========================
    # PACKET LOSS
    # =========================

    if enable_packet_loss and random.random() < 0.5:

        loss_prob = random.uniform(
            0.02,
            0.1
        )

        packet_ms = random.choice(
            [20, 30, 40]
        )

        y = simulate_packet_loss(
            y,
            loss_probability=loss_prob,
            packet_duration_ms=packet_ms,
            sr=sr
        )

    # =========================
    # REVERBERATION
    # =========================

    if enable_reverb and random.random() < 0.6:

        room_scale = random.uniform(
            0.1,
            0.4
        )

        delay = random.randint(
            800,
            1600
        )

        y = apply_reverberation(
            y,
            room_scale=room_scale,
            delay_samples=delay
        )

    # =========================
    # BACKGROUND NOISE
    # =========================

    if enable_noise and random.random() < 0.8:

        noise_level = random.uniform(
            0.003,
            0.01
        )

        noise = np.random.normal(
            0,
            noise_level,
            len(y)
        )

        y = y + noise

    # =========================
    # FINAL CLIPPING
    # =========================

    y = np.clip(y, -1.0, 1.0)

    return y