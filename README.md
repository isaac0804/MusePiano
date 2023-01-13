# MusePiano

This is hobby project to create riffusion-like models dedicated for generating piano music.
The model will learn music in mel spectrogram space.

General Goals:

1. Melody Continuation (Outpainting in both frequency and time dimensions)
2. Auto Accompaniment (Outpainting in frequency dimension)
3. Blending of Melodies (Infilling in either frequency or time dimensions)
4. Text to Melody

Note: Unlike riffusion, this project will not be focusing on text to melody feature, but the first three.

## Proposed Plan

As mentioned, this project will focus on text to melody feature, the main focus is inpainting and outpainting feature.
Current aim is to construct a dataset of spectrograms for piano music.
The model will most likely be diffusion model.
Plan to learn how to use diffusers and accelerate for this project.

## About Riffusion

I was surprised that fine tuning stable diffusion using spectrogram for generating music actually produces some results.
However, the results are far from perfect in my opinion, especially when using it to generate piano music.
On the other hand, the music doesn't really flow when moving from one spectrogram to next.
Hopefully with some modifications and training data dedicated to piano music can improve performance.

