# Technical Report: Multimodal Emotion Recognition on RAVDESS

## 1. Overview

This report documents the multimodal emotion recognition pipeline implemented in [Emotion_Recognition.ipynb](./Emotion_Recognition.ipynb). The notebook builds three model variants on the RAVDESS speech dataset:

1. `EmotionCNN (Audio)` for mel-spectrogram based speech emotion recognition
2. `TextRNN` for transcript-based classification
3. `Late Fusion` for combining the audio and text predictions

The code uses JAX, Flax `nnx`, Optax, Librosa, Whisper, and a custom C++ BPE tokenizer. The evaluation protocol is centered on Leave-One-Subject-Out (LOSO) validation so that performance reflects cross-speaker generalization rather than memorization of actor-specific vocal traits.

## 2. Dataset and Preprocessing

- Dataset: RAVDESS speech subset with `1440` audio files
- Classes: `8` emotions
  - neutral
  - calm
  - happy
  - sad
  - angry
  - fearful
  - disgust
  - surprised
- Input duration: audio is padded or truncated to `3.0 s`
- Audio representation: `128`-bin log-mel spectrograms
- Text representation: Whisper transcripts encoded with a custom BPE tokenizer

### Preprocessing Summary

**Audio branch**

- Resample to `22,050 Hz`
- Compute mel spectrogram with:
  - `n_fft = 1024`
  - `hop_length = 256`
  - `n_mels = 128`
  - `fmin = 50`
  - `fmax = 8000`
- Convert power spectrogram to dB
- Min-max normalize
- Cache each spectrogram to disk as `.npy`

**Text branch**

- Transcribe each audio file with Whisper
- Train/use a custom BPE tokenizer
- Encode transcripts into token IDs
- Pad all sequences to a fixed maximum length for JAX/XLA compatibility

## 3. Evaluation Protocol

The notebook uses two evaluation setups:

1. An initial random split:
   - `80%` train+validation, `20%` test
   - followed by a `90/10` split inside the train+validation portion
2. A final `LOSO` evaluation:
   - one actor held out as test
   - one different actor held out as validation
   - remaining `22` actors used for training

This LOSO design is a strong choice for RAVDESS because emotion recognition systems can otherwise overfit to speaker identity. The notebook also applies class-weighted loss because the neutral class is underrepresented.

## 4. Architecture Diagram

```mermaid
flowchart LR
    A[RAVDESS WAV files] --> B[Parse labels and actor IDs]
    B --> C1[Audio preprocessing]
    B --> C2[Whisper transcription]

    C1 --> D1[3 s waveform normalization]
    D1 --> E1[128-bin log-mel spectrogram]
    E1 --> F1[SpecAugment]
    F1 --> G1[1D CNN block 1<br/>Conv-BN-GELU-MaxPool]
    G1 --> H1[1D CNN block 2<br/>Conv-BN-GELU-MaxPool]
    H1 --> I1[1D CNN block 3<br/>Conv-BN-GELU-MaxPool]
    I1 --> J1[Global max pooling]
    J1 --> K1[Linear 256->128]
    K1 --> L1[Dropout 0.3]
    L1 --> M1[Linear 128->64]
    M1 --> N1[Audio logits 8]

    C2 --> D2[Custom BPE tokenizer]
    D2 --> E2[Embedding 64]
    E2 --> F2[GRU hidden 128]
    F2 --> G2[Masked mean pooling]
    G2 --> H2[Dropout 0.3]
    H2 --> I2[Linear 128->8]
    I2 --> J2[Text logits 8]

    N1 --> K3[Softmax]
    J2 --> L3[Softmax]
    K3 --> M3[Late fusion]
    L3 --> M3
    M3 --> N3[Learned scalar alpha]
    N3 --> O3[Fused class probabilities]
```

## 5. Design Rationale

### Why log-mel spectrograms?

Emotion in speech is often carried by spectral shape, energy, and temporal prosody. Log-mel spectrograms preserve this information while remaining compact and well-suited to convolutional processing.

### Why a 1D CNN for audio?

The audio model treats the spectrogram as a time sequence with `128` mel features per step. This keeps the model lighter than a 2D CNN, which is helpful because RAVDESS is relatively small. The architecture emphasizes temporal pattern extraction while preserving a manageable parameter count.

### Why SpecAugment?

RAVDESS is limited in size, so overfitting is a real risk. Frequency and time masking provide simple but effective regularization by forcing the audio branch to rely on distributed cues instead of narrow local artifacts.

### Why a GRU-based text model?

The text branch exists to test whether transcript content adds any useful signal. A GRU with masked mean pooling is computationally cheap, sequence-aware, and appropriate for short transcriptions.

### Why late fusion instead of early fusion?

The notebook explicitly notes that text is weak for this dataset because actors only speak two fixed sentences. A learned scalar late-fusion weight is therefore a sensible design:

- it keeps the fusion module interpretable
- it limits overfitting
- it allows the model to down-weight the unreliable text branch automatically

### Why class-weighted loss?

The neutral class has fewer examples than the others. Inverse-frequency class weights help reduce bias toward overrepresented emotions during both optimization and weighted accuracy reporting.

### Why LOSO instead of only a random split?

Speaker leakage is a major concern in speech emotion recognition. LOSO tests whether the model generalizes to unseen actors, which is a more realistic measure of robustness than a random file-level split.

## 6. Model Summary

| Model | Input | Core architecture | Output |
|---|---|---|---|
| EmotionCNN (Audio) | `(128, T, 1)` log-mel spectrogram | SpecAugment -> 3 x Conv1D blocks -> global max pool -> MLP | 8-class logits |
| TextRNN | padded BPE token IDs | Embedding -> GRU -> masked mean pooling -> linear classifier | 8-class logits |
| Late Fusion | audio and text probabilities | learned scalar mixing weight `alpha` | fused class probabilities |

## 7. Results Table

The checked-in notebook contains the full training and LOSO evaluation code, but the committed `.ipynb` snapshot does **not** preserve the runtime outputs for the training and evaluation cells. Because of that, the exact final `Accuracy` and `F1-score` values cannot be recovered faithfully from the repository alone.

To keep this report technically honest, the table below records what can be concluded from the notebook source and markdown commentary without inventing numbers.

| Model | Accuracy | F1-score | Evidence from notebook |
|---|---:|---:|---|
| EmotionCNN (Audio) | Not recoverable from saved notebook outputs | Not recoverable from saved notebook outputs | Implemented as the main strong branch and used as the anchor model for fusion |
| TextRNN | Not recoverable from saved notebook outputs | Not recoverable from saved notebook outputs | Notebook commentary says this branch is "pretty useless" on RAVDESS because transcript content carries almost no emotion signal |
| Late Fusion | Not recoverable from saved notebook outputs | Not recoverable from saved notebook outputs | Fusion is designed to learn a scalar weight that should heavily favor the audio branch |

### Qualitative Interpretation

- The audio model is the intended primary performer.
- The text model is expected to be near chance because both spoken sentences are largely emotion-independent.
- The fused model is expected to track the audio model closely, because the learned fusion weight should assign most of the mass to the audio branch.

## 8. Training and Validation Loss Plots

The notebook defines a plotting utility called `plot_training_histories(...)` and records loss histories in:

- `audio_history`
- `text_history`

However, the committed notebook snapshot does not store the populated loss arrays or the rendered plot outputs for the training cells. As a result, the exact loss curves cannot be reconstructed from the repository contents alone.

### Intended Plot Coverage

Once the notebook is rerun with outputs preserved, the report should include:

1. Audio model training loss vs. validation loss per epoch
2. Text model training loss vs. validation loss per epoch

### Expected Behavior of the Curves

- `EmotionCNN (Audio)` should show meaningful convergence with early stopping based on validation loss.
- `TextRNN` is likely to overfit quickly or plateau because the transcript signal is weak.

## 9. Reproducibility Notes

### Key hyperparameters

- `BATCH_SIZE = 32`
- Audio optimizer: AdamW with warmup cosine decay, weight decay `1e-4`
- Text optimizer: AdamW with warmup cosine decay, weight decay `1e-4`
- Audio early stopping patience: `50` for initial split, `30` in LOSO
- Text early stopping patience: `20` for initial split, `10` in LOSO
- Fusion optimizer: Adam with learning rate `1e-2`

### Notebook execution metadata

The notebook metadata indicates a successful Kaggle run on `2026-05-10`, but the saved cell outputs for the training/evaluation sections were stripped before this repository snapshot was committed.

## 10. Conclusion

The notebook implements a sensible multimodal pipeline for RAVDESS, but its own design notes make the central finding clear: this dataset is dominated by acoustic emotion cues, not lexical content. The architecture reflects that reality well:

- a regularized CNN is used as the main speech model
- a lightweight GRU tests the text modality
- a scalar late-fusion layer prevents the weak text branch from hurting performance too much

From a methodological standpoint, the strongest decisions in the notebook are the use of LOSO evaluation, class-weighted training, and a conservative late-fusion strategy.

## 11. Finalization Checklist

To turn this into a fully numeric final report, rerun the notebook and preserve outputs from these sections:

1. Cell `21`: audio training
2. Cell `23`: text training
3. Cell `24`: initial test-set evaluation and loss plots
4. Cell `26`: LOSO audio/text evaluation
5. Cell `28`: fusion training and learned `alpha`
6. Cell `29`: fused LOSO evaluation and aggregate metrics

Once those outputs are available, the placeholders in Sections 7 and 8 can be replaced with:

- aggregate LOSO accuracy for all three models
- aggregate weighted F1-score for all three models
- the rendered loss plots from `plot_training_histories(...)`
