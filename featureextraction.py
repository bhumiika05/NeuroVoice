import librosa
import numpy as np
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path)
    
    duration = librosa.get_duration(y=y, sr=sr)
    pauses = np.sum(librosa.effects.split(y, top_db=25).shape[0])
    speech_rate = len(y) / duration
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    return {
        "duration": duration,
        "pause_count": pauses,
        "speech_rate": speech_rate,
        "mfcc_mean": np.mean(mfcc)
    }

def extract_text_features(text):
    doc = nlp(text)
    words = [token.text for token in doc if token.is_alpha]
    
    vocab_richness = len(set(words)) / max(len(words), 1)
    avg_sentence_length = np.mean([len(sent) for sent in doc.sents])
    repetition_rate = 1 - vocab_richness
    
    return {
        "vocab_richness": vocab_richness,
        "avg_sentence_length": avg_sentence_length,
        "repetition_rate": repetition_rate
    }
