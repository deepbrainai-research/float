gdown --id 1rvWuM12cyvNvBQNCLmG4Fr2L1rpjQBF0

mv float.pth checkpoints/
hf download r-f/wav2vec-english-speech-emotion-recognition \
  --local-dir ./checkpoints/wav2vec-english-speech-emotion-recognition \
  --include "*"
hf download facebook/wav2vec2-base-960h \
  --local-dir ./checkpoints/wav2vec2-base-960h \
  --include "*"
