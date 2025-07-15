import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='bpe',
    vocab_size=8000,
    character_coverage=1.0,
    model_type='bpe',
    pad_id=0, pad_piece='<pad>',
    unk_id=1, unk_piece='<unk>',
    bos_id=2, bos_piece='<sos>',
    eos_id=3, eos_piece='<eos>'
)
