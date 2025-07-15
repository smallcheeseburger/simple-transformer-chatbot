import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("bpe.model")

PAD_ID = sp.piece_to_id('<pad>')
UNK_ID = sp.piece_to_id('<unk>')
SOS_ID = sp.piece_to_id('<sos>')
EOS_ID = sp.piece_to_id('<eos>')

def tokenizer(text):
    return sp.encode(text, out_type=int)