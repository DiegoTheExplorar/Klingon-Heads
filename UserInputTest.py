import torch
from DataPreprocessing import preprocess
def translate_sentence(model, src_sentence, src_vocab, trg_vocab, device, max_len=50):
    model.eval()
    
    with torch.no_grad():
        tokenized_src_sentence = [src_vocab[token] for token in src_sentence.split()]
        tokenized_src_sentence = [src_vocab['<sos>']] + tokenized_src_sentence + [src_vocab['<eos>']]
        src_tensor = torch.LongTensor(tokenized_src_sentence).unsqueeze(1).to(device)
        
        encoder_outputs = model.encoder(src_tensor)
        
        trg_indexes = [trg_vocab['<sos>']]
        for i in range(max_len):
            trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
            output, hidden = model.decoder(trg_tensor, encoder_outputs)
            pred_token = output.argmax(1).item()
            trg_indexes.append(pred_token)
            if pred_token == trg_vocab['<eos>']:
                break
        
    translated_sentence = [trg_vocab.itos[i] for i in trg_indexes]
    return translated_sentence[1:]  # Remove <sos> token

# Example usage
def main():
    # Load model and vocabularies
    model = torch.load('seq2seq_model.pth')  # Load your trained model
    model.eval()
    
    english_tokenizer, klingon_tokenizer = preprocess()[:2]
    src_vocab = english_tokenizer.word_index
    src_vocab['<sos>'] = len(src_vocab) + 1
    src_vocab['<eos>'] = len(src_vocab) + 1
    trg_vocab = klingon_tokenizer.word_index
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test sentences
    src_sentences = ["i like rice"]

    for src_sentence in src_sentences:
        translated_sentence = translate_sentence(model, src_sentence, src_vocab, trg_vocab, device)
        translated_sentence = ' '.join(translated_sentence)
        print(f"English: {src_sentence}")
        print(f"Klingon: {translated_sentence}\n")

if __name__ == "__main__":
    main()
