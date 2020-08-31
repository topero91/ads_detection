import argparse
import numpy as np
import pandas as pd
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler


MODEL_DIR = 'models/'
TOKENIZER_MAX_LENGTH = 64
BATCH_SIZE = 32


def parse_args():
    parser = argparse.ArgumentParser(
        description='''Test task solution.
        For the program to work correctly, you need to enter a file with with one sentence per line. 
        The output is another file with one predicted label per line.''')
    parser.add_argument('--input_file', type=str, help='path to input file in format csv')
    parser.add_argument('--output_file', type=str, help='path to output file')
    args = parser.parse_args()
    return args


def define_device():
    if torch.cuda.is_available():       
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def load_model(model_dir, device):
    model = BertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model.to(device)
    return model, tokenizer


def tokenize_sentences(sentences, max_length, tokenizer):
    input_ids = []
    attention_masks = []

    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = max_length,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            truncation = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )
        
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])
    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return input_ids, attention_masks


def create_dataloader(input_ids, attention_masks):
    batch_size = BATCH_SIZE  
    prediction_data = TensorDataset(input_ids, attention_masks)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
    return prediction_dataloader


def calculate_predictions(prediction_dataloader, model, device):
    # Put model in evaluation mode
    model.eval()

    # Tracking variables 
    predictions = []

    # Predict 
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        
        b_input_ids, b_input_mask = batch
        
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, 
                            attention_mask=b_input_mask)

        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        
        # Store predictions and true labels
        predictions.append(logits)
    flat_predictions = np.concatenate(predictions, axis=0)
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    return flat_predictions


def main():
    args = parse_args()
    device = define_device()
    model_dir = MODEL_DIR
    tokenizer_max_length = TOKENIZER_MAX_LENGTH
    model, tokenizer = load_model(model_dir, device)
    df = pd.read_csv(args.input_file, names = ['sentences'])
    sentences = df.sentences.values
    input_ids, attention_masks = tokenize_sentences(sentences, tokenizer_max_length, tokenizer)
    prediction_dataloader = create_dataloader(input_ids, attention_masks)
    predictions = calculate_predictions(prediction_dataloader, model, device)
    df_predictions = pd.DataFrame(predictions)
    df_predictions.to_csv(args.output_file, index = False)
    

if __name__ == "__main__":
    main()