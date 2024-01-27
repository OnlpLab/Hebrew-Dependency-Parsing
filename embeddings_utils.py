import torch
from data_utils import PAD_TOKEN


def get_words_tensors(tokenizer, input_words, device):
    tokenizer_out = tokenizer(input_words, padding=True, truncation=True, is_split_into_words=True,
                              return_tensors='pt', return_offsets_mapping=True, return_attention_mask=True)
    tok_map = tokenizer_out.offset_mapping  # offsets of original
    tok_map = tok_map.squeeze(0)
    mask = (tok_map[..., 0] == 0) & (tok_map[..., 1] != 0)
    mask = mask.to(device)
    word_tensor = tokenizer_out.input_ids.squeeze(0)
    word_tensor = word_tensor.to(device)
    return word_tensor, mask


def get_new_context(analysis, curr_tokens, token_i):
    curr_analysis_context = []
    start = -1
    end = -1
    for j, t in enumerate(curr_tokens):
        if token_i != j:
            curr_analysis_context.append(t)
        else:
            curr_analysis_context.extend(analysis)
            start = j
            end = j + len(analysis) - 1

    return curr_analysis_context, start, end


def get_contextualized_embeddings(curr_tokens, curr_segments_analyses, tokenizer, bert_model, device):
    # insert 2 PAD tokens to analyses and tokens
    curr_segments_analyses.insert(0, [[PAD_TOKEN]])
    curr_segments_analyses.insert(0, [[PAD_TOKEN]])
    curr_tokens.insert(0, PAD_TOKEN)
    curr_tokens.insert(0, PAD_TOKEN)

    original_tokens_tensor, mask = get_words_tensors(tokenizer, curr_tokens, device)
    original_tokens_embeddings = bert_model(input_ids=original_tokens_tensor.unsqueeze(0)).last_hidden_state.squeeze(0)[mask].detach()

    all_analyses_embedding = []
    for i, token_analyses in enumerate(curr_segments_analyses):
        for analysis in token_analyses:
            if len(analysis) == 1:
                # try:
                all_analyses_embedding.append(original_tokens_embeddings[i])
                # except Exception as error:
                #     print()
            else:
                curr_analysis_context, start, end = get_new_context(analysis, curr_tokens, i)

                word_tensor, mask = get_words_tensors(tokenizer, curr_analysis_context, device)
                current_analysis_token_embeddings = bert_model(input_ids=word_tensor.unsqueeze(0)).last_hidden_state.squeeze(0)[mask].detach()

                for segment_i in range(start, end + 1):
                    all_analyses_embedding.append(current_analysis_token_embeddings[segment_i])

    all_analyses_embedding = torch.stack(all_analyses_embedding)
    return all_analyses_embedding


def linearize_analyses(sentence_analyses):
    linearized_sentence = []

    for token in sentence_analyses.tokens:
        curr_token = []
        for analysis in token.analyses:
            curr_token.append(analysis.analysis)
        linearized_sentence.append(curr_token)

    return linearized_sentence


def get_embeddings(sentence_analyses, tokenizer, bert_model, device):
    all_embeddings = []
    for curr_sent in sentence_analyses:
        curr_tokens = [t.token for t in curr_sent.tokens]
        curr_segments_analyses = linearize_analyses(curr_sent)
        embed = get_contextualized_embeddings(curr_tokens, curr_segments_analyses, tokenizer, bert_model, device)
        all_embeddings.append(embed)

    return all_embeddings
