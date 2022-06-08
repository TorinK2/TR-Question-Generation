import numpy as np
import spacy
from spacy_syllables import SpacySyllables
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import T5ForConditionalGeneration, T5TokenizerFast, BertForSequenceClassification, BertTokenizer

nlp_spacy = spacy.load('en_core_web_lg')
nlp_spacy.add_pipe('syllables', after='tagger')

spacy_tokenizer = nlp_spacy.tokenizer




sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

SEP_TOKEN = '<sep>'
END_TOKEN = '</s>'
DEFAULT_ARGS = dict(
    max_length=256,
    num_beams=4,
    length_penalty=1.5,
    no_repeat_ngram_size=3,
    early_stopping=True
)


class QuestionGenerationModel:

    def __init__(self, **args):
        self.tokenizer = T5TokenizerFast.from_pretrained('qg_tokenizer')
        self.tokenizer.sep_token = SEP_TOKEN
        self.tokenizer.add_tokens([SEP_TOKEN])
        self.model = T5ForConditionalGeneration.from_pretrained('qg_model')
        self.args = args or DEFAULT_ARGS

    def run(self, sentences):
        input_text = f'generate questions: {" ".join(sentences)} {END_TOKEN}'
        input_tokens = self.tokenizer.encode(input_text, return_tensors='pt')
        if input_tokens.size()[1] > 512:
            c = len(sentences) // 2
            return list(set(self.run(sentences[:c]) + self.run(sentences[c + 1:])))
        output_tokens = self.model.generate(input_tokens, **self.args)
        output_text = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
        questions = [question.strip() for question in output_text.split(SEP_TOKEN)]
        questions = [question for question in questions if question.endswith('?')]
        return questions


question_generation_model = QuestionGenerationModel()


def run_qg_model(sentences):
    return question_generation_model.run(sentences)


grammar_tokenizer = BertTokenizer.from_pretrained('grammar_model')
grammar_model = BertForSequenceClassification.from_pretrained('grammar_model')


def run_grammar_model(sentence):
    encoded_dict = grammar_tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        max_length=64, padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    with torch.no_grad():
        output = grammar_model(torch.LongTensor(encoded_dict['input_ids']),
                               attention_mask=torch.LongTensor(encoded_dict['attention_mask']),
                               token_type_ids=None)
    score = np.exp(output.logits[0, 1]) / np.exp(output.logits[0, 0]) + np.exp(output.logits[0, 1])
    return 1 if score >= 0.95 else 0


def tokenize(sentence):
    tokens = []
    for token in spacy_tokenizer(sentence):
        if not token.is_punct:
            tokens.append(token.text.lower())
    return tokens


def s2v_factory_bow(sentences):
    tokens = set()
    for sentence in sentences:
        tokens.update(tokenize(str(sentence)))
    tokens = {t: i for i, t in enumerate(tokens)}
    n = len(tokens)

    def sent2vec_bow(sentences):
        encodings = np.zeros((len(sentences), n))
        for i, sentence in enumerate(sentences):
            for token in tokenize(str(sentence)):
                if token in tokens:
                    encodings[i, tokens[token]] += 1
        return encodings

    return sent2vec_bow


def sent2vec_sbert(sentences):
    return sbert_model.encode(sentences)


def question_similarity(questions, sentences, sent2vec):
    q_embeddings = sent2vec(questions)
    s_embeddings = sent2vec(sentences)
    dist_matrix = util.cos_sim(q_embeddings, s_embeddings)
    importance = torch.max(dist_matrix, dim=1).values
    return importance
