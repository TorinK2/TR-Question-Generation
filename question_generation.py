from models import sbert_model, nlp_spacy, run_qg_model, run_grammar_model
from models import sent2vec_sbert, s2v_factory_bow, question_similarity

import os
import sys
from random import shuffle

import hdbscan
import numpy as np
import umap
from tqdm import tqdm


class QuestionGenerator:
    """
    Reads in an article and processes it
    """

    cluster_size = 10  # minimum size of sentence cluster
    umap_dim = 5  # number of dimensions of umap-dimensionality reduction
    outlier_size = 1  # hdbscan parameter for outliers

    alpha = 0.3
    beta = 1.01

    def __init__(self, file_path, n, verbose=True):
        self.n = n

        if not verbose:
            # block output to standard out
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')

        print(f'Reading {file_path}')

        with open(file_path, 'r') as file_handler:
            lines = file_handler.readlines()

        print('Processing file')
        self.paragraphs, self.sentences = self.process_lines(lines)

        self.paragraph_count = len(self.paragraphs)

        print('Sorting paragraphs')
        self.flesch_kincaid_sort()

        print('Topical clustering')
        self.sent_enc_lg, self.sent_enc_sm, self.umap_encoder = self.build_sentence_encodings()
        self.sent_cluster_model = self.build_sentence_clusters()

        print('Initial question generation')
        self.questions = self.generate_questions()

        print('Grammar evaluation')
        self.grammar_removed = self.check_question_grammar()

        print('Clearing questions')
        self.pronoun_removed = self.clear_pronoun_questions()
        self.ner_removed = self.clear_ner_questions()
        self.other_removed = self.clear_more_questions()

        print('Ranking questions')
        self.ranked_questions = self.rank_questions()

        self.removed_questions = self.pronoun_removed + self.grammar_removed # + self.ner_removed
        shuffle(self.removed_questions)
        self.removed_questions = self.other_removed + self.removed_questions
        self.all_questions = self.ranked_questions + self.removed_questions

        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    def get_sentences(self, text):
        return np.array([str(i) for i in nlp_spacy(text).sents])

    def process_lines(self, lines):
        paragraphs = []
        sentences = []
        for line in lines:
            line = line[:-1]
            sents = self.get_sentences(line)
            if len(sents) <= 1:
                continue
            paragraphs.append(line)
            sentences.append(sents)

        return paragraphs, sentences

    def flesch_kincaid_sort(self):
        docs = [nlp_spacy(p) for p in self.paragraphs]
        word_count = np.array([len(doc) for doc in docs])
        sent_count = np.array([len(list(doc.sents)) for doc in docs])
        syl_count = np.array([sum(tok._.syllables_count for tok in doc if tok._.syllables_count is not None) for doc in docs])
        fk_scores = 206.835 - 1.015 * (word_count / sent_count) - 84.6 * (syl_count / word_count)
        indexes = np.argsort(-fk_scores)
        self.paragraphs = [self.paragraphs[i] for i in indexes]
        self.sentences = [self.sentences[i] for i in indexes]

    def build_sentence_encodings(self):
        encodings = []
        for sents in self.sentences:
            encodings.append(sbert_model.encode(sents, batch_size=32))
        flat_encodings = [enc for par_enc in encodings for enc in par_enc]
        sent_encoder = umap.UMAP(n_neighbors=QuestionGenerator.cluster_size,
                                 n_components=QuestionGenerator.umap_dim,
                                 metric='cosine').fit(flat_encodings)
        return encodings, sent_encoder.embedding_, sent_encoder

    def build_sentence_clusters(self):
        cluster_model = hdbscan.HDBSCAN(min_cluster_size=QuestionGenerator.cluster_size,
                                        min_samples=QuestionGenerator.outlier_size,
                                        metric='euclidean',
                                        prediction_data=True)
        cluster_model.fit(self.sent_enc_sm)
        return cluster_model

    def generate_questions(self):
        questions = []
        q_count = 0
        for sents in tqdm(self.sentences):
            pq = run_qg_model(sents)
            questions.append(pq)
            q_count += len(pq)
            if q_count > 150 * self.n:
                return questions
        return questions

    def check_question_grammar(self):
        grammar_removed = []
        grammar_kept = []
        for pq in self.questions:
            in_list, out_list = [], []
            for q in pq:
                (in_list if run_grammar_model(q) == 1 else out_list).append(q)
            grammar_removed.extend(out_list)
            grammar_kept.append(in_list)

        self.questions = grammar_kept
        return grammar_removed

    def clear_pronoun_questions(self):
        pronoun_removed = []
        pronoun_kept = []
        for pq in self.questions:
            in_list, out_list = [], []
            for q in pq:
                (out_list if 'PRP' in [t.tag_ for t in nlp_spacy(q)] else in_list).append(q)
            pronoun_removed.extend(out_list)
            pronoun_kept.append(in_list)

        self.questions = pronoun_kept
        return pronoun_removed

    def clear_ner_questions(self):
        ner_removed = []
        ner_kept = []
        for pq in self.questions:
            in_list, out_list = [], []
            for q in pq:
                (out_list if len(nlp_spacy(q).ents) == 0 else in_list).append(q)
            ner_removed.extend(out_list)
            ner_kept.append(in_list)

        self.questions = ner_kept
        return ner_removed

    def clear_more_questions(self):
        ner_removed = []
        ner_kept = []
        for pq in self.questions:
            in_list, out_list = [], []
            for q in pq:
                (out_list if
                 '"' in q or
                 q.split(' ')[0].lower() not in ['who', 'what', 'where', 'when', 'why', 'how', 'which']
                 else in_list).append(q)
            ner_removed.extend(out_list)
            ner_kept.append(in_list)

        self.questions = ner_kept
        return ner_removed

    def rank_questions(self):
        questions = [q for pq in self.questions for q in pq]
        sentences = [s for ps in self.sentences for s in ps]
        i1 = np.array(question_similarity(questions, sentences, sent2vec_sbert)).clip(min=0.0)
        i2 = np.array(question_similarity(questions, sentences, s2v_factory_bow(sentences))).clip(min=0.0)
        sim_score = 1 - (QuestionGenerator.alpha * i1 + (1 - QuestionGenerator.alpha) * i2)
        q_embeddings = sbert_model.encode(questions)
        q_sm_embeddings = self.umap_encoder.transform(q_embeddings)
        q_labels, _ = hdbscan.approximate_predict(self.sent_cluster_model, q_sm_embeddings)
        q_labels += 1
        n_labels = max(q_labels) + 1

        indexes = range(len(questions))
        sent_by_class = [
            sorted([s for s in indexes if q_labels[s] == i], key=lambda x: sim_score[x], reverse=True)
            for i in range(n_labels)
        ]
        for indexes in sent_by_class:
            for i, idx in enumerate(indexes):
                sim_score[idx] *= QuestionGenerator.beta ** i

        return [str(i) for i in np.array(questions)[np.argsort(sim_score)]]

    def get_questions(self, n):
        """
        This will always return ranked questions first
        If n is greater than the number of questions generated, we return all our questions
        """
        return self.all_questions[:n]

    def print_questions(self, n):
        for q in self.get_questions(n):
            print(q)


if __name__ == '__main__':
    a = QuestionGenerator('articles/set3/a8.txt', 150, verbose=True)
    a.print_questions(150)
    print(len(a.grammar_removed))
    # questions = []
    # sets = 4
    # articles = 9
    # files = [f'articles/set{s}/a{a}.txt' for s in range(1, sets + 1) for a in range(1, articles + 1)]
    # all_questions = []
    # for i in tqdm(files):
    #     print('-----' * 5)
    #     a = QuestionGenerator(i, 150, verbose=True)
    #     print(f'Generated {len(a.ranked_questions)} total good questions!')
    #     a.print_questions(150)
    #     all_questions.extend(a.get_questions(25))
    #     all_questions.append('---')

    # import csv
    # with open('torin_questions.csv', 'w') as result_file:
    #     wr = csv.writer(result_file)
    #     wr.writerows([[i] for i in all_questions])

