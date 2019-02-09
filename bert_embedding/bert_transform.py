# This is file is a temporary solution. It can be removed once gluonnlp release 0.5.1

import unicodedata

import numpy as np


class BERTBasicTokenizer():
    r"""Runs basic tokenization
    performs invalid character removal (e.g. control chars) and whitespace.
    tokenize CJK chars.
    splits punctuation on a piece of text.
    strips accents and convert to lower case.(If lower is true)
    Parameters
    ----------
    lower : bool, default True
        whether the text strips accents and convert to lower case.
    Examples
    --------
    >>> tokenizer = gluonnlp.data.BERTBasicTokenizer(lower=True)
    >>> tokenizer(u" \tHeLLo!how  \n Are yoU?  ")
    ['hello', '!', 'how', 'are', 'you', '?']
    >>> tokenizer = gluonnlp.data.BERTBasicTokenizer(lower=False)
    >>> tokenizer(u" \tHeLLo!how  \n Are yoU?  ")
    ['HeLLo', '!', 'how', 'Are', 'yoU', '?']
    """

    def __init__(self, lower=True):
        self.lower = lower

    def __call__(self, sample):
        """
        Parameters
        ----------
        sample:  str (unicode for Python 2)
            The string to tokenize. Must be unicode.
        Returns
        -------
        ret : list of strs
            List of tokens
        """
        return self._tokenize(sample)

    def _tokenize(self, text):
        """Tokenizes a piece of text."""
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)
        orig_tokens = self._whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.lower:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = self._whitespace_tokenize(' '.join(split_tokens))
        return output_tokens

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp in (0, 0xfffd) or self._is_control(char):
                continue
            if self._is_whitespace(char):
                output.append(' ')
            else:
                output.append(char)
        return ''.join(output)

    def _is_control(self, char):
        """Checks whether `chars` is a control character."""
        # These are technically control characters but we count them as whitespace
        # characters.
        if char in ['\t', '\n', '\r']:
            return False
        cat = unicodedata.category(char)
        if cat.startswith('C'):
            return True
        return False

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(' ')
                output.append(char)
                output.append(' ')
            else:
                output.append(char)
        return ''.join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF)
                or (cp >= 0x3400 and cp <= 0x4DBF)
                or (cp >= 0x20000 and cp <= 0x2A6DF)
                or (cp >= 0x2A700 and cp <= 0x2B73F)
                or (cp >= 0x2B740 and cp <= 0x2B81F)
                or (cp >= 0x2B820 and cp <= 0x2CEAF)
                or (cp >= 0xF900 and cp <= 0xFAFF)
                or (cp >= 0x2F800 and cp <= 0x2FA1F)):
            return True

        return False

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize('NFD', text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == 'Mn':
                continue
            output.append(char)
        return ''.join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if self._is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return [''.join(x) for x in output]

    def _is_punctuation(self, char):
        """Checks whether `chars` is a punctuation character."""
        cp = ord(char)
        # We treat all non-letter/number ASCII as punctuation.
        # Characters such as "^", "$", and "`" are not in the Unicode
        # Punctuation class but we treat them as punctuation anyways, for
        # consistency.
        group0 = cp >= 33 and cp <= 47
        group1 = cp >= 58 and cp <= 64
        group2 = cp >= 91 and cp <= 96
        group3 = cp >= 123 and cp <= 126
        if (group0 or group1 or group2 or group3):
            return True
        cat = unicodedata.category(char)
        if cat.startswith('P'):
            return True
        return False

    def _is_whitespace(self, char):
        """Checks whether `chars` is a whitespace character."""
        # \t, \n, and \r are technically contorl characters but we treat them
        # as whitespace since they are generally considered as such.
        if char in [' ', '\t', '\n', '\r']:
            return True
        cat = unicodedata.category(char)
        if cat == 'Zs':
            return True
        return False

    def _whitespace_tokenize(self, text):
        """Runs basic whitespace cleaning and splitting on a piece of text."""
        text = text.strip()
        tokens = text.split()
        return tokens


class BERTTokenizer(object):
    r"""End-to-end tokenization for BERT models.
    Parameters
    ----------
    vocab : gluonnlp.Vocab or None, default None
        Vocabulary for the corpus.
    lower : bool, default True
        whether the text strips accents and convert to lower case.
        If you use the BERT pre-training model,
        lower is set to Flase when using the cased model,
        otherwise it is set to True.
    max_input_chars_per_word : int, default 200
    Examples
    --------
    >>> _,vocab = gluonnlp.model.bert_12_768_12(dataset_name='wiki_multilingual',pretrained=False)
    >>> tokenizer = gluonnlp.data.BERTTokenizer(vocab=vocab)
    >>> tokenizer (u"gluonnlp: 使 NLP 变得简单。")
    ['gl', '##uo', '##nn', '##lp', ':', ' 使 ', 'nl', '##p', ' 变 ', ' 得 ', ' 简 ', ' 单 ', '。']
    """

    def __init__(self, vocab, lower=True, max_input_chars_per_word=200):
        self.vocab = vocab
        self.max_input_chars_per_word = max_input_chars_per_word
        self.basic_tokenizer = BERTBasicTokenizer(lower=lower)

    def __call__(self, sample):
        """
        Parameters
        ----------
        sample: str (unicode for Python 2)
            The string to tokenize. Must be unicode.
        Returns
        -------
        ret : list of strs
            List of tokens
        """

        return self._tokenizer(sample)

    def _tokenizer(self, text):
        split_tokens = []
        for token in self.basic_tokenizer(text):
            for sub_token in self._tokenize_wordpiece(token):
                split_tokens.append(sub_token)

        return split_tokens

    def _tokenize_wordpiece(self, text):
        """Tokenizes a piece of text into its word pieces.
        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.
        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]
        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BERTBasicTokenizer.
        Returns:
          A list of wordpiece tokens.
        """

        output_tokens = []
        for token in self.basic_tokenizer._whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.vocab.unknown_token)
                continue
            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = ''.join(chars[start:end])
                    if start > 0:
                        substr = '##' + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end
            if is_bad:
                output_tokens.append(self.vocab.unknown_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        return self.vocab.to_indices(tokens)


class BERTSentenceTransform(object):
    r"""BERT style data transformation.
    Parameters
    ----------
    tokenizer : BERTTokenizer.
        Tokenizer for the sentences.
    max_seq_length : int.
        Maximum sequence length of the sentences.
    pad : bool, default True
        Whether to pad the sentences to maximum length.
    pair : bool, default True
        Whether to transform sentences or sentence pairs.
    """

    def __init__(self, tokenizer, max_seq_length, pad=True, pair=True):
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
        self._pad = pad
        self._pair = pair

    def __call__(self, line):
        """Perform transformation for sequence pairs or single sequences.
        The transformation is processed in the following steps:
        - tokenize the input sequences
        - insert [CLS], [SEP] as necessary
        - generate type ids to indicate whether a token belongs to the first
        sequence or the second sequence.
        - generate valid length
        For sequence pairs, the input is a tuple of 2 strings:
        text_a, text_b.
        Inputs:
            text_a: 'is this jacksonville ?'
            text_b: 'no it is not'
        Tokenization:
            text_a: 'is this jack ##son ##ville ?'
            text_b: 'no it is not .'
        Processed:
            tokens: '[CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]'
            type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
            valid_length: 14
        For single sequences, the input is a tuple of single string:
        text_a.
        Inputs:
            text_a: 'the dog is hairy .'
        Tokenization:
            text_a: 'the dog is hairy .'
        Processed:
            text_a: '[CLS] the dog is hairy . [SEP]'
            type_ids: 0     0   0   0  0     0 0
            valid_length: 7
        Parameters
        ----------
        line: tuple of str
            Input strings. For sequence pairs, the input is a tuple of 2 strings:
            (text_a, text_b). For single sequences, the input is a tuple of single
            string: (text_a,).
        Returns
        -------
        np.array: input token ids in 'int32', shape (batch_size, seq_length)
        np.array: valid length in 'int32', shape (batch_size,)
        np.array: input token type ids in 'int32', shape (batch_size, seq_length)
        """

        # convert to unicode
        text_a = line[0]
        if self._pair:
            assert len(line) == 2
            text_b = line[1]

        tokens_a = self._tokenizer(text_a)
        tokens_b = None

        if self._pair:
            tokens_b = self._tokenizer(text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            self._truncate_seq_pair(tokens_a, tokens_b, self._max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > self._max_seq_length - 2:
                tokens_a = tokens_a[0:(self._max_seq_length - 2)]

        # The embedding vectors for `type=0` and `type=1` were learned during
        # pre-training and are added to the wordpiece embedding vector
        # (and position vector). This is not *strictly* necessary since
        # the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.

        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        vocab = self._tokenizer.vocab
        tokens = []
        tokens.append(vocab.cls_token)
        tokens.extend(tokens_a)
        tokens.append(vocab.sep_token)
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens.extend(tokens_b)
            tokens.append(vocab.sep_token)
            segment_ids.extend([1] * (len(tokens) - len(segment_ids)))

        input_ids = self._tokenizer.convert_tokens_to_ids(tokens)

        # The valid length of sentences. Only real  tokens are attended to.
        valid_length = len(input_ids)

        if self._pad:
            # Zero-pad up to the sequence length.
            padding_length = self._max_seq_length - valid_length
            # use padding tokens for the rest
            input_ids.extend([vocab[vocab.padding_token]] * padding_length)
            segment_ids.extend([0] * padding_length)

        return np.array(input_ids, dtype='int32'), np.array(valid_length, dtype='int32'), \
               np.array(segment_ids, dtype='int32')

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
