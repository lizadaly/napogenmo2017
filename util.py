import logging
import os
import string

from lxml import etree
import torch
import torch.nn as nn
from torch.autograd import Variable

log = logging.getLogger(__name__)

# create a file handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

punctuation = string.punctuation + '’”…—“'

formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
log.addHandler(handler)
log.setLevel(logging.INFO)

DATA_DIR = '/home/liza/data/field-guides/unpacked'
SEQUENCE_LENGTH = 128  # Characters per set of examples
BATCH_SIZE = 128
HIDDEN_SIZE = 512
NUM_LAYERS = 2
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
DROPOUT = 0.5
MODEL_NAME = 'rnn-model'

def load_data(data_dir=DATA_DIR):
    """Quickly parse a set of XHTML files (derived from an EPUB) and toss out any paragraphs
       that seem to be Gutenboilerplate."""
    words = []

    # Compiling the XPath expression up front is a _huge_ time savings, as is checking
    # for the blacklist text in the XPath expression rather than in evaluating the final strings
    xp1 = etree.XPath("descendant::x:p[not(contains(text(), 'gutenberg'))]//text()",
                      namespaces={'x': 'http://www.w3.org/1999/xhtml'})
    for f in os.listdir(data_dir):
        doc = etree.parse(os.path.join(data_dir, f))
        para = xp1(doc)
        for p in para:
            words.append(" ".join(p.split()))

    # Return a single very long string of all the words in all the documents concatenated together
    words = ' '.join(words)
    return words


def get_chunks(words, chunk_length=SEQUENCE_LENGTH):
    """Generate a list of chunks of max len SEQUENCE_LENGTH, suitable for randomization later"""
    remaining = len(words)
    punct = string.punctuation + ' '
    start_index = 0
    chunks = []
    # Ensure that the chunk breaks don't cross a word boundary, since the loading
    # mechanism we're using doesn't resume the sequence across chunk boundaries
    while remaining > chunk_length:
        current_char = None
        while current_char != ' ':
            # Walk forward until we find a space
            current_char = words[start_index]
            start_index += 1
        current_char = 'a'
        end_index = min(start_index + chunk_length + 1, len(words) - 1)
        while current_char not in punct:
            current_char = words[end_index]
            if end_index == len(words):
                break
            if end_index == start_index:
                break
            end_index -= 1
        chunk = words[start_index:end_index + 1]
        chunks.append(chunk)
        start_index = end_index + 1
        remaining = len(words) - end_index
    return chunks


def map_all_chars(doc):
    """Takes a document (all the words in all the training data, concatenated) and
    computes a stable map of integer values for lookup or translation later"""
    i = 0
    charmap = {}
    revmap = {}
    sorted_doc = sorted(list(set(doc[:])))  # Sort it first so it's stable
    for c in sorted_doc:
        if c not in revmap.keys():
            log.debug("Adding new char %s at index %i", c, i)
            charmap[i] = c  # For the reverse lookup: index to char
            revmap[c] = i
            i += 1

    return charmap, revmap


def pad(tensor, length):
    """Pad a Torch tensor to a given length"""
    return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])


def doc_to_seq(doc, revmap, sequence_length=SEQUENCE_LENGTH):
    """Each character in the input will be turned into a Seq by looking it up in
    the sequence `revmap`"""
    out = []
    for i, c in enumerate(doc):
        out.append(revmap[c])
    out.extend([0 for i in range(sequence_length - len(out))])
    return out


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, n_layers=1, dropout=DROPOUT):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        # Our encoding later (character to index)
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, batch_first=True)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, inp, hidden, seq_lengths):
        # Encode the input into the embedding, and then use Torch's padding sequence
        # function to pack it but pass in the true length of the original
        emb = self.encoder(inp)
        emb = torch.nn.utils.rnn.pack_padded_sequence(emb, seq_lengths, batch_first=True)

        # Machine learning!
        output, hidden = self.gru(emb, hidden)

        # The output will be packed, so turn it back into useful outputs: a tensor
        # of BATCH_SIZE, SEQLEN, HIDDEN_SIZE
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        # ex. torch.Size([30, 50, 512])

        # Now we want: BATCHSIZE * SEQLEN, HIDDEN_SIZE
        # torch.Size([1500, 512])
        output = output.contiguous().view(-1, hidden.size(2))

        # Now it's ready to be returns as the logits layer
        logits = self.decoder(output)
        return logits, hidden

    def init_hidden(self):
        # The hidden state will use BATCH_SIZE in the 1st position even if we hand data as batch_first
        return Variable(torch.zeros(self.n_layers, BATCH_SIZE, self.hidden_size).cuda())
