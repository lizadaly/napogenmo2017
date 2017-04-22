import logging
import random

import torch
import torch.nn as nn
from torch.autograd import Variable

import util

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()



def random_training_set(chunks, revmap, volatile=False, start=0, batch_size=util.BATCH_SIZE):
    """Generate a training set from the randomized set of chunks (sequences),
    returning them in descending order by size (per Torch's variable length
    sequence RNN support).

    If `volatile` is True, the variables will not be added to the computational graph
    (as during generation/evaluation).
    """
    end = start + batch_size
    data = []
    seq_lengths = []

    # Select a batch and pre-sort them based on sequence length, descending
    batch = chunks[start:end]
    batch.sort(key=len)
    batch.reverse()
    for chunk in batch:
        # Save the length of the sequence, the sequence itself, and the target--
        # the predictions for the char rnn, which is the sequence offset by one
        seq_lengths.append(len(chunk[:-1]))
        inp = util.doc_to_seq(chunk[:-1], revmap)
        target = util.doc_to_seq(chunk[1:], revmap)
        data.append((inp, target))

    inp = torch.LongTensor([i for i, t in data])  # (BATCH_SIZE, SEQUENCE_LENGTH)
    tar = torch.LongTensor([t for i, t in data])
    inputs = Variable(inp, volatile=volatile, requires_grad=False).cuda()
    targets = Variable(tar, volatile=volatile, requires_grad=False).cuda()
    return inputs, targets, seq_lengths


def main():
    """Run the model training and evaluation"""
    print_every = 10  # How often to output status info during training

    # Load the data and map each character in the set. Keep both forward and
    # backward mappings
    words = util.load_data()
    charmap, revmap = util.map_all_chars(words)
    n_characters = len(charmap.keys())
    log.info("Running with a vocab size of %d characters and %d words", n_characters, len(words))

    # Instantiate the RNN and put it on the GPU
    rnn = util.RNN(n_characters, util.HIDDEN_SIZE, n_characters, n_layers=util.NUM_LAYERS)
    rnn.cuda()

    # Set up the loss function and optimizer
    optimizer = torch.optim.Adam(rnn.parameters(), lr=util.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    criterion.cuda()

    loss_avg = 0
    total_count = 0

    # Break the data up into training chunks and compute the number of batches we'll run
    chunks = util.get_chunks(words)
    batches = len(chunks) // util.BATCH_SIZE
    log.info("Got %d chunks from data set, %s batches", len(chunks), batches)

    for epoch in range(0, util.NUM_EPOCHS + 1):
        # Within each epoch, reshuffle the training set
        dataset = random.sample(chunks, len(chunks))  # Non-inplace shuffle
        start = 0

        # For each batch, set up our Torch inputs, targets, and sequence lengths
        for i in range(batches):
            inp, target, seq_lengths = random_training_set(dataset, revmap,
                                                           start=start, volatile=False)

            # Clear the RNN training variables and ensure it's in training mode
            hidden = rnn.init_hidden()
            rnn.train()
            rnn.zero_grad()

            # Run the evaluation and get the last set out of outputs from the LSTM
            output, hidden = rnn(inp, hidden, seq_lengths)
            target = target.view(-1)

            # input has to be a 2D Tensor of size batch x n.
            # target for each value of a 1D tensor of size n, a class index (0 to nClasses-1) as t
            loss = criterion(output, target)
            loss_avg += loss.data[0]  # [ BATCHSIZE x SEQLEN ]

            # Run the loss and then iterate our batch counter
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            start += util.BATCH_SIZE
            total_count += 1

        # Periodically print some generated text to see how the training is going
        if epoch % print_every == 0:
            log.info("epoch=%d, %d%% loss=%.4f", epoch, epoch / util.NUM_EPOCHS * 100, loss_avg / total_count)
            prime_str = "A"  #  Some text to start the network with
            predict_len = 1000  # How many characters we want to generate
            temperature = 0.8  # Higher temperature is less likely outputs, so more variety
            rnn.eval()  # Put the RNN in evaluation mode

            # Prime the sequence
            inp = Variable(torch.LongTensor([util.doc_to_seq(prime_str, revmap)]), volatile=True).cuda()
            hidden = Variable(torch.zeros(rnn.n_layers, 1, rnn.hidden_size).cuda())
            logits, hidden = rnn(inp, hidden, seq_lengths=[len(prime_str)])
            predicted = ""

            # Character by character, generate some text at our current state
            for p in range(predict_len):

                # We're only generating one character in this loop, so fix the sequence length at 1
                logits, hidden = rnn(inp, hidden, seq_lengths=[1])

                # Take the last set of probabilities from the output
                logits = logits[-1, :]
                output_dist = logits.data.div(temperature).exp()

                # Get the most-probable prediction and turn it into a character
                top_i = torch.multinomial(output_dist, 1)[0]
                predicted_char = charmap[top_i]
                predicted += predicted_char
                # Set the input for the next sequence in the loop to our predicted character
                inp = Variable(torch.LongTensor([util.doc_to_seq(predicted_char, revmap)]), volatile=True).cuda()

            # Print the sequence we built up
            print(predicted)

        # Save the model at the end of every batch
        torch.save(rnn, util.MODEL_NAME)



if __name__ == '__main__':
    main()
