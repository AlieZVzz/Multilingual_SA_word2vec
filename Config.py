class Config(object):
    train_path = 'Dataset/eng_train.txt'
    validation_path = 'Dataset/eng_validation.txt'
    test_path = 'Dataset/eng_validation.txt'
    epoch = 20

    embed_size = 100
    num_hiddens = 100
    batch_size = 64
    momentum = 0.9
    num_layers = 2
    pad_length = 500

    lr = 1e-3

    bidirectional = True
    if bidirectional:
        num_direction = 2
    filter_num = 64
    filter_sizes = '3,4,5'
    dropout = 0.2