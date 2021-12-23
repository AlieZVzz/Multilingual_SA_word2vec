class Config(object):
    train_path = 'Dataset/zh_train_data.txt'
    validation_path = 'Dataset/zh_validation_data.txt'
    test_path = 'Dataset/zh_train_data.txt'
    epoch = 10

    embedding_dim = 300
    hidden_dim = 100
    batch_size = 64
    momentum = 0.9

    lr = 1e-3

    layer_size = 2

    bidirectional = True
    if bidirectional:
        num_direction = 2
    else:
        num_direction = 1

    sequence_length = 60  # 句子长度
    attention_size = 60
    filter_num = 64
    filter_sizes = '3,4,5'
    dropout = 0.2

