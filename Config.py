class Config(object):
    train_path = 'Dataset/eng_train.txt'
    validation_path = 'Dataset/eng_validation.txt'
    test_path = 'Dataset/baidu_zh_token.txt'
    epoch = 20

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

