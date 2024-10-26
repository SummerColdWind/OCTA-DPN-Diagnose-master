config = {
    'device': 'cuda',  # 'cpu' or 'cuda'
    'root': r'data\batch_2\clean',
    'label_dir': None,
    'image_dir': None,
    'label_key': 'label',  # column name of labels
    'val_frac': 0.4,
    'shuffle': True,
    'train_batch_size': 8,
    'val_batch_size': 8,
    'epoch_num': 120,
    "initial_lr": 0.00000001,
    "max_lr": 0.00005,
    "warmup_epochs": 8,
    'weight_decay': 0.0001,
    'data_types': [
        '表层血流.jpg',
        '脉络膜毛细血管.jpg',
        '深层血流.jpg',
        '玻璃体血流.jpg',
        '无血管层血流.jpg',
    ],
    'save': True,
    'save_per_epoches': 20,

}
