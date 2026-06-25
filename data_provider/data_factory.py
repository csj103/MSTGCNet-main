from data_provider.data_loader import Dataset_ALFA, DatasetFD
from torch.utils.data import DataLoader

data_dict = {
    'ALFA': Dataset_ALFA,
    'FD': DatasetFD
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq
    drop_last = False
    data_set = Data(
        args=args,
        win_size=args.seq_len,
        root_path=args.root_path,
        flag=flag,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
