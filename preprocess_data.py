import numpy as np
from numpy.typing import NDArray
import mne
from sklearn.model_selection import train_test_split
import torch
import torch.utils.data as Data


def get_data(filename: str):
    # 1.加载数据（先eeg后emg）
    data: NDArray = np.load(filename, allow_pickle=True)
    eeg_data = data.item()["eeg"].transpose(0, 2, 1)
    emg_data = data.item()["emg"].transpose(0, 2, 1)
    label = data.item()["label"]
    label = np.array(label, dtype=int)

    # 2.构建事件
    sfreq_eeg = 500
    sfreq_emg = 4000

    # 创建info结构
    info_eeg = mne.create_info(
        ch_names=['Fp1', 'Fp2', 'F7', 'F3', 'Fz',
                  'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7',
                  'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5',
                  'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3',
                  'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10'],
        ch_types=['eeg'] * 32,
        sfreq=sfreq_eeg
    )
    info_emg = mne.create_info(
        # 1-三角前肌，2-肱桡肌，3-指屈肌，4-指总伸肌，5-骨间背肌
        # 1-anterior deltoid, 2-brachioradialis, 3-flexor digitorum, 4-common extensor digitorum,
        # 5-first dorsal interosseus
        ch_names=['EMG1', 'EMG2', 'EMG3', 'EMG4', 'EMG5'],
        ch_types='eeg',
        sfreq=sfreq_emg
    )
    # 创建事件
    events = np.array([[idx, 0, label] for idx, label in enumerate(label)])
    event_id = dict(weight_165=1, weight_330=2, weight_660=4)

    emg_epochs = mne.EpochsArray(emg_data, info_emg, events, 0, event_id)
    eeg_epochs = mne.EpochsArray(eeg_data, info_eeg, events, 0, event_id)

    # 对数据进行降采样
    # emg_downsampled = emg_epochs.copy().resample(sfreq=1000)
    # eeg_downsampled = eeg_epochs.copy().resample(sfreq=250)
    # print(emg_downsampled.get_data().shape)
    # eeg_epochs.filter(0.05, 50)
    return eeg_epochs, emg_epochs, np.array((label+0.1)/2, dtype=np.int64)

# 提供给emg_net的数据，数据格式为（x，5，1，1600）


def get_data_cnn(filename):
    eeg_epochs, emg_epochs, label = get_data(filename)
    emg = emg_epochs.copy().resample(sfreq=200)
    # emg, _ = mne.time_frequency.psd_multitaper(emg)
    # res_emg = []
    # for term in emg:
    #     res_emg.append([term[:, :800]])
    res_emg = emg.get_data()
    # 一共8s数据，使用后面4秒的数据进行分类
    length = res_emg.shape[-1]
    res_emg = res_emg[:, np.newaxis, :, length//2:]
    emg_data = np.array(res_emg, dtype=np.float32)

    train_emg, test_emg, train_label, test_label = train_test_split(
        emg_data, label, test_size=0.3, random_state=15)

    emg_train_dataset = Data.TensorDataset(
        torch.tensor(train_emg), torch.tensor(train_label))
    emg_test_dataset = Data.TensorDataset(
        torch.tensor(test_emg), torch.tensor(test_label))

    emg_train_loader = Data.DataLoader(
        dataset=emg_train_dataset, batch_size=10, shuffle=True)
    emg_test_loader = Data.DataLoader(
        dataset=emg_test_dataset, batch_size=10, shuffle=True)
    return emg_train_loader, emg_test_loader


def get_data_mmtm(filename):
    eeg_epochs, emg_epochs, label = get_data(filename)
    # emg = emg_epochs.copy().resample(sfreq=200)  ##emgnet（x，5，1600）
    emg = emg_epochs.copy().resample(sfreq=500)  # dgcnn（x，5，4000）

    # emg, _ = mne.time_frequency.psd_multitaper(emg)
    # res_emg = []
    # for term in emg:
    #     res_emg.append([term[:, :800]])
    res_emg = emg.get_data()
    res_emg = res_emg[:, np.newaxis, :, :]
    emg_data = np.array(res_emg, dtype=np.float32)
    eeg_data = eeg_epochs.get_data()

    train_eeg, test_eeg, train_emg, test_emg, train_label, test_label = train_test_split(
        eeg_data, emg_data, label, test_size=0.3, random_state=15)

    mmtm_train_dataset = Data.TensorDataset(
        torch.tensor(train_eeg), torch.tensor(train_emg), torch.tensor(train_label))
    mmtm_test_dataset = Data.TensorDataset(
        torch.tensor(test_eeg), torch.tensor(test_emg), torch.tensor(test_label))

    mmtm_train_loader = Data.DataLoader(
        dataset=mmtm_train_dataset, batch_size=10, shuffle=True)
    mmtm_test_loader = Data.DataLoader(
        dataset=mmtm_test_dataset, batch_size=10, shuffle=True)
    return mmtm_train_loader, mmtm_test_loader


def get_data_split(filename, select_label):  # label[0,1,2]
    eeg_epochs, emg_epochs, label = get_data(filename)
    # emg = emg_epochs.copy().resample(sfreq=200)  ##emgnet（x，5，1600）
    emg = emg_epochs.copy().resample(sfreq=500)  # dgcnn（x，5，4000）
    # emg, _ = mne.time_frequency.psd_multitaper(emg)
    # res_emg = []
    # for term in emg:
    #     res_emg.append([term[:, :800]])
    res_emg = emg.get_data()
    res_emg = res_emg[:, :, :]
    emg_data = np.array(res_emg, dtype=np.float32)
    eeg_data = eeg_epochs.get_data()

    index = np.where(label == select_label)
    eeg, emg = eeg_data[index], emg_data[index]
    rest_eeg, action_eeg = eeg[:, :, :2000], eeg[:, :, 2000:]
    rest_emg, action_emg = emg[:, :, :2000], emg[:, :, 2000:]
    all_eeg = np.append(rest_eeg, action_eeg, axis=0)
    all_emg = np.append(rest_emg, action_emg, axis=0)
    # rest=0,action=1
    data_size = len(eeg)
    all_label = [0 for i in range(data_size)]
    for i in range(data_size):
        all_label.append(1)
    all_label = np.array(all_label, dtype=np.int64)

    train_eeg, test_eeg, train_emg, test_emg, train_label, test_label = train_test_split(
        all_eeg, all_emg, all_label, test_size=0.3, random_state=15)

    mmtm_train_dataset = Data.TensorDataset(
        torch.tensor(train_eeg), torch.tensor(train_emg), torch.tensor(train_label))
    mmtm_test_dataset = Data.TensorDataset(
        torch.tensor(test_eeg), torch.tensor(test_emg), torch.tensor(test_label))

    mmtm_train_loader = Data.DataLoader(
        dataset=mmtm_train_dataset, batch_size=10, shuffle=True)
    mmtm_test_loader = Data.DataLoader(
        dataset=mmtm_test_dataset, batch_size=10, shuffle=True)
    return mmtm_train_loader, mmtm_test_loader

# 将每个样本根据时间划分为split_num段，这里（split_num=4）,一个样本8秒数据，每段2秒


def get_data_gcn_lstm(filename, split_num=4):
    eeg_epochs, emg_epochs, label = get_data(filename)
    # 对数据进行0.05~50的滤波
    eeg_epochs.filter(0.05, h_freq=50)
    emg = emg_epochs.copy().resample(sfreq=500)  # dgcnn（x，5，4000）

    res_emg = emg.get_data()
    # res_emg = res_emg[:, np.newaxis, :, :]
    emg_data = np.array(res_emg, dtype=np.float32)
    eeg_data = eeg_epochs.get_data()

    # emg_data.shape=eeg_data.shape=(x,node,4000)
    length = emg_data.shape[2]//split_num
    emg_temp = np.empty((170, 5, 0, 1000))
    eeg_temp = np.empty((170, 32, 0, 1000))
    emg_data = emg_data[:, :, np.newaxis, :]
    eeg_data = eeg_data[:, :, np.newaxis, :]
    print(length)
    print(emg_data.shape)
    for i in range(split_num):
        emg_temp = np.concatenate(
            (emg_temp, emg_data[:, :, :, i*length:(i+1)*length]), axis=2)
        eeg_temp = np.concatenate(
            (eeg_temp, eeg_data[:, :, :, i*length:(i+1)*length]), axis=2)

    eeg_data = eeg_temp
    emg_data = emg_temp

    train_eeg, test_eeg, train_emg, test_emg, train_label, test_label = train_test_split(
        eeg_data, emg_data, label, test_size=0.3, random_state=15)

    lstm_train_dataset = Data.TensorDataset(
        torch.tensor(train_eeg), torch.tensor(train_emg), torch.tensor(train_label))
    lstm_test_dataset = Data.TensorDataset(
        torch.tensor(test_eeg), torch.tensor(test_emg), torch.tensor(test_label))

    lstm_train_loader = Data.DataLoader(
        dataset=lstm_train_dataset, batch_size=10, shuffle=True)
    lstm_test_loader = Data.DataLoader(
        dataset=lstm_test_dataset, batch_size=10, shuffle=True)
    return lstm_train_loader, lstm_test_loader


def get_data_lhy():
    split_num = 5
    eeg_data_train = np.load("data/lhy_data/EEG_EMG_train_data.npy")
    eeg_data_test = np.load("data/lhy_data/EEG_EMG_test_data.npy")
    label_train = np.load("data/lhy_data/EEG_EMG_train_labels.npy")
    label_test = np.load("data/lhy_data/EEG_EMG_test_labels.npy")

    length = eeg_data_train.shape[2]//split_num
    train_shape = eeg_data_train.shape
    test_shape = eeg_data_test.shape
    temp_train = np.empty((train_shape[0], train_shape[1], 0, length))
    temp_test = np.empty((test_shape[0], test_shape[1], 0, length))
    eeg_data_train = eeg_data_train[:, :, np.newaxis, :]
    eeg_data_test = eeg_data_test[:, :, np.newaxis, :]

    print(length)

    for i in range(split_num):
        temp_train = np.concatenate(
            (temp_train, eeg_data_train[:, :, :, i*length:(i+1)*length]), axis=2)
        temp_test = np.concatenate(
            (temp_test, eeg_data_test[:, :, :, i*length:(i+1)*length]), axis=2)

    eeg_train = temp_train
    eeg_test = temp_test

    lstm_train_dataset = Data.TensorDataset(
        torch.tensor(eeg_train), torch.tensor(label_train))
    lstm_test_dataset = Data.TensorDataset(
        torch.tensor(eeg_test), torch.tensor(label_test))

    lstm_train_loader = Data.DataLoader(
        dataset=lstm_train_dataset, batch_size=10, shuffle=True)
    lstm_test_loader = Data.DataLoader(
        dataset=lstm_test_dataset, batch_size=10, shuffle=True)
    return lstm_train_loader, lstm_test_loader

# 取出标签为0，2的数据


def getdata_GCN_LSTM_0_2(filename, split_num=4):
    label = []
    for i in range(1, 13):
        filename = "data/ws_subj{:}.npy".format(i)
        eeg_epochs, emg_epochs, label_ = get_data(filename)
        # 对数据进行0.05~50的滤波
        eeg_epochs.filter(0.05, h_freq=50)
        emg = emg_epochs.copy().resample(sfreq=500)  # dgcnn（x，5，4000）

        res_emg = emg.get_data()
        # res_emg = res_emg[:, np.newaxis, :, :]
        if i == 1:
            emg_data = np.array(res_emg, dtype=np.float32)
            eeg_data = eeg_epochs.get_data()
            label = label_
        else:
            emg_data = np.concatenate(
                (emg_data, np.array(res_emg, dtype=np.float32)), axis=0)
            eeg_data = np.concatenate(
                (eeg_data, eeg_epochs.get_data()), axis=0)
            label = np.concatenate((label, label_), axis=0)

    index0 = np.where(label == 0)[0]
    index2 = np.where(label == 2)[0]
    index0_2 = np.concatenate((index0, index2))
    label = label[index0_2]//2
    eeg_data = eeg_data[index0_2]
    # emg_data.shape=eeg_data.shape=(x,node,4000)
    length = emg_data.shape[2]//split_num
    emg_temp = np.empty((eeg_data.shape[0], 5, 0, 1000))
    eeg_temp = np.empty((eeg_data.shape[0], 32, 0, 1000))
    emg_data = emg_data[:eeg_data.shape[0], :, np.newaxis, :]
    eeg_data = eeg_data[:, :, np.newaxis, :]
    print(length)
    print(emg_data.shape)
    print(emg_temp.shape)
    print(eeg_data.shape)
    for i in range(split_num):
        emg_temp = np.concatenate(
            (emg_temp, emg_data[:, :, :, i*length:(i+1)*length]), axis=2)
        eeg_temp = np.concatenate(
            (eeg_temp, eeg_data[:, :, :, i*length:(i+1)*length]), axis=2)

    eeg_data = eeg_temp
    emg_data = emg_temp[:eeg_data.shape[0]]

    train_eeg, test_eeg, train_emg, test_emg, train_label, test_label = train_test_split(
        eeg_data, emg_data, label, test_size=0.3, random_state=15)

    lstm_train_dataset = Data.TensorDataset(
        torch.tensor(train_eeg), torch.tensor(train_emg), torch.tensor(train_label))
    lstm_test_dataset = Data.TensorDataset(
        torch.tensor(test_eeg), torch.tensor(test_emg), torch.tensor(test_label))

    lstm_train_loader = Data.DataLoader(
        dataset=lstm_train_dataset, batch_size=10, shuffle=True)
    lstm_test_loader = Data.DataLoader(
        dataset=lstm_test_dataset, batch_size=10, shuffle=True)
    return lstm_train_loader, lstm_test_loader


# 用于GCN_LSTM实现EEG运动意图检测的数据，取前4s的脑电数据(x,32,2000)，然后前两秒为休息0，后两秒为运动1
def get_data_check_intend(filename: str):
    eeg_epochs, _, label = get_data(filename)
    
    # 对数据进行0.05~50的滤波
    eeg_epochs.filter(0.05, h_freq=50)
    eeg_data = eeg_epochs.get_data()
    eeg_data = eeg_data[:, :, :2000]
    
    # 将数据拆分开
    eeg = np.concatenate((eeg_data[:, :, :1000], eeg_data[:, :, 1000:]), axis=0)
    length = eeg.shape[2] // 4
    eeg_shape = eeg.shape
    eeg_temp = np.empty((eeg_shape[0], eeg_shape[1], 0, length))

    eeg_data = eeg[:, :, np.newaxis, :]

    for i in range(4):
        eeg_temp = np.concatenate((eeg_temp, eeg_data[:, :, :, i*length:(i+1)*length]), axis=2)
    eeg = eeg_temp
    
    trial_num = eeg.shape[0]
    label = np.ones((trial_num))
    label[:trial_num//2] = 0
    
    # 划分训练集和测试集
    train_eeg, test_eeg, train_label, test_label = train_test_split(eeg,  label, test_size=0.3, random_state=15)
    # 将数据放进迭代器中
    lstm_train_dataset = Data.TensorDataset(torch.tensor(train_eeg), torch.tensor(train_label))
    lstm_test_dataset = Data.TensorDataset(torch.tensor(test_eeg), torch.tensor(test_label))
    
    lstm_train_loader = Data.DataLoader(dataset=lstm_train_dataset, batch_size=10, shuffle=True)
    lstm_test_loader = Data.DataLoader(dataset=lstm_test_dataset, batch_size=10, shuffle=True)
    return lstm_train_loader, lstm_test_loader


def get_data_check_intend_eegnet(filename):
    eeg_epochs, emg_epochs, label = get_data(filename)
    # 对数据进行0.05~50的滤波
    eeg_epochs.filter(0.05, h_freq=50)
    eeg_data = eeg_epochs.get_data()
    eeg_data = eeg_data[:, :, :2000]
    # 将数据拆分开
    eeg = np.concatenate(
        (eeg_data[:, :, :1000], eeg_data[:, :, 1000:]), axis=0)

    print(eeg.shape)
    trial_num = eeg.shape[0]
    label = np.ones((trial_num))
    label[:trial_num//2] = 0
    print(label.shape)
    # 划分训练集和测试集
    train_eeg, test_eeg, train_label, test_label = train_test_split(
        eeg,  label, test_size=0.3, random_state=15)
    # 将数据放进迭代器中
    lstm_train_dataset = Data.TensorDataset(
        torch.tensor(train_eeg), torch.tensor(train_label))
    lstm_test_dataset = Data.TensorDataset(
        torch.tensor(test_eeg), torch.tensor(test_label))
    #
    lstm_train_loader = Data.DataLoader(
        dataset=lstm_train_dataset, batch_size=10, shuffle=True)
    lstm_test_loader = Data.DataLoader(
        dataset=lstm_test_dataset, batch_size=10, shuffle=True)
    return lstm_train_loader, lstm_test_loader
