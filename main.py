from concurrent.futures import ProcessPoolExecutor

import cv2
import joblib
import numpy as np
import pywt
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from functools import partial
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
)
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring, Initializer, LRScheduler, TensorBoard
from skorch.dataset import Dataset
from skorch.helper import predefined_split
from torch.backends import cudnn
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

cudnn.benchmark = False
cudnn.deterministic = True

torch.manual_seed(0)


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 7)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.pooling1 = nn.MaxPool2d(5)
        self.pooling2 = nn.MaxPool2d(3)
        self.pooling3 = nn.AdaptiveMaxPool2d((1, 1))
        self.fc1 = nn.Linear(68, 32)
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x1, x2):
        x1 = F.relu(self.bn1(self.conv1(x1)))  # (16 x 94 x 94)
        x1 = self.pooling1(x1)  # (16 x 18 x 18)
        x1 = F.relu(self.bn2(self.conv2(x1)))  # (32 x 16 x 16)
        x1 = self.pooling2(x1)  # (32 x 5 x 5)
        x1 = F.relu(self.bn3(self.conv3(x1)))  # (64 x 3 x 3)
        x1 = self.pooling3(x1)  # (64 x 1 x 1)
        x1 = x1.view((-1, 64))  # (64,)
        x = torch.cat((x1, x2), dim=1)  # (68,)
        x = F.relu(self.fc1(x))  # (32,)
        x = self.fc2(x)  # (4,)
        return x


def worker(data, wavelet, scales, sampling_period):
    # 심장 박동 구간을 분석하기 위한 구간 설정
    before, after = 90, 110

    # CWT 변환
    coeffs, frequencies = pywt.cwt(data["signal"], scales, wavelet, sampling_period)
    r_peaks, categories = data["r_peaks"], data["categories"]

    # 간격을 줄이기 위해 평균 간격을 구한다.
    avg_rri = np.mean(np.diff(r_peaks))

    x1, x2, y, groups = [], [], [], []
    for i in range(len(r_peaks)):
        if i == 0 or i == len(r_peaks) - 1:  # 첫번째 및 마지막 심장 박동은 제외
            continue

        if categories[i] == 4:
            continue

        # coeffs을 이용해 100*100 크기의 이미지로 변환
        x1.append(
            cv2.resize(coeffs[:, r_peaks[i] - before : r_peaks[i] + after], (100, 100))
        )
        x2.append(
            [
                r_peaks[i] - r_peaks[i - 1] - avg_rri,  # previous RR Interval
                r_peaks[i + 1] - r_peaks[i] - avg_rri,  # post RR Interval
                (r_peaks[i] - r_peaks[i - 1])
                / (r_peaks[i + 1] - r_peaks[i]),  # ratio RR Interval
                np.mean(np.diff(r_peaks[np.maximum(i - 10, 0) : i + 1]))
                - avg_rri,  # local RR Interval
            ]
        )
        y.append(categories[i])
        groups.append(data["record"])
    # x1, x2, y, groups: 박동의 분류와 레코드를 저장함
    return x1, x2, y, groups


def load_data(
    wavelet,
    scales,
    sampling_rate,
    filename="./dataset/physionet.org/files/mitdb/1.0.0/mitdb.pkl",
):
    # 파일 오픈
    import pickle
    from sklearn.preprocessing import RobustScaler

    with open(filename, "rb") as f:
        train_data, test_data = pickle.load(f)

    cpus = (
        22 if joblib.cpu_count() > 22 else joblib.cpu_count() - 1
    )  # 멀티 프로세스 CPU 설정

    # CWT 변환 -> 멀티프로세싱 이용
    x1_train, x2_train, y_train, groups_train = [], [], [], []
    with ProcessPoolExecutor(max_workers=cpus) as executor:
        for x1, x2, y, groups in executor.map(
            partial(
                worker,
                wavelet=wavelet,
                scales=scales,
                sampling_period=1.0 / sampling_rate,
            ),
            train_data,
        ):
            x1_train.append(x1)
            x2_train.append(x2)
            y_train.append(y)
            groups_train.append(groups)

    # 각 Train 데이터 합치기
    x1_train = np.expand_dims(np.concatenate(x1_train, axis=0), axis=1).astype(
        np.float32
    )
    x2_train = np.concatenate(x2_train, axis=0).astype(np.float32)
    y_train = np.concatenate(y_train, axis=0).astype(np.int64)
    groups_train = np.concatenate(groups_train, axis=0)

    # Test 데이터 전처리
    x1_test, x2_test, y_test, groups_test = [], [], [], []
    with ProcessPoolExecutor(max_workers=cpus) as executor:
        for x1, x2, y, groups in executor.map(
            partial(
                worker,
                wavelet=wavelet,
                scales=scales,
                sampling_period=1.0 / sampling_rate,
            ),
            test_data,
        ):
            x1_test.append(x1)
            x2_test.append(x2)
            y_test.append(y)
            groups_test.append(groups)

    # Test 데이터 합치기
    x1_test = np.expand_dims(np.concatenate(x1_test, axis=0), axis=1).astype(np.float32)
    x2_test = np.concatenate(x2_test, axis=0).astype(np.float32)
    y_test = np.concatenate(y_test, axis=0).astype(np.int64)
    groups_test = np.concatenate(groups_test, axis=0)

    # Normalization 진행
    scaler = RobustScaler()
    x2_train = scaler.fit_transform(x2_train)
    x2_test = scaler.transform(x2_test)

    return (x1_train, x2_train, y_train, groups_train), (
        x1_test,
        x2_test,
        y_test,
        groups_test,
    )


def main():
    # 데이터 샘플링 비율 선정
    sampling_rate = 360

    # Wavelet 설정 - 종류
    wavelet = "mexh"  # mexh, morl, gaus8, gaus4
    # Wavelet 설정 - 스케일
    scales = pywt.central_frequency(wavelet) * sampling_rate / np.arange(1, 101, 1)

    # 데이터 로드 함수 호출 (훈련 및 테스트 데이터)
    (x1_train, x2_train, y_train, groups_train), (
        x1_test,
        x2_test,
        y_test,
        groups_test,
    ) = load_data(wavelet=wavelet, scales=scales, sampling_rate=sampling_rate)
    print("Data loaded successfully!")

    # Tensorboard 로그 디렉토리 생성
    log_dir = "./logs/{}".format(wavelet)
    shutil.rmtree(log_dir, ignore_errors=True)

    # 학습 프로세스에 사용할 콜백 함수 설정
    callbacks = [
        Initializer(
            "[conv|fc]*.weight", fn=torch.nn.init.kaiming_normal_
        ),  # He initialization
        Initializer(
            "[conv|fc]*.bias", fn=partial(torch.nn.init.constant_, val=0.0)
        ),  # bias 초기화
        LRScheduler(policy=StepLR, step_size=5, gamma=0.1),  # Learning Rate Scheduler
        EpochScoring(
            scoring=make_scorer(f1_score, average="macro"),
            lower_is_better=False,
            name="valid_f1",
        ),  # F1 Score
        TensorBoard(SummaryWriter(log_dir)),  # Tensorboard을 통한 학습과정로깅
    ]

    # 모델 구성 및 설정
    net = NeuralNetClassifier(  # skorch is extensive package of pytorch for compatible with scikit-learn
        MyModule,  # 사용할 모듈 지정
        criterion=torch.nn.CrossEntropyLoss,  # 손실 함수 설정
        optimizer=torch.optim.Adam,  # 최적화 알고리즘 설정
        lr=0.001,  # 학습률 설정
        max_epochs=30,  # 최대 학습 에폭 설정
        batch_size=1024,  # 배치 사이즈 설정
        train_split=predefined_split(
            Dataset({"x1": x1_test, "x2": x2_test}, y_test)
        ),  # 훈련 및 검증 데이터 분리
        verbose=1,  # 학습 과정 출력
        device="cuda",  # GPU 사용
        callbacks=callbacks,  # 콜백 함수 설정
        iterator_train__shuffle=True,  # 훈련 데이터 셔플
        optimizer__weight_decay=0,  # 가중치 감쇠 설정
    )
    # 모델 학습
    net.fit({"x1": x1_train, "x2": x2_train}, y_train)
    # 모델 테스트
    y_true, y_pred = y_test, net.predict({"x1": x1_test, "x2": x2_test})

    # 결과 출력
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=4))

    net.save_params(f_params="./models/model_{}.pkl".format(wavelet))


if __name__ == "__main__":
    main()
