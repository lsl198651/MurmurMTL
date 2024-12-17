import os
import shutil

import numpy as np
import pandas as pd
from pydub import AudioSegment
from tools.subdir import maksubdir


def label_most(label, label_slice_len):
    label_sliced = []
    for i in range(len(label) // label_slice_len):
        slice = label[label_slice_len * i: label_slice_len * (i + 1)]
        slice = slice.astype(np.int64)
        count = np.bincount(slice)
        appro_label = np.argmax(count)
        label_sliced.append(appro_label)
    label_sliced = np.asarray(label_sliced)
    return label_sliced


def load_label(label_path):
    df = pd.read_csv(label_path, names=["START", "END", "STATE"], sep="\t")
    starts = np.asarray(df["START"]) * 4000
    starts = starts.astype(int)
    states = np.asarray(df["STATE"], dtype=int)
    return starts, states


def load_murmur(location, murmurInfo):
    timeTypes = ["nan", "Early", "Mid", "Late", "Holo"]
    sysMurmur = False
    sysTime = 0  # 0: no murmur, 1: early systolic murmur, 2: mid systolic murmur, 3: late systolic murmur, 4: holosystolic murmur
    diaMurmur = False
    diaTime = 0
    if (murmurInfo["Murmur"].values[0] != "Present") or (
            location not in murmurInfo["Murmur locations"].values[0]
    ):
        return sysMurmur, sysTime, diaMurmur, diaTime
    else:
        sysMurmur = str(murmurInfo["Systolic murmur timing"].values[0]) != "nan"
        for timetype in timeTypes:
            if timetype in str(murmurInfo["Systolic murmur timing"].values[0]):
                sysTime = timeTypes.index(timetype)
                break
        diaMurmur = str(murmurInfo["Diastolic murmur timing"].values[0]) != "nan"
        for timetype in timeTypes:
            if timetype in str(murmurInfo["Diastolic murmur timing"].values[0]):
                diaTime = timeTypes.index(timetype)
                break
    return sysMurmur, sysTime, diaMurmur, diaTime


def split_label(label_path):
    starts, states = load_label(label_path)
    s1 = []
    s2 = []
    sys = []
    dia = []
    count = 0
    for start, state in zip(starts, states):
        index = (
            np.array([start, starts[count + 1]])
            if count < len(starts) - 1
            else np.array([start])
        )
        if state == 1:
            s1.append(index)
        elif state == 3:
            s2.append(index)
        elif state == 2:
            sys.append(index)
        elif state == 4:
            dia.append(index)

        count += 1

    return s1, s2, sys, dia


def getMurmurEnds(stateStarts, stateEnds, murmurType):
    murmurStart = stateStarts
    murmurEnd = stateEnds
    if murmurType == 1:
        murmurEnd = stateStarts + int(0.5 * (stateEnds - stateStarts))
    elif murmurType == 2:
        murmurStart = stateStarts + int(0.25 * (stateEnds - stateStarts))
        murmurEnd = stateStarts + int(0.75 * (stateEnds - stateStarts))
    elif murmurType == 3:
        murmurStart = stateStarts + int(0.5 * (stateEnds - stateStarts))
    return murmurStart, murmurEnd


def gen_label(label_path_, wav_len, murmurInfo):
    s1, s2, sys, dia = split_label(label_path_)
    sysMurmur, sysTime, diaMurmur, diaTime = load_murmur(
        label_path_.split("\\")[-1].split(".")[0].split("_")[1], murmurInfo
    )
    murmurs = np.zeros(wav_len)
    labels = np.zeros(wav_len)
    # s1: 1, s2: 3, sys: 2, dia: 4, no_signal: 0
    for duration in s1:
        end = duration[1] if len(duration) == 2 else -1
        labels[duration[0]: end] = 1
    for duration in s2:
        end = duration[1] if len(duration) == 2 else -1
        labels[duration[0]: end] = 3
    for duration in sys:
        end = duration[1] if len(duration) == 2 else -1
        labels[duration[0]: end] = 2
        if sysMurmur:
            murmurStart, murmurEnd = getMurmurEnds(duration[0], end, sysTime)
            murmurs[murmurStart:murmurEnd] = 1
    for duration in dia:
        end = duration[1] if len(duration) == 2 else -1
        labels[duration[0]: end] = 4
        if diaMurmur:
            murmurStart, murmurEnd = getMurmurEnds(duration[0], end, diaTime)
            murmurs[murmurStart:murmurEnd] = 1
    return labels, murmurs


inputBase = "..\..\database5_test\\1.0rawdata"
cutBase = "..\..\database5_test\\1.0rowSegInfo"

# 保存与seg中信息长度一致的wav
wavSameBase = "..\..\database5_test\\1.1wavSame"
maksubdir(wavSameBase)

outputBase = "..\..\database5_test\\2.0wav10s"
maksubdir(outputBase)

labelBase = "..\..\database5_test\\2.1SegInfo"
murmurBase = "..\..\database5_test\\2.2MurmurInfo"
maksubdir(labelBase)
maksubdir(murmurBase)

murmurInfoBase = "..\..\database5_test\\test_data.csv"
murmurInfo = pd.read_csv(murmurInfoBase, header=0)

targetLen = 10  # 10s

files = os.listdir(inputBase)
for file in files:
    wav = AudioSegment.from_wav(os.path.join(inputBase, file))
    wavMurmur = murmurInfo[int(file.split("_")[0]) == murmurInfo["Patient ID"]]
    segInfo = pd.read_csv(
        os.path.join(cutBase, file.split(".")[0] + ".tsv"), header=None, sep="\t"
    )
    lastEnd = segInfo.iloc[-1, 1]
    # 保存与seg中信息长度一致的wav
    if lastEnd >= len(wav.raw_data) / 8000:
        shutil.copyfile(
            os.path.join(inputBase, file),
            os.path.join(wavSameBase, file),
        )
    else:
        wav = wav[0: lastEnd * 1000]
        wav.export(
            os.path.join(wavSameBase, file),
            format="wav",
        )
    assert lastEnd >= len(wav.raw_data) / 8000
    label, murmurs = gen_label(
        os.path.join(cutBase, file.split(".")[0] + ".tsv"),
        len(wav.raw_data) // 2,
        wavMurmur,
    )

    # romove the zero label at the beginning and the end
    notZero = label != 0
    if notZero.sum() == 0:
        continue
    startZeroIndex = notZero.tolist().index(True) if notZero[0] == False else 0
    notZero = notZero[startZeroIndex:]
    endZeroIndex = (
                       notZero.tolist().index(False) if notZero[-1] == False else len(notZero)
                   ) + startZeroIndex
    if startZeroIndex != 0 or endZeroIndex != len(label):
        # startZeroIndex 与 endZeroIndex 需要为4的倍数，且在原始startZeroIndex与endZeroIndex的范围内
        startZeroIndex = (
            startZeroIndex + (4 - startZeroIndex % 4)
            if startZeroIndex % 4 != 0
            else startZeroIndex
        )
        endZeroIndex = endZeroIndex - endZeroIndex % 4
        label = label[startZeroIndex:endZeroIndex]
        murmurs = murmurs[startZeroIndex:endZeroIndex]
        wav = wav[int(startZeroIndex / 4): int(endZeroIndex / 4)]

    label = label_most(label, 80)  # 80个采样点，4000采样率，20ms
    murmurs = label_most(murmurs, 80)

    newWav = wav[0: len(label) * 20]
    wavLen = newWav.duration_seconds
    if wavLen == 0:
        continue
    if wav.duration_seconds < targetLen:
        repeatTimes = int(targetLen // wavLen)
        outWav = AudioSegment.empty()
        outSeg = np.array([])
        outMurmur = np.array([])
        for i in range(repeatTimes):
            outWav += newWav
            outSeg = np.hstack((outSeg, label))
            outMurmur = np.hstack((outMurmur, murmurs))
        outWav += newWav[
                  0: int(np.round(targetLen * 1000 - wavLen * 1000 * repeatTimes))
                  ]
        outSeg = np.hstack(
            (
                outSeg,
                label[0: int(np.round(targetLen * 50 - wavLen * 50 * repeatTimes))],
            )
        )
        outMurmur = np.hstack(
            (
                outMurmur,
                murmurs[0: int(np.round(targetLen * 50 - wavLen * 50 * repeatTimes))],
            )
        )
        # if (
        #     AudioSegment.from_wav(
        #         os.path.join(outputBase, file.split(".")[0] + "_1.wav")
        #     ).duration_seconds
        #     != 10
        # ):
        #     print(file)
        assert outWav.duration_seconds == 10
        outWav.export(
            os.path.join(outputBase, file.split(".")[0] + "_1.wav"), format="wav"
        )
        if len(outSeg) != 500:
            print(file)
        np.save(os.path.join(labelBase, file.split(".")[0] + "_1.npy"), outSeg)
        np.save(os.path.join(murmurBase, file.split(".")[0] + "_1.npy"), outMurmur)
    else:
        cutNum = int(np.ceil(wavLen / targetLen))
        padWav = (
                newWav
                + newWav[0: int(np.round(cutNum * targetLen * 1000 - wavLen * 1000))]
        )
        padSeg = np.hstack(
            (label, label[0: int(np.round(cutNum * targetLen * 50 - wavLen * 50))])
        )
        padMurmur = np.hstack(
            (murmurs, murmurs[0: int(np.round(cutNum * targetLen * 50 - wavLen * 50))])
        )
        for i in range(cutNum):
            outWav = padWav[i * targetLen * 1000: (i + 1) * targetLen * 1000]
            outSeg = padSeg[i * targetLen * 50: (i + 1) * targetLen * 50]
            outMurmur = padMurmur[i * targetLen * 50: (i + 1) * targetLen * 50]
            # if (
            #     AudioSegment.from_wav(
            #         os.path.join(outputBase, file.split(".")[0] + "_1.wav")
            #     ).duration_seconds
            #     != 10
            # ):
            #     print(file)
            assert outWav.duration_seconds == 10
            outWav.export(
                os.path.join(
                    outputBase, file.split(".")[0] + "_" + str(i + 1) + ".wav"
                ),
                format="wav",
            )
            if len(outSeg) != 500:
                print(file)
            np.save(
                os.path.join(labelBase, file.split(".")[0] + "_" + str(i + 1) + ".npy"),
                outSeg,
            )
            np.save(
                os.path.join(
                    murmurBase, file.split(".")[0] + "_" + str(i + 1) + ".npy"
                ),
                outMurmur,
            )
