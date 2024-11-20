import os
import shutil

name_list = """2004/MIDI-Unprocessed_SMF_13_01_2004_01-05_ORIG_MID--AUDIO_13_R1_2004_05_Track05_wav.midi
2004/MIDI-Unprocessed_XP_10_R1_2004_01-02_ORIG_MID--AUDIO_10_R1_2004_02_Track02_wav.midi
2004/MIDI-Unprocessed_XP_09_R1_2004_01-02_ORIG_MID--AUDIO_09_R1_2004_02_Track02_wav.midi
2004/MIDI-Unprocessed_XP_09_R1_2004_01-02_ORIG_MID--AUDIO_09_R1_2004_03_Track03_wav.midi
2004/MIDI-Unprocessed_XP_15_R1_2004_03_ORIG_MID--AUDIO_15_R1_2004_03_Track03_wav.midi
2004/MIDI-Unprocessed_XP_15_R1_2004_01-02_ORIG_MID--AUDIO_15_R1_2004_01_Track01_wav.midi
2004/MIDI-Unprocessed_SMF_17_R1_2004_01-03_ORIG_MID--AUDIO_17_R1_2004_02_Track02_wav--2.midi
2004/MIDI-Unprocessed_SMF_17_R1_2004_01-03_ORIG_MID--AUDIO_17_R1_2004_03_Track03_wav.midi
2004/MIDI-Unprocessed_XP_01_R1_2004_01-02_ORIG_MID--AUDIO_01_R1_2004_02_Track02_wav.midi
2004/MIDI-Unprocessed_XP_01_R1_2004_01-02_ORIG_MID--AUDIO_01_R1_2004_03_Track03_wav.midi
2004/MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_10_Track10_wav.midi
2004/MIDI-Unprocessed_XP_21_R1_2004_02_ORIG_MID--AUDIO_21_R1_2004_02_Track02_wav.midi
2004/MIDI-Unprocessed_XP_21_R1_2004_02_ORIG_MID--AUDIO_21_R1_2004_03_Track03_wav.midi
2006/MIDI-Unprocessed_11_R1_2006_01-06_ORIG_MID--AUDIO_11_R1_2006_02_Track02_wav.midi
2006/MIDI-Unprocessed_10_R1_2006_01-04_ORIG_MID--AUDIO_10_R1_2006_02_Track02_wav.midi
2006/MIDI-Unprocessed_21_R1_2006_01-04_ORIG_MID--AUDIO_21_R1_2006_02_Track02_wav.midi
2006/MIDI-Unprocessed_13_R1_2006_01-06_ORIG_MID--AUDIO_13_R1_2006_02_Track02_wav.midi
2006/MIDI-Unprocessed_05_R1_2006_01-05_ORIG_MID--AUDIO_05_R1_2006_01_Track01_wav.midi
2008/MIDI-Unprocessed_06_R1_2008_01-04_ORIG_MID--AUDIO_06_R1_2008_wav--2.midi
2008/MIDI-Unprocessed_05_R1_2008_01-04_ORIG_MID--AUDIO_05_R1_2008_wav--2.midi
2008/MIDI-Unprocessed_17_R1_2008_01-04_ORIG_MID--AUDIO_17_R1_2008_wav--2.midi
2008/MIDI-Unprocessed_09_R1_2008_01-05_ORIG_MID--AUDIO_09_R1_2008_wav--2.midi
2008/MIDI-Unprocessed_18_R1_2008_01-04_ORIG_MID--AUDIO_18_R1_2008_wav--2.midi
2008/MIDI-Unprocessed_15_R1_2008_01-04_ORIG_MID--AUDIO_15_R1_2008_wav--2.midi
2008/MIDI-Unprocessed_03_R1_2008_01-04_ORIG_MID--AUDIO_03_R1_2008_wav--2.midi
2008/MIDI-Unprocessed_11_R1_2008_01-04_ORIG_MID--AUDIO_11_R1_2008_wav--2.midi
2008/MIDI-Unprocessed_01_R1_2008_01-04_ORIG_MID--AUDIO_01_R1_2008_wav--2.midi
2008/MIDI-Unprocessed_07_R1_2008_01-04_ORIG_MID--AUDIO_07_R1_2008_wav--2.midi
2008/MIDI-Unprocessed_10_R1_2008_01-04_ORIG_MID--AUDIO_10_R1_2008_wav--2.midi
2008/MIDI-Unprocessed_02_R3_2008_01-03_ORIG_MID--AUDIO_02_R3_2008_wav--1.midi
2008/MIDI-Unprocessed_03_R3_2008_01-03_ORIG_MID--AUDIO_03_R3_2008_wav--1.midi
2008/MIDI-Unprocessed_04_R3_2008_01-07_ORIG_MID--AUDIO_04_R3_2008_wav--1.midi
2008/MIDI-Unprocessed_09_R3_2008_01-07_ORIG_MID--AUDIO_09_R3_2008_wav--4.midi
2008/MIDI-Unprocessed_12_R3_2008_01-04_ORIG_MID--AUDIO_12_R3_2008_wav--1.midi
2009/MIDI-Unprocessed_10_R1_2009_01-02_ORIG_MID--AUDIO_10_R1_2009_10_R1_2009_02_WAV.midi
2009/MIDI-Unprocessed_11_R1_2009_01-05_ORIG_MID--AUDIO_11_R1_2009_11_R1_2009_01_WAV.midi
2009/MIDI-Unprocessed_06_R1_2009_01-02_ORIG_MID--AUDIO_06_R1_2009_06_R1_2009_02_WAV.midi
2009/MIDI-Unprocessed_09_R1_2009_01-04_ORIG_MID--AUDIO_09_R1_2009_09_R1_2009_03_WAV.midi
2009/MIDI-Unprocessed_21_R1_2009_01-03_ORIG_MID--AUDIO_21_R1_2009_21_R1_2009_01_WAV.midi
2009/MIDI-Unprocessed_17_R1_2009_01-03_ORIG_MID--AUDIO_17_R1_2009_17_R1_2009_01_WAV.midi
2009/MIDI-Unprocessed_20_R1_2009_01-05_ORIG_MID--AUDIO_20_R1_2009_20_R1_2009_01_WAV.midi
2009/MIDI-Unprocessed_16_R1_2009_01-02_ORIG_MID--AUDIO_16_R1_2009_16_R1_2009_01_WAV.midi
2011/MIDI-Unprocessed_09_R1_2011_MID--AUDIO_R1-D3_13_Track13_wav.midi
2011/MIDI-Unprocessed_19_R1_2011_MID--AUDIO_R1-D7_13_Track13_wav.midi
2011/MIDI-Unprocessed_08_R1_2011_MID--AUDIO_R1-D3_08_Track08_wav.midi
2011/MIDI-Unprocessed_15_R1_2011_MID--AUDIO_R1-D6_08_Track08_wav.midi
2011/MIDI-Unprocessed_11_R1_2011_MID--AUDIO_R1-D4_08_Track08_wav.midi
2011/MIDI-Unprocessed_13_R1_2011_MID--AUDIO_R1-D5_04_Track04_wav.midi
2011/MIDI-Unprocessed_18_R1_2011_MID--AUDIO_R1-D7_08_Track08_wav.midi
2011/MIDI-Unprocessed_12_R2_2011_MID--AUDIO_R2-D4_03_Track03_wav.midi
2011/MIDI-Unprocessed_20_R1_2011_MID--AUDIO_R1-D8_04_Track04_wav.midi
2011/MIDI-Unprocessed_25_R1_2011_MID--AUDIO_R1-D9_15_Track15_wav.midi
2011/MIDI-Unprocessed_21_R1_2011_MID--AUDIO_R1-D8_08_Track08_wav.midi
2011/MIDI-Unprocessed_24_R1_2011_MID--AUDIO_R1-D9_09_Track09_wav.midi
2011/MIDI-Unprocessed_16_R1_2011_MID--AUDIO_R1-D6_14_Track14_wav.midi
2011/MIDI-Unprocessed_17_R1_2011_MID--AUDIO_R1-D7_03_Track03_wav.midi
2011/MIDI-Unprocessed_14_R1_2011_MID--AUDIO_R1-D6_03_Track03_wav.midi
2011/MIDI-Unprocessed_22_R1_2011_MID--AUDIO_R1-D8_13_Track13_wav.midi
2011/MIDI-Unprocessed_23_R1_2011_MID--AUDIO_R1-D9_03_Track03_wav.midi
2011/MIDI-Unprocessed_06_R1_2011_MID--AUDIO_R1-D2_15_Track15_wav.midi
2011/MIDI-Unprocessed_12_R3_2011_MID--AUDIO_R3-D4_03_Track03_wav.midi
2011/MIDI-Unprocessed_17_R3_2011_MID--AUDIO_R3-D6_02_Track02_wav.midi
2011/MIDI-Unprocessed_22_R3_2011_MID--AUDIO_R3-D7_03_Track03_wav.midi
2011/MIDI-Unprocessed_25_R3_2011_MID--AUDIO_R3-D9_04_Track04_wav.midi
2013/ORIG-MIDI_02_7_6_13_Group__MID--AUDIO_08_R1_2013_wav--2.midi
2013/ORIG-MIDI_01_7_6_13_Group__MID--AUDIO_03_R1_2013_wav--2.midi
2013/ORIG-MIDI_02_7_6_13_Group__MID--AUDIO_05_R1_2013_wav--2.midi
2013/ORIG-MIDI_01_7_6_13_Group__MID--AUDIO_01_R1_2013_wav--2.midi
2013/ORIG-MIDI_01_7_6_13_Group__MID--AUDIO_02_R1_2013_wav--2.midi
2013/ORIG-MIDI_02_7_6_13_Group__MID--AUDIO_06_R1_2013_wav--1.midi
2013/ORIG-MIDI_01_7_7_13_Group__MID--AUDIO_13_R1_2013_wav--2.midi
2013/ORIG-MIDI_02_7_7_13_Group__MID--AUDIO_16_R1_2013_wav--2.midi
2013/ORIG-MIDI_01_7_10_13_Group_MID--AUDIO_07_R3_2013_wav--2.midi
2013/ORIG-MIDI_01_7_10_13_Group_MID--AUDIO_08_R3_2013_wav--1.midi
2013/ORIG-MIDI_02_7_10_13_Group_MID--AUDIO_12_R3_2013_wav--1.midi
2013/ORIG-MIDI_02_7_10_13_Group_MID--AUDIO_14_R3_2013_wav--1.midi
2014/MIDI-UNPROCESSED_04-08-12_R3_2014_MID--AUDIO_12_R3_2014_wav--1.midi
2014/MIDI-UNPROCESSED_14-15_R1_2014_MID--AUDIO_15_R1_2014_wav--1.midi
2014/MIDI-UNPROCESSED_14-15_R1_2014_MID--AUDIO_15_R1_2014_wav--2.midi
2014/MIDI-UNPROCESSED_16-18_R1_2014_MID--AUDIO_16_R1_2014_wav--2.midi
2014/MIDI-UNPROCESSED_04-05_R1_2014_MID--AUDIO_04_R1_2014_wav--2.midi
2014/MIDI-UNPROCESSED_01-03_R1_2014_MID--AUDIO_03_R1_2014_wav--3.midi
2014/MIDI-UNPROCESSED_11-13_R1_2014_MID--AUDIO_13_R1_2014_wav--3.midi
2014/MIDI-UNPROCESSED_16-18_R1_2014_MID--AUDIO_17_R1_2014_wav--1.midi
2014/MIDI-UNPROCESSED_16-18_R1_2014_MID--AUDIO_18_R1_2014_wav--2.midi
2014/MIDI-UNPROCESSED_06-08_R1_2014_MID--AUDIO_06_R1_2014_wav--3.midi
2014/MIDI-UNPROCESSED_01-03_R1_2014_MID--AUDIO_02_R1_2014_wav--2.midi
2014/MIDI-UNPROCESSED_19-20_R1_2014_MID--AUDIO_19_R1_2014_wav--2.midi
2015/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_03_R1_2015_wav--3.midi
2015/MIDI-Unprocessed_R1_D2-13-20_mid--AUDIO-from_mp3_13_R1_2015_wav--2.midi
2015/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_06_R1_2015_wav--2.midi
2015/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_07_R1_2015_wav--2.midi
2015/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_08_R1_2015_wav--2.midi
2015/MIDI-Unprocessed_R1_D1-9-12_mid--AUDIO-from_mp3_10_R1_2015_wav--3.midi
2015/MIDI-Unprocessed_R1_D2-21-22_mid--AUDIO-from_mp3_22_R1_2015_wav--2.midi
2015/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_05_R1_2015_wav--3.midi
2015/MIDI-Unprocessed_R1_D1-9-12_mid--AUDIO-from_mp3_09_R1_2015_wav--2.midi
2015/MIDI-Unprocessed_R1_D2-13-20_mid--AUDIO-from_mp3_17_R1_2015_wav--2.midi
2015/MIDI-Unprocessed_R1_D2-13-20_mid--AUDIO-from_mp3_16_R1_2015_wav--2.midi
2015/MIDI-Unprocessed_R2_D2-19-21-22_mid--AUDIO-from_mp3_19_R2_2015_wav--2.midi
2015/MIDI-Unprocessed_R2_D2-12-13-15_mid--AUDIO-from_mp3_12_R2_2015_wav--1.midi
2015/MIDI-Unprocessed_R2_D2-19-21-22_mid--AUDIO-from_mp3_21_R2_2015_wav--2.midi
2015/MIDI-Unprocessed_R2_D2-19-21-22_mid--AUDIO-from_mp3_22_R2_2015_wav--2.midi
2015/MIDI-Unprocessed_R2_D1-2-3-6-7-8-11_mid--AUDIO-from_mp3_08_R2_2015_wav--1.midi
2015/MIDI-Unprocessed_R2_D1-2-3-6-7-8-11_mid--AUDIO-from_mp3_11_R2_2015_wav--1.midi
2017/MIDI-Unprocessed_058_PIANO058_MID--AUDIO-split_07-07-17_Piano-e_2-02_wav--3.midi
2017/MIDI-Unprocessed_044_PIANO044_MID--AUDIO-split_07-06-17_Piano-e_1-04_wav--2.midi
2017/MIDI-Unprocessed_046_PIANO046_MID--AUDIO-split_07-06-17_Piano-e_2-02_wav--2.midi
2017/MIDI-Unprocessed_049_PIANO049_MID--AUDIO-split_07-06-17_Piano-e_2-06_wav--2.midi
2017/MIDI-Unprocessed_050_PIANO050_MID--AUDIO-split_07-06-17_Piano-e_3-01_wav--2.midi
2017/MIDI-Unprocessed_053_PIANO053_MID--AUDIO-split_07-06-17_Piano-e_3-04_wav--3.midi
2017/MIDI-Unprocessed_055_PIANO055_MID--AUDIO-split_07-07-17_Piano-e_1-04_wav--2.midi
2017/MIDI-Unprocessed_057_PIANO057_MID--AUDIO-split_07-07-17_Piano-e_1-07_wav--2.midi
2017/MIDI-Unprocessed_066_PIANO066_MID--AUDIO-split_07-07-17_Piano-e_3-02_wav--2.midi
2017/MIDI-Unprocessed_041_PIANO041_MID--AUDIO-split_07-06-17_Piano-e_1-01_wav--2.midi
2017/MIDI-Unprocessed_043_PIANO043_MID--AUDIO-split_07-06-17_Piano-e_1-03_wav--2.midi
2017/MIDI-Unprocessed_048_PIANO048_MID--AUDIO-split_07-06-17_Piano-e_2-05_wav--2.midi
2017/MIDI-Unprocessed_056_PIANO056_MID--AUDIO-split_07-07-17_Piano-e_1-05_wav--3.midi
2017/MIDI-Unprocessed_059_PIANO059_MID--AUDIO-split_07-07-17_Piano-e_2-03_wav--2.midi
2017/MIDI-Unprocessed_062_PIANO062_MID--AUDIO-split_07-07-17_Piano-e_2-07_wav--1.midi
2017/MIDI-Unprocessed_045_PIANO045_MID--AUDIO-split_07-06-17_Piano-e_2-01_wav--2.midi
2017/MIDI-Unprocessed_051_PIANO051_MID--AUDIO-split_07-06-17_Piano-e_3-02_wav--2.midi
2017/MIDI-Unprocessed_065_PIANO065_MID--AUDIO-split_07-07-17_Piano-e_3-01_wav--2.midi
2017/MIDI-Unprocessed_060_PIANO060_MID--AUDIO-split_07-07-17_Piano-e_2-04_wav--4.midi
2017/MIDI-Unprocessed_061_PIANO061_MID--AUDIO-split_07-07-17_Piano-e_2-05_wav--2.midi
2017/MIDI-Unprocessed_047_PIANO047_MID--AUDIO-split_07-06-17_Piano-e_2-04_wav--1.midi
2017/MIDI-Unprocessed_067_PIANO067_MID--AUDIO-split_07-07-17_Piano-e_3-03_wav--2.midi
2017/MIDI-Unprocessed_071_PIANO071_MID--AUDIO-split_07-08-17_Piano-e_1-04_wav--1.midi
2017/MIDI-Unprocessed_073_PIANO073_MID--AUDIO-split_07-08-17_Piano-e_2-02_wav--1.midi
2017/MIDI-Unprocessed_075_PIANO075_MID--AUDIO-split_07-08-17_Piano-e_2-06_wav--2.midi
2017/MIDI-Unprocessed_078_PIANO078_MID--AUDIO-split_07-09-17_Piano-e_1-02_wav--1.midi
2017/MIDI-Unprocessed_079_PIANO079_MID--AUDIO-split_07-09-17_Piano-e_1-04_wav--3.midi
2017/MIDI-Unprocessed_080_PIANO080_MID--AUDIO-split_07-09-17_Piano-e_1-06_wav--2.midi
2017/MIDI-Unprocessed_082_PIANO082_MID--AUDIO-split_07-09-17_Piano-e_2_-04_wav--1.midi
2018/MIDI-Unprocessed_Chamber6_MID--AUDIO_20_R3_2018_wav--1.midi
2018/MIDI-Unprocessed_Recital13-15_MID--AUDIO_14_R1_2018_wav--3.midi
2018/MIDI-Unprocessed_Recital17-19_MID--AUDIO_18_R1_2018_wav--1.midi
2018/MIDI-Unprocessed_Recital4_MID--AUDIO_04_R1_2018_wav--3.midi
2018/MIDI-Unprocessed_Recital1-3_MID--AUDIO_02_R1_2018_wav--1.midi
2018/MIDI-Unprocessed_Recital13-15_MID--AUDIO_14_R1_2018_wav--2.midi
2018/MIDI-Unprocessed_Recital17-19_MID--AUDIO_19_R1_2018_wav--2.midi
2018/MIDI-Unprocessed_Recital9-11_MID--AUDIO_10_R1_2018_wav--2.midi
2018/MIDI-Unprocessed_Recital13-15_MID--AUDIO_15_R1_2018_wav--3.midi
2018/MIDI-Unprocessed_Recital1-3_MID--AUDIO_03_R1_2018_wav--1.midi
2018/MIDI-Unprocessed_Recital5-7_MID--AUDIO_06_R1_2018_wav--2.midi
2018/MIDI-Unprocessed_Recital5-7_MID--AUDIO_05_R1_2018_wav--1.midi"""

file_list = [
    line.split("/")[-1].replace(".midi", "")
    for line in name_list.strip().split("\n")
]

path = 'D:Inżynierka\maestro-v3.0.0'

for root, _, files in os.walk(path):
    for file in files:
        if file[:-4] in file_list:
            shutil.copy2(os.path.join(root, file), 'C:\\Users\\micha\\PycharmProjects\\AMT\\train_model\\midi')
        if file[:-5] in file_list:
            shutil.copy2(os.path.join(root, file), 'C:\\Users\\micha\\PycharmProjects\\AMT\\train_model\\wav')