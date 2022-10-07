# Run this to turn all the UrbanSound8K clips into pcm_16
import os
import wave
import soundfile


base_folder = os.path.join("UrbanSound8K", "audio")
folders = [os.path.join(base_folder, "fold") + str(i) for i in range(1,11)]
print(folders)


def main():
    print(check_bad_files())


def convert_all_to_pcm16():
    print()
    for folder in folders:
        for file_path in os.listdir(folder):
            file_path = os.path.join(folder, file_path)
            try:
                wav, samplerate = soundfile.read(file_path)
            except:
                print("fail")
                continue
            if soundfile.check_format(wav, 'PCM_16'):
                # Fix format if necessary, hopefully it doesn't break anything if I ctrl + C while this is running during modelling
                soundfile.write(file_path, wav, 11025, subtype='PCM_16')
                print("reformatting .wav")


def check_bad_files():
    print()
    bad_files = []
    for folder in folders:
        for file_path in os.listdir(folder):
            file_path = os.path.join(folder, file_path)
            try:
                with wave.open(file_path,'r') as f:
                    frames = f.getnframes()
                    rate = f.getframerate()
                    duration = frames / float(rate)
                    if duration < 1:
                        print("BAD")
                        bad_files.append(file_path)
                    #print(duration)
            except:
                print("BAD")
                bad_files.append(file_path)
    return bad_files


BAD_FILES = ['UrbanSound8K/audio/fold1/101415-3-0-2.wav', 'UrbanSound8K/audio/fold1/160094-3-0-0.wav',
    'UrbanSound8K/audio/fold1/87275-1-0-0.wav', 'UrbanSound8K/audio/fold1/87275-1-1-0.wav', 
    'UrbanSound8K/audio/fold1/87275-1-2-0.wav', 'UrbanSound8K/audio/fold1/87275-1-3-0.wav', 
    'UrbanSound8K/audio/fold1/87275-1-4-0.wav', 'UrbanSound8K/audio/fold1/87275-1-5-0.wav', 'UrbanSound8K/audio/fold2/147672-3-0-0.wav', 
    'UrbanSound8K/audio/fold2/147672-3-1-0.wav', 'UrbanSound8K/audio/fold2/147672-3-2-0.wav', 'UrbanSound8K/audio/fold2/155129-1-0-0.wav', 
    'UrbanSound8K/audio/fold2/155129-1-1-0.wav', 'UrbanSound8K/audio/fold2/160092-3-0-0.wav', 'UrbanSound8K/audio/fold2/17307-1-0-0.wav', 
    'UrbanSound8K/audio/fold2/4201-3-0-0.wav', 'UrbanSound8K/audio/fold2/4911-3-0-0.wav', 'UrbanSound8K/audio/fold2/76091-6-2-0.wav', 
    'UrbanSound8K/audio/fold2/76091-6-3-0.wav', 'UrbanSound8K/audio/fold2/76091-6-4-0.wav', 'UrbanSound8K/audio/fold3/118070-1-0-0.wav', 
    'UrbanSound8K/audio/fold3/118496-1-0-0.wav', 'UrbanSound8K/audio/fold3/118496-1-1-0.wav', 'UrbanSound8K/audio/fold3/151359-1-0-0.wav', 
    'UrbanSound8K/audio/fold3/151359-1-1-0.wav', 'UrbanSound8K/audio/fold3/151359-1-2-0.wav', 'UrbanSound8K/audio/fold3/151359-1-3-0.wav', 
    'UrbanSound8K/audio/fold3/160093-3-0-0.wav', 'UrbanSound8K/audio/fold3/18594-1-2-0.wav', 'UrbanSound8K/audio/fold3/18594-1-3-0.wav', 
    'UrbanSound8K/audio/fold3/18594-1-4-0.wav', 'UrbanSound8K/audio/fold4/47926-3-1-0.wav', 'UrbanSound8K/audio/fold5/26177-1-0-0.wav', 
    'UrbanSound8K/audio/fold5/43803-1-0-0.wav', 'UrbanSound8K/audio/fold5/71439-1-1-0.wav', 'UrbanSound8K/audio/fold6/162702-1-0-0.wav', 
    'UrbanSound8K/audio/fold6/43802-1-0-0.wav', 'UrbanSound8K/audio/fold6/43802-1-1-0.wav', 'UrbanSound8K/audio/fold6/4912-3-0-0.wav', 
    'UrbanSound8K/audio/fold7/170243-1-0-0.wav', 'UrbanSound8K/audio/fold7/43784-3-0-0.wav', 'UrbanSound8K/audio/fold8/155313-3-0-0.wav', 
    'UrbanSound8K/audio/fold9/155130-1-0-0.wav', 'UrbanSound8K/audio/fold9/180156-1-12-0.wav', 'UrbanSound8K/audio/fold9/27068-1-0-0.wav', 
    'UrbanSound8K/audio/fold10/17124-1-0-0.wav', 'UrbanSound8K/audio/fold10/26255-3-0-0.wav']



main()