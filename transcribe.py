import torch
from pydub import AudioSegment
import os
from lang import get_text as tx
from PyQt6.QtCore import QThread, pyqtSignal
from pathlib import Path
from faster_whisper import WhisperModel

class TranscriptionWorker(QThread):
    transcription_complete = pyqtSignal(tuple)
    error_occurred = pyqtSignal(str)
    initialize_progressbar = pyqtSignal(float)
    update_progressbar = pyqtSignal(float)

    def __init__(self, filename_path, translation, diarization, accuracy, device):
        super().__init__()
        self.filename_path = filename_path
        self.translation = translation
        self.diarization = diarization
        self.accuracy = accuracy
        self.device = device

    def run(self):
        try:
            if not self.filename_path:
                raise Exception(tx("No_file_selected"))

            # Set the current working directory to where the file is located
            os.chdir(self.filename_path.parent)
            filename = self.filename_path.name

            # Check if the file exists and is not empty
            if not os.path.exists(filename):
                raise Exception(tx("File_not_found"))

            filesize = Path(filename).stat().st_size
            if filesize == 0:
                self.error_occurred.emit(tx("No_sound_found"))
                return

            # Perform the transcription
            transcribe_audio(self)

        except ValueError as e:
            #self.error_occurred.emit(tx("No_sound_text")) # No sound is only one possible source of the issue.
            self.error_occurred.emit(tx("Transcription_error") + str(e))
        except Exception as e:
            print(e)
            self.error_occurred.emit(tx("Transcription_error") + str(e))

def transcribe_audio(self):
    """
    This function turns a m4a-file into mp3 if the supplied file is a m4a-file. It then transcribes the audio file to a txt-file.
    Transcribe the given audio file and return the transcribed text:

    :param filename: Path to the audio file.
    :param translation: Boolean indicating if translation is required.
    :param accuracy: Accuracy level (1-5).
    :param device: Device to use for transcription.
    :return: Transcribed text.
    """
    # Get length of audio file:
    filename = self.filename_path.name
    audio_segment = AudioSegment.from_file(filename)
    duration = audio_segment.duration_seconds

    # Convert m4a to mp3 if necessary
    if filename.endswith(".m4a"):
        filename = filename.replace(".m4a", ".mp3")
        audio_segment.export(filename, format="mp3")

    # Set length of audio as max value for the progress bar:
    self.initialize_progressbar.emit(duration)
    # Start at 0 - this sets the bar back to 0 when a new transcription is started after it was at 100% from a previous transcription.
    self.update_progressbar.emit(0) 

    # Set up information needed to use Whisper
    accuracy = self.accuracy
    if accuracy == 1:
        model_size = "tiny"
    elif accuracy == 2:
        model_size = "small"
    elif accuracy == 3:
        model_size = "medium" 
    elif accuracy == 4:
        model_size = "large-v3"
    elif accuracy == 5:
        model_size = "large-v3-turbo"

    # Transcribe the audio
    task = "translate" if self.translation else "transcribe"
    # Use float16 if the computer supports it
    if torch.cuda.is_available():
        model = WhisperModel(model_size, device=self.device, compute_type="float16")
    else:
        model = WhisperModel(model_size, device=self.device, compute_type="float32")
    # Convert m4a to mp3 if necessary
    filename = self.filename_path.name
    if filename.endswith(".m4a"):
        audio_segment = AudioSegment.from_file(filename)
        filename = filename.replace(".m4a", ".mp3")
        audio_segment.export(filename, format="mp3")
    segments, info = model.transcribe(filename, beam_size=5, task=task)


    try:# Save the transcribed text - should I move this to the transcribe function?
        with open(f'{self.filename_path.stem}.txt', 'w', encoding="utf-8") as f:
            #total = len(list(segments))
            print("Info:", info.duration)
            total = info.duration
            self.initialize_progressbar.emit(total)
            # Set that as max value.
            for i, segment in enumerate(segments):
                f.write(segment.text)
                print("Transcribed segment:", i)
                # Set that as current value with each iteration:
                print(segment.end)
                print(type(segment.end))
                self.update_progressbar.emit(segment.end)
        self.transcription_complete.emit((tx("Transcription_message_1"), self.filename_path.stem, tx("Transcription_message_2")))
        # Finish the progressbar if it hasn't finished yet - that can happen if the last part of the audio doesn't contain speech.
        self.update_progressbar.emit(duration)
    except Exception as e:
        print(e)
        self.error_occurred.emit(tx("Transcription_error") + str(e))
    return 