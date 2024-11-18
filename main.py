import librosa
import numpy as np
import matplotlib.pyplot as plt
import pretty_midi
import mir_eval
import os


def DTW(gt_path, eval_path):

    def extract_midi_features(midi_path):
        """
        Extracts onset times and pitches from a MIDI file.
        """
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        pitches = []
        onsets = []

        for instrument in midi_data.instruments:
            for note in instrument.notes:
                pitches.append(note.pitch)  # MIDI pitch (integer)
                onsets.append(note.start)  # Note onset time in seconds

        # Convert to numpy arrays
        pitches = np.array(pitches)
        onsets = np.array(onsets)

        return onsets, pitches

    def compute_dtw_distance(onsets_1, pitches_1, onsets_2, pitches_2):
        """
        Computes DTW distance between two sequences of onsets and pitches.
        """
        # Combine onset times and pitches as features
        sequence1 = np.column_stack((onsets_1, pitches_1))
        sequence2 = np.column_stack((onsets_2, pitches_2))

        # Compute the DTW alignment
        dist, wp = librosa.sequence.dtw(X=sequence1.T, Y=sequence2.T, metric='euclidean')

        # DTW distance (final cost at the optimal path endpoint)
        dtw_distance = dist[-1, -1]

        return dtw_distance, wp

    # Extract features
    onsets1, pitches1 = extract_midi_features(gt_path)
    onsets2, pitches2 = extract_midi_features(eval_path)

    # Compute DTW distance
    return  compute_dtw_distance(onsets1, pitches1, onsets2, pitches2)
    """
    dtw_distance, warping_path = compute_dtw_distance(onsets1, pitches1, onsets2, pitches2)
    """

def mir(gt_path, eval_path):
    def midi_to_intervals_and_pitches(midi_path):

        midi = pretty_midi.PrettyMIDI(midi_path)
        intervals = []
        pitches = []

        for instrument in midi.instruments:
            for note in instrument.notes:
                intervals.append([note.start, note.end])
                pitches.append(note.pitch)

        intervals = np.array(intervals)
        pitches = np.array(pitches)
        return intervals, pitches

    def evaluate_transcription_notes(ref_midi_path, est_midi_path, onset_tolerance=0.05, offset_ratio=0.2):

        # Konwersja MIDI na interwały i wysokości
        ref_intervals, ref_pitches = midi_to_intervals_and_pitches(ref_midi_path)
        est_intervals, est_pitches = midi_to_intervals_and_pitches(est_midi_path)

        # Obliczanie metryk
        metrics = mir_eval.transcription.evaluate(
            ref_intervals=ref_intervals,
            ref_pitches=ref_pitches,
            est_intervals=est_intervals,
            est_pitches=est_pitches,
            onset_tolerance=onset_tolerance,
            offset_ratio=offset_ratio
        )
        return metrics

    metrics = evaluate_transcription_notes(gt_path, eval_path, onset_tolerance=0.05, offset_ratio=0.3)

    # Display results
    print(f"Precision: {metrics['Precision']:.2f}")
    print(f"Recall: {metrics['Recall']:.2f}")
    print(f"F1-Score: {metrics['F-measure']:.2f}")



root_path = 'files_to_evaluate\models'
gt_root_path = 'files_to_evaluate\gt'

list_subfolders_with_paths = [f.path for f in os.scandir(root_path) if f.is_dir()]

for model in list_subfolders_with_paths:
    for file in os.listdir(model):
        if file.endswith(".mid"):
            gt_path = os.path.abspath("files_to_evaluate\gt\\" + file[:-8] + '.mid')
        else:
            gt_path = os.path.abspath("files_to_evaluate\gt\\" + file[:-9] + '.mid')
        eval_path = os.path.abspath(os.path.join(model, file))
        print(gt_path)
        print(eval_path)
        mir(gt_path, eval_path)
