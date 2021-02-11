import re
import math

import numpy as np
import pandas as pd
import pretty_midi


#
# Classes
#


class Note:

    C, Db, D, Eb, E, F, Gb, G, Ab, A, Bb, B = range(0, 12)

    def __init__(self, pitch=0, startBeat=0., startTime=0., durBeats=1.,
                 durS=1., instrument=1, prevNote=None, nextNote=None, otherVoices=None,
                 levels=None, timingDev=np.nan, timingDevLocal=np.nan, localTempo=np.nan):
        self.pitch = pitch
        self.startBeat = startBeat
        self.startTime = startTime
        self.durBeats = durBeats
        self.durS = durS
        self.instrument = instrument
        self.prevNote = prevNote
        self.nextNote = nextNote
        if otherVoices is None:
            self.otherVoices = []
        else:
            self.otherVoices = otherVoices
        if levels is None:
            self.levels = []
        else:
            self.levels = levels
        self.timingDev = timingDev
        self.timingDevLocal = timingDevLocal
        self.localTempo = localTempo

    def interval(self, note):
        if note is None:
            return 0
        else:
            return self.pitch - note.pitch

    def gridOnset(self, piece_tempo, piece_start_time, piece_start_beat):
        return piece_start_time + (self.startBeat - piece_start_beat) / piece_tempo

    def __getIoiBeats(self):
        if self.nextNote is None:
            return self.durBeats
        else:
            return self.nextNote.startBeat - self.startBeat

    def __getIoiS(self):
        if self.nextNote is None:
            return self.durS
        else:
            return self.nextNote.startTime - self.startTime

    def __getPitchInOctave(self):
        return self.pitch % 12

    def __getAllPitches(self):
        return np.array([o[2] for o in self.otherVoices] + [float(self.pitch)])

    ioiBeats = property(__getIoiBeats)

    ioiS = property(__getIoiS)

    pitchInOctave = property(__getPitchInOctave)

    allPitches = property(__getAllPitches)


class Mode:
    major, dorian, frigian, lydian, mixolydian, minor, locrian = range(0, 7)


class Piece:
    def __init__(self, part=None, key=Note.C, mode=Mode.major, name='untitled', beats_per_measure=0,
                 time_sig_type=4, first_down_beat=0, dynMean=-30, dynStd=7,
                 startTime=-1.0, startBeat=-1.0, endTime=-1.0, endBeat=-1.0):
        if part is None:
            self.part = []
        else:
            self.part = part
        self.key = key
        self.mode = mode
        self.name = name
        self.beats_per_measure = beats_per_measure
        self.time_sig_type = time_sig_type
        self.first_down_beat = first_down_beat
        self.dynMean = dynMean
        self.dynStd = dynStd
        self.startTime = startTime
        self.startBeat = startBeat
        self.endTime = endTime
        self.endBeat = endBeat


#
# Parsing functions
#

def buildPart(notearray, levels, srate):
    part = []
    # global tempo calculation
    global_start_time = notearray['start_time'][0] / srate
    global_start_beat = notearray['start_beat'][0]
    global_end_time = notearray['end_time'][-1] / srate
    global_end_beat = notearray['start_beat'][-1] + notearray['end_beat'][-1]
    global_tempo = (global_end_beat - global_start_beat) / (global_end_time - global_start_time)

    # time trackers
    current_beat = global_start_beat
    current_grid_time_global = global_start_time  # expected time in s at global tempo
    current_grid_time_local = global_start_time  # time in s synced to last onset and incremented at local tempo
    local_tempo_notes = []  # most recent onsets (0 = beat and 1 = s) for computing local tempo
    local_tempo = global_tempo  # tempo as moving average according to onsets in the 4 beats leading up to prev note

    running_voices = []  # the set of notes ringing at the current beat
    polyphony = []  # simultaneous notes buffer for properly setting running_voices

    for entry in notearray:
        beat = entry['start_beat']
        if beat > current_beat:
            # set running voices for all notes in polyphony buffer
            for n in polyphony:
                n.otherVoices = [x for x in running_voices if x[2] != n.pitch and x[1] != n.instrument]
            polyphony = []
        else:
            if local_tempo_notes:
                local_tempo_notes.pop(-1)  # exclude simultaneous note

        current_grid_time_global += (beat - current_beat) / global_tempo
        if local_tempo_notes:
            local_tempo_notes = [x for x in local_tempo_notes if x[0] >= local_tempo_notes[-1][0] - 4]  # keep 4 beats before previous note

        # if there's more than 1 beat logged, update local_tempo
        if local_tempo_notes and (local_tempo_notes[-1][0] - local_tempo_notes[0][0]) > 1 and (local_tempo_notes[-1][1] - local_tempo_notes[0][1]) > 0:
            local_tempo = (local_tempo_notes[-1][0] - local_tempo_notes[0][0]) / (local_tempo_notes[-1][1] - local_tempo_notes[0][1])

        current_grid_time_local += (beat - current_beat) / local_tempo

        current_beat = beat
        dur = entry['end_beat']
        local_tempo_notes.append((entry['start_beat'], entry['start_time'] / srate))
        running_voices = [x for x in running_voices if x[0] > current_beat]

        this_note = Note(pitch=entry['note'],
                         instrument=entry['instrument'],
                         startBeat=current_beat,
                         startTime=entry['start_time'] / srate,
                         durS=(entry['end_time'] - entry['start_time']) / srate,
                         durBeats=dur,
                         levels=levels[int(entry['start_time'] * 10 // srate):int(entry['end_time'] * 10 // srate + 1)],
                         timingDev=current_grid_time_global - entry['start_time'] / srate,
                         timingDevLocal=current_grid_time_local - entry['start_time'] / srate,
                         localTempo=local_tempo)
        if part:
            this_note.prevNote = part[-1]
            part[-1].nextNote = this_note
        part.append(this_note)
        polyphony.append(this_note)
        running_voices.append((current_beat + dur, this_note.instrument, this_note.pitch))

        current_grid_time_local = entry['start_time'] / srate

    for n in polyphony:
        n.otherVoices = [x for x in running_voices if x[2] != n.pitch and x[1] != n.instrument]

    return part


def noteValueParser(s, quarter_value):
    tokens = s.split()
    t = tokens[0].lower()
    if 'tied' == t:
        tokens.pop(0)
        v1, tokens = __valueNameParser(tokens)
        v2, tokens = __valueNameParser(tokens)
        return (v1 + v2) * quarter_value
    else:
        v, tokens = __valueNameParser(tokens)
        return v * quarter_value


def __valueNameParser(tokens):
    dotted = False
    dash = False
    triplet = False
    faceValue = 0.5
    while tokens:
        t = tokens.pop(0).lower()
        if 'dotted' == t:
            dotted = True
            continue
        elif re.search(r'-', t):
            tk = t.split('-')
            tokens.insert(0, tk[1])
            tokens.insert(0, tk[0])
            dash = True
            continue
        elif 'triplet' == t:
            triplet = True
            if dash:
                break
            else:
                continue
        elif 'whole' == t:
            faceValue = 4.0
            break
        elif 'half' == t:
            faceValue = 2.0
            break
        elif 'quarter' == t:
            faceValue = 1.0
            break
        elif 'eighth' == t:
            faceValue = 0.5
            break
        elif 'sixteenth' == t:
            faceValue = 0.25
            break
        elif 'thirty' == t:
            faceValue = 0.125
            continue
        elif 'sixty' == t:
            faceValue = 0.0625
            break
        elif 'second' == t:
            break
        elif 'fourth' == t:
            break
        else:  # unknown
            faceValue = np.nan
            break
    if triplet:
        faceValue = faceValue / 0.5 * 0.33
    if dotted:
        faceValue += faceValue * 0.5
    return faceValue, tokens


#
# Harmonic Functions
#


def estimateKey(pitches):
    # D. Temperley pitch profiles
    major = np.array([0.748, 0.060, 0.488, 0.082, 0.67, 0.46, 0.096, 0.715, 0.104, 0.366, 0.057, 0.4])
    minor = np.array([0.712, 0.084, 0.474, 0.618, 0.049, 0.46, 0.105, 0.747, 0.404, 0.067, 0.133, 0.33])
    hist = np.zeros(shape=(12, 1))
    for p in pitches:
        hist[int(p % 12)] += 1
    llhoodM = [0] * 12  # likelihoods for each major key starting in C
    llhoodm = [0] * 12  # likelihoods for each minor key starting in Cm
    for i in range(0, 12):
        llhoodM[i] = np.dot(np.roll(major, i), hist)[0] / len(pitches)
        llhoodm[i] = np.dot(np.roll(minor, i), hist)[0] / len(pitches)
    maxM = max(llhoodM)
    maxm = max(llhoodm)
    return maxM > maxm, llhoodM.index(maxM) if maxM > maxm else llhoodm.index(maxm), llhoodM, llhoodm


def keyInFifths(key):
    lookup = [0, -5, 2, -3, 4, -1, 6, 1, -4, 3, -2, 5]
    return lookup[key]


def estimateChord(pitches, key, mode):
    lookup = [0, 2, 4, 5, 7, 9, 11]
    scale = (key - lookup[mode] + 12) % 12
    pitches = (pitches - scale + 12) % 12
    candidates = np.array([
        np.sum(np.isin([0, 4, 7], pitches)) / len(pitches),
        np.sum(np.isin([2, 5, 9], pitches)) / len(pitches),
        np.sum(np.isin([4, 7, 8, 11], pitches)) / len(pitches),  # 8 from harmonic minor implies V/vi
        np.sum(np.isin([5, 9, 0], pitches)) / len(pitches),
        np.sum(np.isin([7, 11, 2], pitches)) / len(pitches),
        np.sum(np.isin([9, 0, 4], pitches)) / len(pitches),
        np.sum(np.isin([11, 2, 5], pitches)) / len(pitches)]
    )
    candidates = np.roll(candidates, 7 - mode)
    return np.argmax(candidates), np.exp(candidates) / sum(np.exp(candidates))


def bassNote(pitches):
    return np.min(pitches) % 12


def isDissonance(pitch, key, mode):
    return dissonancePresent([pitch], key, mode)


def dissonancePresent(pitches, key, mode):
    lookup = [0, 2, 4, 5, 7, 9, 11]
    scale = (key - lookup[mode] + 12) % 12
    pitches = [(p - scale + 12) % 12 for p in pitches]
    return not np.all(np.isin(pitches, lookup))


#
# Phrase recognition
#


def boundaries(part):
    """ Boundary probabilitites according to Emilios Cambouropoulos'
        Local Boundary Detection Model

    """
    eps = 1e-6
    s = len(part) - 1
    pp = np.zeros((s, 3))  # pitch profile
    po = np.zeros((s, 3))  # ioi profile
    pr = np.zeros((s, 3))  # rest profile

    for i in range(0, s):
        n = part[i]
        pp[i, 0] = abs(n.nextNote.pitch - n.pitch)
        po[i, 0] = n.ioiBeats
        pr[i, 0] = max(0, n.nextNote.startBeat - (n.startBeat + n.durBeats))

    # degrees of change
    pp[:-1, 1] = abs(pp[1:, 0] - pp[0:-1, 0]) / (eps + pp[1:, 0] + pp[0:-1, 0])
    po[:-1, 1] = abs(po[1:, 0] - po[0:-1, 0]) / (eps + po[1:, 0] + po[0:-1, 0])
    pr[:-1, 1] = abs(pr[1:, 0] - pr[0:-1, 0]) / (eps + pr[1:, 0] + pr[0:-1, 0])

    # strengths
    pp[1:, 2] = pp[0:-1, 1] + pp[1:, 1]
    pp[:, 2] = pp[:, 1] * pp[:, 2]
    pp[:, 2] = pp[:, 2] / max(pp[:, 2]) if max(pp[:, 2]) > 0.1 else pp[:, 2]

    po[1:, 2] = po[0:-1, 1] + po[1:, 1]
    po[:, 2] = po[:, 1] * po[:, 2]
    po[:, 2] = po[:, 2] / max(po[:, 2]) if max(po[:, 2]) > 0.1 else po[:, 2]

    pr[1:, 2] = pr[0:-1, 1] + pr[1:, 1]
    pr[:, 2] = pr[:, 1] * pr[:, 2]
    pr[:, 2] = pr[:, 2] / max(pr[:, 2]) if max(pr[:, 2]) > 0.1 else pr[:, 2]

    ret = np.ones(s + 1)
    ret[1:] = 0.25 * pp[:, 2] + 0.5 * po[:, 2] + 0.25 * pr[:, 2]
    return ret


def groupMotifs(part, bounds=None, offset=0):
    if bounds is None:
        bounds = boundaries(part)
    max_notes = 32
    ind = 0
    if (len(part) <= max_notes / 2):
        ind = 0  # no split
    else:
        z = np.asanyarray(bounds[2:-2])
        std = z.std(ddof=1)
        if std != 0:
            z = (z - z.mean()) / std
            ind = np.argmax(z)
            ind = ind + 2 if z[ind] > 2 or len(part) > max_notes else 0
    if ind == 0:
        return [len(part) + offset]
    else:
        return (groupMotifs(part[0:ind], bounds[0:ind], offset)
                + groupMotifs(part[ind:], bounds[ind:], offset + ind))


#
# Motif features
#


def pitchContour(part):
    return [p.interval(p.prevNote) for p in part]


def normalizedBeats(part):
    st = part[0].startBeat
    end = part[-1].startBeat - st
    return [(p.startBeat - st) / end for p in part]


def metricStrength(n, beats_per_measure, anacrusis_beats):
    eps = 1e-3
    sb = (n.startBeat + anacrusis_beats) % beats_per_measure
    if (sb < eps):
        return 3
    elif (beats_per_measure == 4 and abs(sb - 2) < eps):
        return 2
    elif (sb - math.floor(sb) < eps):
        return 1
    else:
        return 0


def buildNoteLevelDataframe(piece, transpose=0):
    df = {
        'beatDiff': [],         # beats since last onset
        'instrument': [],       # MIDI number of instrument
        'pitch': [],            # MIDI pitch of instrument
        'probChord_I': [],      # see estimateChord function
        'probChord_II': [],     # see estimateChord function
        'probChord_III': [],    # see estimateChord function
        'probChord_IV': [],     # see estimateChord function
        'probChord_V': [],      # see estimateChord function
        'probChord_VI': [],     # see estimateChord function
        'probChord_VII': [],    # see estimateChord function
        "duration": [],         # note duration in beats
        "ioi": [],              # beats since last note of this instrument
        "bassNote": [],         # lowest sounding pitch (in 8ve) at onset time
        "harmony": [],          # binary array encoding all other sounding tones (in 8ve) at onset time
        "isDissonance": [],     # True if pitch not in piece.mode scale
        "startTime": [],        # onset time in s
        "durationSecs": [],     # note duration in s
        "ioiRatio": [],         # ioi in s / ioi in beats (local tempo estimate)
        "timingDev": [],        # total onset deviation time assuming steady tempo
        "timingDevLocal": [],   # onset deviation time from local tempo
        "localTempo": [],       # moving average of tempo before onset
        "peakLevel": [],        # peak performance dBu within note duration
    }
    hasMeasures = piece.beats_per_measure != 0  # if there are measure divisions
    if hasMeasures:
        df['metricStrength'] = []   # see metricStrength function

    for i, n in enumerate(piece.part):
        if i > 0:
            df['beatDiff'].append(n.startBeat - piece.part[i - 1].startBeat)
        else:
            df['beatDiff'].append(n.startBeat)
        df['instrument'].append(n.instrument)
        df['pitch'].append(n.pitch + transpose)
        _, probs = estimateChord(np.asarray(n.allPitches).flatten() + transpose, piece.key + transpose, piece.mode)
        df['probChord_I'].append(probs[0])
        df['probChord_II'].append(probs[1])
        df['probChord_III'].append(probs[2])
        df['probChord_IV'].append(probs[3])
        df['probChord_V'].append(probs[4])
        df['probChord_VI'].append(probs[5])
        df['probChord_VII'].append(probs[6])
        df['duration'].append(n.durBeats)
        df['ioi'].append(n.ioiBeats)
        bass = 1000
        for (_, _, v) in n.otherVoices:
            bass = v + transpose if v + transpose < bass else bass
        df['bassNote'].append(bass % 12)
        df['harmony'].append(np.sum([1 << k for k in set([v[2] % 12 for v in n.otherVoices])]))
        df['isDissonance'].append(isDissonance(n.pitch + transpose, piece.key + transpose, piece.mode))
        df['startTime'].append(n.startTime)
        df['durationSecs'].append(n.durS)
        df['ioiRatio'].append(n.ioiS / (n.ioiBeats + 1e-6))
        df['timingDev'].append(n.timingDev)
        df['timingDevLocal'].append(n.timingDevLocal)
        df['localTempo'].append(np.log(n.localTempo))
        peak = -1000
        for lvl in n.levels:
            peak = lvl if lvl > peak else peak
        df['peakLevel'].append(peak)
        if hasMeasures:
            df['metricStrength'].append(metricStrength(n, piece.beats_per_measure, piece.first_down_beat))

    df = pd.DataFrame(data=df)

    #  span of valid midi notes for relevant instruments
    #  df['pitch'] = df['pitch'].astype(pd.CategoricalDtype(list(range(36, 109))))

    #  pitches in an octave
    df['bassNote'] = df['bassNote'].astype(pd.CategoricalDtype(list(range(0, 12))))

    # #  one-hot encode nominal values
    # for attrib in ['metricStrength', 'pitch', 'bassNote']:
    #     df = pd.concat([df, pd.get_dummies(df[attrib], prefix=attrib)], axis=1)
    #     df.drop([attrib], axis=1, inplace=True)

    return df


def midi_performance(test, prediction, moments, ix_to_lex, method='ioiRatio'):
    """
    Returns a pretty_midi object with a performance generated according to the
    given numpy array of performance actions. Method specifies which measurement
    of tempo and timing deviations was used.
    """
    tempo = math.exp(test[1].localTempo.iloc[0] * test[2][0, 1] + test[2][0, 0])
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(1, is_drum=False, name='piano')
    violin = pretty_midi.Instrument(41, is_drum=False, name='violin')
    pm.instruments.append(piano)
    pm.instruments.append(violin)

    if method == 'ioiRatio':
        # for now, ratio calculated wrt next note of same instrument, which is
        # inconvenient and may produce unsynced performances. Also, durations
        # are copied from reference performance (since the model wasn't trained
        # on articulation).
        ioiRatio = prediction * test[2][2, 1] + test[2][2, 0]

        ioi = 0
        start = 0.
        for x, y, dev in zip(test[0].itertuples(), test[1].itertuples(), ioiRatio):
            pitch = ix_to_lex.get(x.pitch)
            if pitch:
                pitch = pitch[0]
                start += (x.beatDiff * moments['beatDiff'][1] + moments['beatDiff'][0] + 1e-6) * ioi
                end = start + (y.durationSecs * moments['durationSecs'][1] + moments['durationSecs'][0])
                if x.instrument_1:
                    piano.notes.append(pretty_midi.Note(100, pitch, start, end))
                else:
                    violin.notes.append(pretty_midi.Note(100, pitch, start, end))
                ioi = dev

    elif method == 'timingDevLocal':
        timingDevLocal = prediction[:, 0] * moments['timingDevLocal'][1] + moments['timingDevLocal'][0]
        localTempo = prediction[:, 1] * test[2][0, 1] + test[2][0, 0]

        for x, y, dev, loc_tmp in zip(test[0].itertuples(), test[1].itertuples(), timingDevLocal, localTempo):
            pitch = ix_to_lex.get(x.pitch)
            if pitch:
                pitch = pitch[0]
                start += ((x.beatDiff * moments['beatDiff'][1] + moments['beatDiff'][0]) / (1e-6 + math.exp(loc_tmp))
                          - (dev * moments['timingDevLocal'][1] + moments['timingDevLocal'][0]))
                end = start + (x.duration * moments['duration'][1] + moments['duration'][0]) / (1e-6 + math.exp(loc_tmp))
                note = pretty_midi.Note(100, pitch, start, end)
                if x.instrument_1:
                    piano.notes.append(note)
                else:
                    violin.notes.append(note)
    else:
        raise ValueError("Unknown method")

    return pm
