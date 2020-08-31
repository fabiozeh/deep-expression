import re
import math
import sys

import numpy as np
import pandas as pd

#
# Classes
#


class Note:

    C, Db, D, Eb, E, F, Gb, G, Ab, A, Bb, B = range(0, 12)

    def __init__(self, pitch=0, startBeat=0., startTime=0., durBeats=1.,
                 durS=1., prevNote=None, nextNote=None, otherVoices=None,
                 levels=None, timingDev=np.nan, timingDevLocal=np.nan, localTempo=np.nan):
        self.pitch = pitch
        self.startBeat = startBeat
        self.startTime = startTime
        self.durBeats = durBeats
        self.durS = durS
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
    def __init__(self, parts=None, key=Note.C, mode=Mode.major, name='untitled', dynMean=-30, dynStd=7,
                 startTime=-1.0, startBeat=-1.0, endTime=-1.0, endBeat=-1.0):
        if parts is None:
            self.parts = {}
        else:
            self.parts = parts
        self.key = key
        self.mode = mode
        self.name = name
        self.dynMean = dynMean
        self.dynStd = dynStd
        self.startTime = startTime
        self.startBeat = startBeat
        self.endTime = endTime
        self.endBeat = endBeat

    def __getGlobalTempo(self):
        if self.startTime < 0 or self.startBeat < 0 or self.endTime < 0 or self.endBeat < 0:
            self.startTime = np.inf
            self.startBeat = np.inf
            self.endTime = 0.0
            self.endBeat = 0.0
            if not self.parts:
                return np.nan
            for p in self.parts:
                if p[0].startTime < self.startTime:
                    self.startTime = p[0].startTime
                    self.startBeat = p[0].startBeat
                if p[-1].startTime > self.endTime:
                    self.endTime = p[-1].startTime
                    self.endBeat = p[-1].startBeat
        return (self.endBeat - self.startBeat) / (self.endTime - self.startTime)

    globalTempo = property(__getGlobalTempo)


class Dataset:
    def __init__(self, relation='new relation', description='', attributes=None, valsArray=None, moreVals=None):
        self.relation = relation
        self.description = description
        if attributes is None:
            self.attributes = []
        else:
            self.attributes = attributes
        self.valsArray = valsArray
        self.moreVals = moreVals

    def __getData(self):
        return self.valsArray.tolist()

    data = property(__getData)

    def toArffDict(self):
        d = self.__dict__.copy()
        d['data'] = self.data
        del d['valsArray']
        return d

    def toCsv(self, outfile=None):
        fmt = ''
        for _, tp in self.attributes:
            if tp == 'REAL':
                fmt += ',%.4f'
            elif tp == 'INTEGER':
                fmt += ',%d'
            else:
                fmt += ',%s'
        if outfile is None:
            outfile = open(self.relation + '.csv', 'w+')
        np.savetxt(outfile, self.valsArray, fmt=fmt[1:])


#
# Parsing functions
#


def buildNoteParts(notearray, levels, srate, instruments=None):
    parts = {}
    # instruments in this piece
    if instruments is None:
        instruments = set(notearray['instrument'])
    # global tempo calculation
    global_start_time = notearray['start_time'][0] / srate
    global_start_beat = notearray['start_beat'][0]
    global_end_time = notearray['end_time'][-1] / srate
    global_end_beat = notearray['start_beat'][-1] + notearray['end_beat'][-1]
    global_tempo = (global_end_beat - global_start_beat) / (global_end_time - global_start_time)
    for i in instruments:
        X = []  # melody notes array
        running_melody = False
        current_beat = global_start_beat
        running_voices = []
        current_grid_time_global = global_start_time
        current_grid_time_local = global_start_time
        local_tempo_notes = []
        local_tempo = global_tempo
        for entry in notearray:
            beat = entry['start_beat']
            if beat > current_beat:
                if running_melody:
                    X[-1].otherVoices = running_voices
                    running_melody = False
            else:
                if local_tempo_notes:
                    local_tempo_notes.pop(-1)  # exclude simultaneous note

            current_grid_time_global += (beat - current_beat) / global_tempo
            if local_tempo_notes:
                local_tempo_notes = [x for x in local_tempo_notes if x[0] >= local_tempo_notes[-1][0] - 4]  # keep 4 beats before previous note

            if local_tempo_notes and (local_tempo_notes[-1][0] - local_tempo_notes[0][0]) > 1 and (local_tempo_notes[-1][1] - local_tempo_notes[0][1]) > 0:
                local_tempo = (local_tempo_notes[-1][0] - local_tempo_notes[0][0]) / (local_tempo_notes[-1][1] - local_tempo_notes[0][1])
                current_grid_time_local += (beat - local_tempo_notes[-1][0]) / local_tempo
            else:
                # keep the same local_tempo from previous entry.
                current_grid_time_local = entry['start_time'] / srate
            current_beat = beat
            running_voices = [x for x in running_voices if x[0] > current_beat]
            dur = noteValueParser(entry['note_value'])
            if np.isnan(dur):
                dur = entry['end_beat']
            local_tempo_notes.append((entry['start_beat'], entry['start_time'] / srate))

            if entry['instrument'] != i:  # not the desired instrument
                running_voices.append((current_beat + dur, entry['instrument'], entry['note']))
            else:
                # desired instrument
                if running_melody:
                    # polyphony in melody instrument.
                    # Melody = highest pitch
                    if entry['note'] > X[-1].pitch:
                        running_voices.append((current_beat + X[-1].durBeats, i, X[-1].pitch))
                        X[-1].pitch = entry['note']
                        X[-1].startTime = entry['start_time'] / srate
                        X[-1].durS = (entry['end_time'] - entry['start_time']) / srate
                        X[-1].durBeats = dur
                        X[-1].levels = levels[int(entry['start_time'] * 10 // srate):int(entry['end_time'] * 10 // srate + 1)]
                        X[-1].timingDev = current_grid_time_global - X[-1].startTime
                        X[-1].timingDevLocal = current_grid_time_local - X[-1].startTime
                        X[-1].localTempo = local_tempo
                    else:
                        running_voices.append((current_beat + dur, entry['instrument'], entry['note']))
                else:
                    running_melody = True
                    X.append(Note(pitch=entry['note'],
                                  startBeat=current_beat,
                                  startTime=entry['start_time'] / srate,
                                  durS=(entry['end_time'] - entry['start_time']) / srate,
                                  durBeats=dur,
                                  levels=levels[int(entry['start_time'] * 10 // srate):int(entry['end_time'] * 10 // srate + 1)],
                                  timingDev=current_grid_time_global - entry['start_time'] / srate,
                                  timingDevLocal=current_grid_time_local - entry['start_time'] / srate,
                                  localTempo=local_tempo))
                    if len(X) > 1:
                        X[-1].prevNote = X[-2]
                        X[-2].nextNote = X[-1]
            current_grid_time_local = entry['start_time'] / srate
        parts[i] = X
    return parts


def noteValueParser(s):
    tokens = s.split()
    t = tokens[0].lower()
    if 'tied' == t:
        tokens.pop(0)
        v1, tokens = __valueNameParser(tokens)
        v2, tokens = __valueNameParser(tokens)
        return v1 + v2
    else:
        v, tokens = __valueNameParser(tokens)
        return v


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
# Timing calculations
#


def localTempo(notes):
    if not notes:
        return np.nan
    notes.sort(key=(lambda x: x.startTime))
    totalBeats = notes[-1].startBeat - notes[0].startBeat
    totalS = notes[-1].startTime - notes[0].startTime
    if totalBeats <= 1 or totalS <= 0:
        return np.nan
    else:
        return totalBeats / totalS

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


def metricStrength(n):
    eps = 1e-3
    sb = n.startBeat % 4
    if (sb < eps):
        return 3
    elif (abs(sb - 2) < eps):
        return 2
    elif (sb - math.floor(sb) < eps):
        return 1
    else:
        return 0


def toMotifDataframe(piece, instrument, dataframe=None):
    if dataframe is None:
        dataframe = {
            'startTime': [],
            'beatInMeasure': [],
            'metricStrength': [],
            'numberOfNotes': [],
            'duration': [],
            'durationSecs': [],
            'locationInPiece': [],
            'pitchX2': [],
            'pitchX1': [],
            'pitchX0': [],
            'pitchContourX2': [],
            'pitchContourX1': [],
            'pitchContourX0': [],
            'rhythmDrops': [],
            'rhythmRises': [],
            'rhythmContourX2': [],
            'rhythmContourX1': [],
            'rhythmContourX0': [],
            'boundaryValue': [],
            'locationStrongestNote': [],
            'pieceKey': [],
            'pieceMode': [],
            'probChord_I': [],
            'probChord_II': [],
            'probChord_III': [],
            'probChord_IV': [],
            'probChord_V': [],
            'probChord_VI': [],
            'probChord_VII': [],
            'initialChord': [],
            'finalChord': [],
            'hasDissonance': [],
            'dissonanceLocation': [],
            'isSoloPiece': [],
            'pieceId': [],
            'motifId': [],
            'pieceDynMean': [],
            'pieceDynStd': [],
            'dynamicsX2': [],
            'dynamicsX1': [],
            'dynamicsX0': []
        }
    boundaryVals = boundaries(piece.parts[instrument])
    isSolo = not any([len(x.otherVoices) > 0 for x in piece.parts[instrument]])
    motifs = groupMotifs(piece.parts[instrument], bounds=boundaryVals)
    dynamicsLevels = []

    sys.stdout.write('piece=' + piece.name)
    sys.stdout.flush()

#    vals = np.zeros(shape=(len(motifs), len(dataset.attributes)), dtype=object)
#    allLevels = [[], '', -1] * len(motifs)

    startInd = 0
    totalBeats = piece.parts[instrument][-1].startBeat + piece.parts[instrument][-1].durBeats
    for i in range(0, len(motifs)):
        sb = piece.parts[instrument][startInd].startBeat
        st = piece.parts[instrument][startInd].startTime
        incl = 0  # extra included motifs (for dataset augmentation)
        incl_flag = True
        while incl_flag:
            incl_flag = i + incl < len(motifs)
            if incl_flag:
                if motifs[i + incl] < len(piece.parts[instrument]):
                    dur = (piece.parts[instrument][motifs[i + incl]].startBeat - sb)
                else:
                    last = piece.parts[instrument][motifs[i + incl] - 1]
                    dur = last.startBeat + last.durBeats - sb
                incl_flag = incl_flag and (incl == 0 or dur <= 8)
            if incl_flag:
                motif = piece.parts[instrument][startInd:motifs[i + incl]]
                x = np.linspace(0, 1, len(motif))

                dataframe['startTime'].append(st)
                dataframe['beatInMeasure'].append(sb % 4)
                dataframe['metricStrength'].append(metricStrength(motif[0]))
                dataframe['numberOfNotes'].append(motifs[i + incl] - startInd)
                dataframe['duration'].append(dur)
                if motifs[i + incl] < len(piece.parts[instrument]):
                    dataframe['durationSecs'].append(piece.parts[instrument][motifs[i + incl]].startTime - st)
                else:
                    last = piece.parts[instrument][motifs[i + incl] - 1]
                    dataframe['durationSecs'].append(last.startTime + last.durS - st)
                dataframe['locationInPiece'].append(sb / totalBeats)

                # pitch curve coefficients:
                pitches = np.array([n.pitch for n in motif])
                coeff = np.polyfit(x, pitches - min(pitches), 2 if len(x) > 2 else 1)
                coeff = coeff if len(x) > 2 else [0, coeff[0], coeff[1]]
                dataframe['pitchX2'].append(coeff[0])
                dataframe['pitchX1'].append(coeff[1])
                dataframe['pitchX0'].append(coeff[2])

                pContour = pitchContour(motif)
                coeff = np.polyfit(x, pContour, 2 if len(x) > 2 else 1)
                coeff = coeff if len(x) > 2 else [0, coeff[0], coeff[1]]
                dataframe['pitchContourX2'].append(coeff[0])
                dataframe['pitchContourX1'].append(coeff[1])
                dataframe['pitchContourX0'].append(coeff[2])

                # phrase separation criteria
                dataframe['boundaryValue'].append(boundaryVals[startInd])

                # rhythmic descriptors
                rhContour = np.array([n.ioiBeats / n.prevNote.ioiBeats if n.prevNote is not None else 1 for n in motif])
                dataframe['rhythmDrops'].append(max(rhContour) > 1)
                dataframe['rhythmRises'].append(min(rhContour) < 1)

                # rhythmic contour coefficients
                coeff = np.polyfit(x, rhContour, 2 if len(x) > 2 else 1)
                coeff = coeff if len(x) > 2 else [0, coeff[0], coeff[1]]
                dataframe['rhythmContourX2'].append(coeff[0])
                dataframe['rhythmContourX1'].append(coeff[1])
                dataframe['rhythmContourX0'].append(coeff[2])

                ms = [''] * len(motif)
                for j, nt in enumerate(motif):
                    ms[j] = metricStrength(nt)
                dataframe['locationStrongestNote'].append((motif[np.argmax(ms)].startBeat - sb) / (motif[-1].startBeat - sb))

                dataframe['pieceKey'].append(keyInFifths(piece.key))
                dataframe['pieceMode'].append('Major' if piece.mode == Mode.major else 'Minor')

                # chord probabilities
                _, probs = estimateChord(np.asarray([n.allPitches for n in motif]).flatten(), piece.key, piece.mode)
                dataframe['probChord_I'].append(probs[0])
                dataframe['probChord_II'].append(probs[1])
                dataframe['probChord_III'].append(probs[2])
                dataframe['probChord_IV'].append(probs[3])
                dataframe['probChord_V'].append(probs[4])
                dataframe['probChord_VI'].append(probs[5])
                dataframe['probChord_VII'].append(probs[6])

                # initial / final chord
                ch, _ = estimateChord(motif[0].allPitches, piece.key, piece.mode)
                dataframe['initialChord'].append(str(ch))
                ch, _ = estimateChord(motif[-1].allPitches, piece.key, piece.mode)
                dataframe['finalChord'].append(str(ch))

                # hasDissonance / dissonanceLocation
                dataframe['hasDissonance'].append(False)
                dataframe['dissonanceLocation'].append(-1)
                for j in range(0, len(motif)):
                    if isDissonance(motif[j].pitch, piece.key, piece.mode):
                        dataframe['hasDissonance'][-1] = True
                        dataframe['dissonanceLocation'][-1] = x[j]
                        break

                dataframe['isSoloPiece'].append(isSolo)
                dataframe['pieceId'].append(int(piece.name.split('.')[0]))
                dataframe['motifId'].append(len(dynamicsLevels))
                dataframe['pieceDynMean'].append(piece.dynMean)
                dataframe['pieceDynStd'].append(piece.dynStd)

                # dynamics curve coefficients
                levels = []
                coeff = [0] * 3
                for n in motif:
                    levels += list(n.levels)
                if len(levels) > 2:
                    coeff = np.polyfit(np.linspace(0, 1, len(levels)), levels, 2)
                elif len(levels) > 1:
                    coeff[0] = 0
                    coeff[1:] = np.polyfit(np.linspace(0, 1, len(levels)), levels, 1)
                elif len(levels) == 1:
                    coeff[0:1] = [0, 0]
                    coeff[2] = levels[0]
                else:
                    coeff[0:1] = [0, 0]
                    coeff[2] = - np.inf
                dataframe['dynamicsX2'].append(coeff[0])
                dataframe['dynamicsX1'].append(coeff[1])
                dataframe['dynamicsX0'].append(coeff[2])

                # timing deviation curve descriptors (TODO)

                # storing levels for error calculation
                dynamicsLevels.append(levels)

                incl += 1

        # update start index
        startInd = motifs[i]

    print(', {m} motifs'.format(m=len(dataframe['startTime'])))
    return dataframe, dynamicsLevels


def buildNoteLevelDataframe(piece, instrument, include_instrument_col=True, transpose=0):
    df = {
        'pitch': [],
        'probChord_I': [],
        'probChord_II': [],
        'probChord_III': [],
        'probChord_IV': [],
        'probChord_V': [],
        'probChord_VI': [],
        'probChord_VII': [],
        "duration": [],
        "ioi": [],
        "metricStrength": [],
        "bassNote": [],
        "isDissonance": [],
        "startTime": [],
        "durationSecs": [],
        "timingDev": [],
        "timingDevLocal": [],
        "localTempo": [],
        "peakLevel": [],
    }
    if include_instrument_col:
        df['instrument'] = []

    for n in piece.parts[instrument]:
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
        df['metricStrength'].append(metricStrength(n))
        bass = 1000
        for (_, _, v) in n.otherVoices:
            bass = v + transpose if v + transpose < bass else bass
        df['bassNote'].append(bass % 12)
        df['isDissonance'].append(isDissonance(n.pitch + transpose, piece.key + transpose, piece.mode))
        df['startTime'].append(n.startTime)
        df['durationSecs'].append(n.durS)
        df['timingDev'].append(n.timingDev)
        df['timingDevLocal'].append(n.timingDevLocal)
        df['localTempo'].append(n.localTempo)
        peak = -1000
        for lvl in n.levels:
            peak = lvl if lvl > peak else peak
        df['peakLevel'].append(peak)
        if include_instrument_col:
            df['instrument'].append(instrument)

    df = pd.DataFrame(data=df)

    #  span of valid midi notes for relevant instruments
    #  df['pitch'] = df['pitch'].astype(pd.CategoricalDtype(list(range(36, 109))))

    #  pitches in an octave
    df['bassNote'] = df['bassNote'].astype(pd.CategoricalDtype(list(range(0, 12))))

    #  valid metric strength values
    df['metricStrength'] = df['metricStrength'].astype(pd.CategoricalDtype(list(range(0, 4))))

    # #  one-hot encode nominal values
    # for attrib in ['metricStrength', 'pitch', 'bassNote']:
    #     df = pd.concat([df, pd.get_dummies(df[attrib], prefix=attrib)], axis=1)
    #     df.drop([attrib], axis=1, inplace=True)

    return df
