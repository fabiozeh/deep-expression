import re
import math

import numpy as np

#
# Classes
#


class Note:

    C, Db, D, Eb, E, F, Gb, G, Ab, A, Bb, B = range(0, 12)

    def __init__(self, pitch=0, startBeat=0., startTime=0., durBeats=1.,
                 durS=1., prevNote=None, nextNote=None, otherVoices=None,
                 levels=None, timingDev=np.nan):
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

    def interval(self, note):
        if note is None:
            return 0
        else:
            return self.pitch - note.pitch

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
    def __init__(self, parts=None, key=Note.C, mode=Mode.major, name='untitled'):
        if parts is None:
            self.parts = {}
        else:
            self.parts = parts
        self.key = key
        self.mode = mode
        self.name = name


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


def buildNoteParts(notearray, levels, srate):
    parts = {}
    # instruments in this piece
    instruments = set(notearray['instrument'])
    for i in instruments:
        X = []  # melody notes array
        runningMelody = False
        currentBeat = 0.
        runningVoices = np.empty(shape=(1, 3))

        for entry in notearray:
            beat = entry['start_beat']
            if beat > currentBeat and runningMelody:
                X[-1].otherVoices = runningVoices
                runningMelody = False
            currentBeat = beat
            runningVoices = runningVoices[runningVoices[:, 0] > currentBeat, :]
            dur = noteValueParser(entry['note_value'])
            if entry['instrument'] != i:  # not the desired instrument
                runningVoices = np.append(runningVoices, [[currentBeat + dur, entry['instrument'], entry['note']]], axis=0)
            else:
                # desired instrument
                if runningMelody:
                    # polyphony in melody instrument.
                    # Melody = highest pitch
                    if entry['note'] > X[-1].pitch:
                        runningVoices = np.append(runningVoices, [[currentBeat + X[-1].durBeats, 41, X[-1].pitch]], axis=0)
                        X[-1].pitch = entry['note']
                        X[-1].startTime = entry['start_time'] / srate
                        X[-1].durS = (entry['end_time'] - entry['start_time']) / srate
                        X[-1].durBeats = dur
                        X[-1].levels = levels[int(entry['start_time'] * 10 // srate):int(entry['end_time'] * 10 // srate)]
                    else:
                        runningVoices = np.append(runningVoices, [[currentBeat + dur, entry['instrument'], entry['note']]], axis=0)
                else:
                    runningMelody = True
                    X.append(Note(pitch=entry['note'],
                                  startBeat=currentBeat,
                                  startTime=entry['start_time'] / srate,
                                  durS=(entry['end_time'] - entry['start_time']) / srate,
                                  durBeats=dur,
                                  levels=levels[int(entry['start_time'] * 10 // srate):int(entry['end_time'] * 10 // srate)]))
                    if len(X) > 1:
                        X[-1].prevNote = X[-2]
                        X[-2].nextNote = X[-1]
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
        np.sum(np.isin([4, 7, 11], pitches)) / len(pitches),
        np.sum(np.isin([5, 9, 0], pitches)) / len(pitches),
        np.sum(np.isin([7, 11, 2], pitches)) / len(pitches),
        np.sum(np.isin([9, 0, 4], pitches)) / len(pitches),
        np.sum(np.isin([11, 2, 5], pitches)) / len(pitches)]
    )
    candidates = np.roll(candidates, 7 - mode)
    return np.argmax(candidates), candidates


def isDissonance(pitch, key, mode):
    lookup = [0, 2, 4, 5, 7, 9, 11]
    scale = (key - lookup[mode] + 12) % 12
    pitch = (pitch - scale + 12) % 12
    return not np.isin(pitch, lookup).item()

#
# Timing calculations
#


def medianTempo(notes):
    localT = [n.ioiBeats / n.ioiS
              for n in notes if n.durBeats > 0.125]  # thirty-seconds and faster disconsidered
    localT.sort()
    mid = len(localT) // 2
    return (localT[mid] + localT[~mid]) / 2


def meanTempo(notes):
    localT = [n.ioiBeats / n.ioiS
              for n in notes if n.durBeats > 0.125]  # thirty-seconds and faster disconsidered
    if len(localT) == 0:
        return np.nan
    return sum(localT) / len(localT)


def computeTimingDev(piece):
    """ Timing deviation for a note:
    - compute mean tempo for all notes in the 4 beats prior to onset.
    - using the onset time of this instrument's previous note and the
    mean tempo, predict onset for this note.
    - timingDev = difference between predicted and real onset time.

    """
    allnotes = [n for v in piece.parts.values() for n in v]
    for note in allnotes:
        if note.prevNote is None:
            continue
        localT = meanTempo([n for n in allnotes if n.startBeat < note.startBeat and
                            n.startBeat >= note.startBeat - 4])
        predOnset = note.prevNote.startTime + \
            (note.startBeat - note.prevNote.startBeat) / localT
        note.timingDev = note.startTime - predOnset


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
    max_notes = 10
    ind = 0
    if (len(part) <= max_notes):
        ind = 0  # no split
    else:
        z = np.asanyarray(bounds[2:-2])
        std = z.std(ddof=1)
        if std != 0:
            z = (z - z.mean()) / std
            ind = np.argmax(z)
            ind = ind + 2 if z[ind] > 2 else 0
    if ind == 0:
        return [len(part) + offset]
    else:
        return (groupMotifs(part[0:ind], bounds[0:ind], offset) +
                groupMotifs(part[ind:], bounds[ind:], offset + ind))


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
        return '3'
    elif (sb - 2 < eps):
        return '2'
    elif (sb - math.floor(sb) < eps):
        return '1'
    else:
        return '0'


def toMotifDataset(piece, instrument, dataset=None):
    '''  Generates or appends to an ARFF object with data from the piece:
    start beat mod 4 (can we do time signature discovery?)
    metric strength of first note (sergio's scale)
    number of notes
    motif duration (beats)
    start beat / total beats (where in the piece)
    pitch contour polyfit
    rhythmic contour polyfit
    aggregate pitches chord probs
    piece key
    chords before / after strong beat
    dissonance presence
    dissonance location (normalized in beat axis)
'''
    if dataset is None:
        dataset = Dataset()

    if not dataset.attributes:
        dataset.attributes = [
            ('beatInMeasure', 'REAL'),
            ('metricStrength', ['3', '2', '1', '0']),
            ('numberOfNotes', 'INTEGER'),
            ('duration', 'REAL'),
            ('locationInPiece', 'REAL'),
            ('pitchX2', 'REAL'),
            ('pitchX1', 'REAL'),
            ('pitchX0', 'REAL'),
            ('pitchContourX2', 'REAL'),
            ('pitchContourX1', 'REAL'),
            ('pitchContourX0', 'REAL'),
            ('rhythmDrops', ['TRUE', 'FALSE']),
            ('rhythmRises', ['TRUE', 'FALSE']),
            ('rhythmContourX2', 'REAL'),
            ('rhythmContourX1', 'REAL'),
            ('rhythmContourX0', 'REAL'),
            ('locationStrongestNote', 'REAL'),
            ('pieceKey', 'INTEGER'),  # ['-5', '-4', '-3', '-2', '-1', '0', '1', '2', '3', '4', '5']
            ('pieceMode', ['Major', 'Minor']),
            ('probChord_I', 'REAL'),
            ('probChord_II', 'REAL'),
            ('probChord_III', 'REAL'),
            ('probChord_IV', 'REAL'),
            ('probChord_V', 'REAL'),
            ('probChord_VI', 'REAL'),
            ('probChord_VII', 'REAL'),
            ('initalChord', ['0', '1', '2', '3', '4', '5', '6']),
            ('finalChord', ['0', '1', '2', '3', '4', '5', '6']),
            ('hasDissonance', ['TRUE', 'FALSE']),
            ('dissonanceLocation', 'REAL'),
            ('isSoloPiece', ['TRUE', 'FALSE']),
            ('pieceId', 'INTEGER'),
            ('motifId', 'INTEGER'),
            ('dynamicsX2', 'REAL'),
            ('dynamicsX1', 'REAL'),
            ('dynamicsX0', 'REAL')
            # ('timingX2', 'REAL'),
            # ('timingX1', 'REAL'),
            # ('timingX0', 'REAL')
        ]

    # 'f, i, i, f, f, f, f, f, f, f, f, f, f, f, i, U5, f, f, f, f, f, f, f, i, i, U5, f, f, f, f, f, f, f'
    motifs = groupMotifs(piece.parts[instrument])

    print('piece={p}, {m} motifs'.format(p=piece.name, m=len(motifs)))

    vals = np.zeros(shape=(len(motifs), len(dataset.attributes)), dtype=object)
    allLevels = [[], '', -1] * len(motifs)

    startInd = 0
    totalBeats = piece.parts[instrument][-1].startBeat + piece.parts[instrument][-1].durBeats
    for i in range(0, len(motifs)):
        sb = piece.parts[instrument][startInd].startBeat
        motif = piece.parts[instrument][startInd:motifs[i]]
        x = np.linspace(0, 1, len(motif))

        idx = 0
        # beatInMeasure:
        vals[i, idx] = sb % 4
        idx = idx + 1

        # metricStrength:
        vals[i, idx] = metricStrength(motif[0])
        idx = idx + 1

        # numberOfNotes:
        vals[i, idx] = motifs[i] - startInd
        idx = idx + 1

        # duration:
        if motifs[i] < len(piece.parts[instrument]):
            vals[i, idx] = piece.parts[instrument][motifs[i]].startBeat - sb
        else:
            last = piece.parts[instrument][motifs[i] - 1]
            vals[i, idx] = last.startBeat + last.durBeats - sb
        idx = idx + 1

        # locationInPiece:
        vals[i, idx] = sb / totalBeats
        idx = idx + 1

        # pitch curve coefficients:
        pitches = np.array([n.pitch for n in motif])
        coeff = np.polyfit(x, pitches - min(pitches), 2 if len(x) > 2 else 1)
        vals[i, idx:idx + 3] = coeff if len(x) > 2 else [0, coeff[0], coeff[1]]
        idx = idx + 3

        pContour = pitchContour(motif)
        coeff = np.polyfit(x, pContour, 2 if len(x) > 2 else 1)
        vals[i, idx:idx + 3] = coeff if len(x) > 2 else [0, coeff[0], coeff[1]]
        idx = idx + 3

        # rhythmic descriptors
        rhContour = np.array([n.ioiBeats / n.prevNote.ioiBeats if n.prevNote is not None else 1 for n in motif])
        vals[i, idx] = 'TRUE' if max(rhContour) > 1 else 'FALSE'  # rhythm drops
        idx = idx + 1
        vals[i, idx] = 'TRUE' if min(rhContour) < 1 else 'FALSE'  # rhythm rises
        idx = idx + 1

        # rhythmic contour coefficients
        coeff = np.polyfit(x, rhContour, 2 if len(x) > 2 else 1)
        vals[i, idx:idx + 3] = coeff if len(x) > 2 else [0, coeff[0], coeff[1]]
        idx = idx + 3

        # locationStrongestNote
        ms = [''] * len(motif)
        for j, nt in enumerate(motif):
            ms[j] = metricStrength(nt)
        vals[i, idx] = (motif[np.argmax(ms)].startBeat - sb) / (motif[-1].startBeat - sb)
        idx = idx + 1

        # piece key and mode
        vals[i, idx] = keyInFifths(piece.key)
        idx = idx + 1
        vals[i, idx] = 'Major' if piece.mode == Mode.major else 'Minor'
        idx = idx + 1

        # chord probabilities
        _, probs = estimateChord(np.asarray([n.allPitches for n in motif]).flatten(), piece.key, piece.mode)
        vals[i, idx:idx + 7] = probs
        idx = idx + 7

        # initial / final chord
        ch, _ = estimateChord(motif[0].allPitches, piece.key, piece.mode)
        vals[i, idx] = str(ch)
        idx = idx + 1
        ch, _ = estimateChord(motif[-1].allPitches, piece.key, piece.mode)
        vals[i, idx] = str(ch)
        idx = idx + 1

        # hasDissonance / dissonanceLocation
        vals[i, idx] = 'FALSE'
        for j in range(0, len(motif)):
            if isDissonance(motif[j].pitch, piece.key, piece.mode):
                vals[i, idx] = 'TRUE'
                vals[i, idx + 1] = x[j]
                break
        idx = idx + 2

        # isSoloPiece
        vals[i, idx] = 'FALSE' if len(piece.parts) > 1 else 'TRUE'
        idx = idx + 1

        # piece/motif ID
        vals[i, idx] = int(piece.name.split('.')[0])
        idx = idx + 1
        vals[i, idx] = i
        idx = idx + 1

        # dynamics curve coefficients
        levels = []
        for n in motif:
            levels += list(n.levels)
        if len(levels) > 2:
            vals[i, idx:idx + 3] = np.polyfit(np.linspace(0, 1, len(levels)), levels, 2)
        elif len(levels) > 1:
            vals[i, idx] = 0
            vals[i, idx + 1:idx + 3] = np.polyfit(np.linspace(0, 1, len(levels)), levels, 1)
        elif len(levels) == 1:
            vals[i, idx:idx + 2] = 0
            vals[i, idx + 2] = levels[0]
        else:
            vals[i, idx:idx + 2] = 0
            vals[i, idx + 2] = - np.inf
        idx = idx + 3

        # timing deviation curve descriptors (TODO)

        # storing levels for error calculation
        allLevels[i] = [levels, int(piece.name.split('.')[0]), i]

        # update start index
        startInd = motifs[i]

    if dataset.valsArray is not None:
        print('vals array not none')
        print(dataset.valsArray)
        dataset.valsArray = np.append(dataset.valsArray, vals, axis=0)
    else:
        dataset.valsArray = vals

    if dataset.moreVals is None:
        dataset.moreVals = allLevels
    else:
        dataset.moreVals += allLevels

    return dataset
