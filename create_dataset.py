import h5py
import numpy as np
import logging
from simfile.dir import SimfilePack
from simfile.notes import NoteData, NoteType
from simfile.notes.count import count_grouped_notes
from simfile.notes.group import group_notes, SameBeatNotes
from simfile.timing import TimingData
from simfile.timing.engine import TimingEngine
from pathlib import Path
from itertools import islice


def get_song(collection_name: str,
             pack_index: int = 1,
             song_index: int = 1):
    overcollection = Path('raw_data') / collection_name
    folders = [SimfilePack(f) for f in overcollection.iterdir() if f.is_dir()]

    simfile_pack = folders[pack_index]
    song = next(islice(simfile_pack.simfiles(), song_index, None))
    return song

notetypes = {NoteType.TAP, NoteType.HOLD_HEAD, NoteType.ROLL_HEAD, NoteType.TAIL}

def iterate_chart(chart):
   """Used for other stuff."""
   return group_notes(NoteData(chart),
                       include_note_types=notetypes,
                       same_beat_notes=SameBeatNotes.JOIN_ALL,
                       join_heads_to_tails=True)

def produce_tokens(chart, timing_engine=None):
  assert chart.stepstype == 'dance-single'
  for notegroup in iterate_chart(chart):
    beat = notegroup[0].beat
    cols = [n.column for n in notegroup]
    col_arr = tuple(1 if (i in cols) else 0 for i in range(4))
    if timing_engine is not None:  # include the times
        time = timing_engine.time_at(beat)
        yield (beat, time, *col_arr)  # Beat, Decimal, 0/1, 0/1, 0/1, 0/1
    else:
       yield (beat, 0, *col_arr)

def process_simfile(simfile) -> dict:
    # We're gonna make one giant numpy array for each song with difficulties and all that.
    ds = dict()
    charts = [chart for chart in simfile.charts if chart.stepstype == 'dance-single']
    chart_sizes = [count_grouped_notes(iterate_chart(c)) for c in charts]
    dtypes = [
      ('Beat', np.float16),
      ('Time', np.float16),
      ('c0', bool),
      ('c1', bool),
      ('c2', bool),
      ('c3', bool)]
   
    for i, chart in enumerate(charts):
        output = np.empty(chart_sizes[i], dtype=dtypes)
        timing_engine = TimingEngine(TimingData(simfile, chart))
        l = list(produce_tokens(chart, timing_engine))
        output[:] = l
        ds[chart.meter] = output
    return ds

if __name__ == "__main__":
    dataset_file = 'data.hdf5'
    datasets = ['fraxtil', 'gpop', 'in-the-groove', 'otakus-dream']
    dataset_dir = Path('raw_data') / datasets[0]
    folders = [SimfilePack(f) for f in dataset_dir.iterdir() if f.is_dir()]
    for folder in folders:
        simfile_pack = folder
        for song in simfile_pack.simfiles():
            h5path = f"/{dataset_dir.name}/{folder.name}/{song.title}"
            with h5py.File(dataset_file, 'a') as h5file:
                h5file.create_group(h5path)
                h5file[h5path].attrs['title'] = song.title
            songdata = process_simfile(song)
            logging.info(f"Saving {song.title} to {h5path}")
            with h5py.File(dataset_file, 'a') as h5file:
                for k, v in songdata.items():
                  h5file[f"{h5path}/{k}"] = v
                  h5file[f"{h5path}/{k}"].attrs['title'] = song.title
                  h5file[f"{h5path}/{k}"].attrs['difficulty'] = int(k)